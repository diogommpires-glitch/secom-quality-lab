"""Paired future-speed diagnostics using development-training rows only.

This does not select models, read held-out images, control the Pod or change the
frozen study. A separate operator must reserve the GPU and enforce its lease.
"""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

SCOPE = "development_training_only_performance_not_selection"
RESOLUTIONS = ((704, 256), (1056, 384))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dependencies():
    import numpy as np
    import psutil
    import torch
    from safetensors import safe_open
    from threadpoolctl import threadpool_limits

    from secom_quality.vision import models
    from secom_quality.vision.data import load_development
    from secom_quality.vision.protocol import checked_protocol

    return SimpleNamespace(
        np=np,
        torch=torch,
        process=psutil.Process(),
        safe_open=safe_open,
        threadpool_limits=threadpool_limits,
        models=models,
        load_development=load_development,
        checked_protocol=checked_protocol,
    )


def verify_sources(current_models, future_path):
    current_path = Path(current_models.__file__).resolve()
    future_path = Path(future_path).resolve()
    if future_path.parts[-4:] != ("src", "secom_quality", "vision", "models.py"):
        raise ValueError("Future models must belong to a verified extracted source package")
    package = future_path.parents[3]
    manifest_path = package / "handoff-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "secom-runpod-source-v1":
        raise ValueError("Unsupported future source manifest")
    members = manifest.get("files", {})
    if members.get("src/secom_quality/vision/models.py") != file_hash(future_path):
        raise ValueError("Future models checksum differs from the handoff manifest")
    scientific_dependencies = {}
    prefix = "src/secom_quality/vision/"
    for name, expected in members.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe future source member")
        path = (package / relative).resolve()
        if not path.is_relative_to(package) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Future source member checksum differs: {name}")
        if name == prefix + "transforms.py" or name.startswith(prefix + "vendor/"):
            current = current_path.parent / name.removeprefix(prefix)
            if not current.is_file() or file_hash(current) != expected:
                raise ValueError(f"Future/current transform or vendor source differs: {name}")
            scientific_dependencies[name] = expected
    required_shared = {
        prefix + "transforms.py",
        prefix + "vendor/loss.py",
        prefix + "vendor/supersimplenet.py",
    }
    if not required_shared.issubset(scientific_dependencies):
        raise ValueError("Future source package lacks the shared transform/vendor proof")
    record_name = manifest.get("cloud_optimizations")
    if record_name not in members:
        raise ValueError("Future source package lacks its optimization provenance")
    record = json.loads((package / record_name).read_text())
    current_sha, future_sha = file_hash(current_path), file_hash(future_path)
    priority, prefetch = record.get("future_priority_reservoir"), record.get("future_ssn_prefetch")
    if (
        not isinstance(priority, dict)
        or not isinstance(prefetch, dict)
        or current_sha == future_sha
    ):
        raise ValueError("Future package must include both isolated optimization candidates")
    if (
        record.get("optimized_models_sha256") != future_sha
        or priority.get("cache_only_models_sha256") != current_sha
    ):
        raise ValueError("Future provenance does not cover this paired current/future model source")
    for item, name in (
        (priority, "migration/priority-reservoir-candidate.py"),
        (prefetch, "migration/ordered-prefetch-candidate.py"),
    ):
        if name not in members or item.get("candidate_source_sha256") != members[name]:
            raise ValueError("Future helper source provenance differs")
    current_linux_lock = current_path.parents[3] / "configs/runpod-requirements.lock"
    future_linux_lock = package / "configs/runpod-requirements.lock"
    if "configs/runpod-requirements.lock" not in members or file_hash(
        current_linux_lock
    ) != file_hash(future_linux_lock):
        raise ValueError("Current/future pinned Linux dependency locks differ")
    return {
        "runtime_linux_lock_sha256": file_hash(current_linux_lock),
        "future_linux_lock_sha256": file_hash(future_linux_lock),
        "runtime_dependency_policy": "Execute both modules with the current pinned Linux environment; no future canonical Windows lock was installed or overlaid",
        "future_canonical_vision_lock_sha256": members.get("configs/vision-requirements.lock"),
        "current_models_path": str(current_path),
        "current_models_sha256": file_hash(current_path),
        "future_models_path": str(future_path),
        "future_models_sha256": file_hash(future_path),
        "future_manifest_sha256": file_hash(manifest_path),
        "shared_transform_vendor_sha256": scientific_dependencies,
        "future_optimization_record": record_name,
        "future_optimization_record_sha256": file_hash(package / record_name),
        "paired_cache_only_models_sha256": priority["cache_only_models_sha256"],
        "candidate_source_sha256": {
            "priority_reservoir": priority["candidate_source_sha256"],
            "ordered_prefetch": prefetch["candidate_source_sha256"],
        },
    }


def load_future(path):
    name = "secom_quality.vision.future_speed_models"
    if name in sys.modules:
        raise RuntimeError("Future model module already loaded; use a fresh pilot process")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "secom_quality.vision"
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cpu_weights(predictor):
    return {
        key: value.detach().cpu().clone() for key, value in predictor.network.state_dict().items()
    }


def weight_difference(first, second, runtime):
    if first.keys() != second.keys() or any(
        first[name].shape != second[name].shape for name in first
    ):
        raise ValueError("Short diagnostic model-state names or shapes differ")
    maximum, changed, total = 0.0, 0, 0
    integer_equal = True
    for name, value in first.items():
        other = second[name]
        if value.is_floating_point():
            if not value.isfinite().all() or not other.isfinite().all():
                raise FloatingPointError("Short diagnostic model weights are nonfinite")
            maximum = max(maximum, float((value.double() - other.double()).abs().max()))
            changed += int((value != other).count_nonzero())
            total += value.numel()
        else:
            integer_equal = integer_equal and runtime.torch.equal(value, other)
    return {
        "floating_max_abs_difference": maximum,
        "floating_values_changed": changed,
        "floating_values_compared": total,
        "integer_buffers_equal": integer_equal,
        "interpretation": "Descriptive only; adaptive maxpool CUDA backward prevents a cross-fit bitwise-weight claim",
    }


def rng_hash(runtime):
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode())
    kind, values, position, gaussian, cached = runtime.np.random.get_state()
    digest.update(kind.encode())
    digest.update(values.tobytes())
    digest.update(repr((position, gaussian, cached)).encode())
    digest.update(bytes.fromhex(tensor_hash(runtime.torch.get_rng_state())))
    for state in runtime.torch.cuda.get_rng_state_all():
        digest.update(bytes.fromhex(tensor_hash(state)))
    return digest.hexdigest()


def checkpoint_rng_hash(directory, runtime):
    metadata = json.loads((directory / "checkpoint.json").read_text())
    tensor_path = directory / "checkpoint.safetensors"
    if file_hash(tensor_path) != metadata["safetensors_sha256"]:
        raise ValueError("Diagnostic checkpoint pair is incomplete or corrupt")
    encoded = dict(metadata["state"]["__dict__"])
    keys = ("numpy_rng", "torch_rng", "cuda_rng", "python_rng", "numpy_global_rng", "frequencies")
    with runtime.safe_open(tensor_path, framework="pt", device="cpu") as tensors:

        def replace(value):
            if isinstance(value, dict):
                if "__tensor__" in value:
                    return {"tensor_sha256": tensor_hash(tensors.get_tensor(value["__tensor__"]))}
                return {key: replace(item) for key, item in value.items()}
            if isinstance(value, list):
                return [replace(item) for item in value]
            return value

        selected = {key: replace(encoded[key]) for key in keys}
    return hashlib.sha256(
        json.dumps(selected, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def pinned_memory_stats(cuda):
    api = getattr(cuda, "host_memory_stats", None)
    if not callable(api):
        api = getattr(getattr(cuda, "memory", None), "host_memory_stats", None)
    if not callable(api):
        return {
            "status": "not_measured",
            "reason": "No supported host-pinned allocator statistics API",
        }
    try:
        counters = {
            str(key): value
            for key, value in api().items()
            if isinstance(value, (int, float)) and math.isfinite(value)
        }
    except (RuntimeError, OSError, TypeError, AttributeError):
        return {"status": "not_measured", "reason": "Host-pinned allocator counters unavailable"}
    return {
        "status": "measured",
        "scope": "Cumulative process allocator counters after this fit, not a reset isolated peak",
        "counters": counters,
    }


def fit(runtime, module, config, train, seed, directory, *, audit=False):
    cuda = runtime.torch.cuda
    cuda.synchronize(0)
    cuda.empty_cache()
    cuda.reset_peak_memory_stats(0)
    checkpoint_seconds = []
    original_save, original_sample = module._save_checkpoint, module._training_sample
    sample_digest, sample_calls, lock = hashlib.sha256(), [0], threading.Lock()

    def timed_save(*args, **kwargs):
        started = time.perf_counter()
        result = original_save(*args, **kwargs)
        checkpoint_seconds.append(time.perf_counter() - started)
        return result

    def audited_sample(row, resolution, flip):
        result = original_sample(row, resolution, flip)
        header = json.dumps([str(row.record_id), list(resolution), int(flip)]).encode()
        with lock:
            sample_digest.update(header)
            for value in result:
                sample_digest.update(str(value.dtype).encode())
                sample_digest.update(json.dumps(list(value.shape)).encode())
                sample_digest.update(value.tobytes(order="C"))
            sample_calls[0] += 1
        return result

    module._save_checkpoint = timed_save
    if audit:
        module._training_sample = audited_sample
    stop, rss_samples = threading.Event(), [runtime.process.memory_info().rss]

    def sample_rss():
        while not stop.is_set():
            rss_samples.append(runtime.process.memory_info().rss)
            stop.wait(0.2)

    sampler = threading.Thread(target=sample_rss, name="speed-pilot-rss", daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        predictor = module.fit_model(config, train, seed, directory, device="cuda:0")
        cuda.synchronize(0)
        elapsed = time.perf_counter() - started
        peak_allocated, peak_reserved = cuda.max_memory_allocated(0), cuda.max_memory_reserved(0)
    finally:
        stop.set()
        sampler.join()
        module._save_checkpoint, module._training_sample = original_save, original_sample
    if any(
        value.is_floating_point() and not value.isfinite().all()
        for value in predictor.network.state_dict().values()
    ):
        raise FloatingPointError("Diagnostic fitted model weights are nonfinite")
    record = {
        "config": config,
        "training_rows": len(train),
        "seed": seed,
        "fit_wall_seconds_including_setup_all_checkpoints_and_final_save": elapsed,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "process_rss_end_bytes": runtime.process.memory_info().rss,
        "sampled_process_rss_peak_bytes": max(rss_samples),
        "rss_sampling_interval_seconds": 0.2,
        "process_rss_note": "Process RSS includes both modules and retained preprocessing caches, not isolated model memory",
        "checkpoint_wall_seconds": checkpoint_seconds,
        "host_pinned_memory_statistics": pinned_memory_stats(cuda),
    }
    if config["family"] == "supersimplenet":
        history = json.loads((directory / "training_history.json").read_text())
        if len(history) != config["epochs"] or any(
            row["epoch"] != index + 1
            or row.get("epochs") != config["epochs"]
            or row.get("examples") != 8 * int((train.label == 1).sum())
            or not math.isfinite(row["mean_loss"])
            or not math.isfinite(row["seconds"])
            or row["seconds"] <= 0
            for index, row in enumerate(history)
        ):
            raise ValueError("Short diagnostic training history is incomplete or nonfinite")
        if predictor.training.get("batch_size") != 8:
            raise ValueError("Diagnostic SSN batch size differs from the frozen recipe")
        record["history"] = history
        record["end_global_rng_sha256"] = rng_hash(runtime)
        record["checkpoint_rng_and_frequencies_sha256"] = checkpoint_rng_hash(directory, runtime)
        if audit:
            if sample_calls[0] != 8 * int((train.label == 1).sum()):
                raise ValueError("Short audit sample call count differs from the frozen recipe")
            record.update(
                sample_call_count=sample_calls[0],
                ordered_sample_bytes_sha256=sample_digest.hexdigest(),
            )
        if len(history) >= 12:
            if len(checkpoint_seconds) != len(history):
                raise ValueError("Diagnostic checkpoint timing count differs from epochs")
            record.update(
                warm_10_epoch_numbers=list(range(3, 13)),
                warm_10_training_seconds=[row["seconds"] for row in history[2:12]],
                warm_10_checkpoint_seconds=checkpoint_seconds[2:12],
                warm_10_training_plus_checkpoint_seconds=sum(
                    row["seconds"] for row in history[2:12]
                )
                + sum(checkpoint_seconds[2:12]),
                warm_note="Inputs prewarmed separately; first two epochs are kernel/model warmup, not cold data epochs",
            )
    atomic_json(directory / "performance.json", record)
    return predictor, record


def prewarm(module, train, resolution, runtime):
    if not hasattr(module, "_TRAINING_SAMPLE_CACHE_BYTES"):
        raise ValueError("Both warm profiles need the reviewed deterministic preprocessing cache")
    before = rng_hash(runtime)
    started = time.perf_counter()
    for row in train.itertuples():
        module._training_sample(row, resolution, 0)
    result = {
        "seconds": time.perf_counter() - started,
        "training_rows": len(train),
        "resolution": list(resolution),
        "cache_bytes_after": module._TRAINING_SAMPLE_CACHE_BYTES,
        "global_rng_unchanged": before == rng_hash(runtime),
    }
    if not result["global_rng_unchanged"]:
        raise ValueError("Deterministic prewarming consumed global RNG state")
    return result


def checkpoint_roundtrip(module, predictor, directory, seed, runtime):
    before = rng_hash(runtime)
    model = copy.deepcopy(predictor.network)
    optimizer, scheduler = model.get_optimizers()
    local_rng = runtime.np.random.default_rng(seed + 1)
    epoch, frequencies = module._restore_checkpoint(
        directory, model, optimizer, scheduler, local_rng
    )
    original = predictor.network.state_dict()
    if any(
        not runtime.torch.equal(value, model.state_dict()[name]) for name, value in original.items()
    ):
        raise ValueError("CUDA checkpoint model restoration differs")
    output = directory / "self-roundtrip"
    module._save_checkpoint(output, model, optimizer, scheduler, epoch, local_rng, frequencies)
    first = json.loads((directory / "checkpoint.json").read_text())["state"]
    second = json.loads((output / "checkpoint.json").read_text())["state"]
    if first != second:
        raise ValueError("Checkpoint optimizer/scheduler/RNG metadata differs after restoration")
    with (
        runtime.safe_open(
            directory / "checkpoint.safetensors", framework="pt", device="cpu"
        ) as first_tensors,
        runtime.safe_open(
            output / "checkpoint.safetensors", framework="pt", device="cpu"
        ) as second_tensors,
    ):
        keys = set(first_tensors.keys())
        if keys != set(second_tensors.keys()) or any(
            not runtime.torch.equal(first_tensors.get_tensor(key), second_tensors.get_tensor(key))
            for key in sorted(keys)
        ):
            raise ValueError("Checkpoint optimizer/model/RNG tensors differ after restoration")
    if rng_hash(runtime) != before:
        raise ValueError("Checkpoint self-roundtrip changed end-of-fit global RNG")
    del model, optimizer, scheduler
    return {"state": "passed", "epoch": epoch, "model_optimizer_scheduler_rng_tensors_exact": True}


def run(workspace, future_models, output, epochs=12, seed=20261026):
    workspace, future_models, output = (
        Path(path).resolve() for path in (workspace, future_models, output)
    )
    if epochs < 12:
        raise ValueError("At least twelve epochs are needed for two warmup plus ten warm epochs")
    if output == workspace or any(
        output.is_relative_to(workspace / "vision" / name)
        for name in (
            "research",
            "protected",
            "images",
            "data",
            "source",
            "ad2",
            "ad2_downloads",
            "bundles",
            "reports",
        )
    ):
        raise ValueError(
            "Diagnostic output must stay outside frozen scientific artifact/data directories"
        )
    if future_models.parts[-4:] == (
        "src",
        "secom_quality",
        "vision",
        "models.py",
    ) and output.is_relative_to(future_models.parents[3]):
        raise ValueError("Diagnostic output must not modify the verified future source package")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": SCOPE,
        "state": "running",
        "seed": seed,
        "epochs": epochs,
        "heldout_or_final_images_used": False,
        "stages": [],
    }

    def progress(stage, **extra):
        atomic_json(output / "report.partial.json", report)
        atomic_json(
            output / "progress.json",
            {
                "scope": SCOPE,
                "state": "running",
                "stage": stage,
                "updated_unix": time.time(),
                **extra,
            },
        )

    try:
        progress("preflight")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "4"
        runtime = dependencies()
        if not runtime.torch.cuda.is_available() or not runtime.torch.cuda.is_bf16_supported():
            raise RuntimeError("This isolated pilot needs the reviewed CUDA0/BF16 environment")
        runtime.torch.cuda.set_device(0)
        runtime.torch.cuda.synchronize(0)
        runtime.torch.set_num_threads(4)
        report["sources"] = verify_sources(runtime.models, future_models)
        future = load_future(future_models)
        report["environment"] = {
            "python": platform.python_version(),
            "gpu": runtime.torch.cuda.get_device_name(0),
            "gpu_vram_bytes": runtime.torch.cuda.get_device_properties(0).total_memory,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "torchvision", "safetensors")
            },
            "numerical_threads": 4,
        }
        protocol = runtime.checked_protocol(workspace)
        if seed != protocol["seeds"][0]:
            raise ValueError("Pilot seed differs from the frozen reference seed")
        development = runtime.load_development(workspace)
        train = development.loc[development.fold != 0].copy()
        if (
            len(train) != 1554
            or int((train.label == 1).sum()) != 164
            or not set(train.label).issubset({0, 1})
        ):
            raise ValueError("Frozen fold0 training cohort count/label contract differs")
        report.update(
            protocol_hash=protocol["protocol_hash"],
            training_rows=len(train),
            positive_training_rows=164,
            training_filter="development.fold !=0",
            timing_note="Paired diagnostics with ordered profiles, not replicated performance estimates or model-selection evidence",
        )
        variants = {"original": runtime.models, "future": future}
        short = train.sort_values("record_id").groupby("label", sort=True).head(16).copy()
        pc = next(
            dict(config)
            for config in protocol["candidate_configs"]
            if config["family"] == "patchcore"
        )
        if pc.get("resolution") != [704, 256] or pc.get("coreset_fraction") != 0.01:
            raise ValueError("Expected fixed first1percent704 PatchCore diagnostic")
        with runtime.threadpool_limits(limits=4):
            reference = None
            pc_records = {}
            image = Path(train.loc[train.label == 0].iloc[0].image_path)
            for name in ("original", "future"):
                progress(
                    "paired_patchcore",
                    profile=name,
                    fit_directory=str(output / f"patchcore-{name}"),
                )
                predictor, record = fit(
                    runtime, variants[name], pc, train, seed, output / f"patchcore-{name}"
                )
                scores, maps = predictor.predict([image])
                if not runtime.np.isfinite(scores).all() or not runtime.np.isfinite(maps[0]).all():
                    raise FloatingPointError(
                        "Training-image PatchCore diagnostic output is nonfinite"
                    )
                memory = predictor.memory.detach().cpu().clone()
                weights = {
                    key: value.detach().cpu().clone()
                    for key, value in predictor.network.state_dict().items()
                }
                record["memory_sha256"] = tensor_hash(memory)
                if reference is None:
                    reference = memory, weights, scores.copy(), maps[0].copy()
                else:
                    old_memory, old_weights, old_scores, old_map = reference
                    exact = (
                        record["memory_sha256"] == pc_records["original"]["memory_sha256"]
                        and runtime.torch.equal(memory, old_memory)
                        and weights.keys() == old_weights.keys()
                        and all(
                            runtime.torch.equal(value, old_weights[key])
                            for key, value in weights.items()
                        )
                    )
                    record["paired_parity"] = {
                        "memory_and_backbone_exact": exact,
                        "training_score_max_abs_difference": float(
                            runtime.np.max(runtime.np.abs(scores - old_scores))
                        ),
                        "training_map_max_abs_difference": float(
                            runtime.np.max(runtime.np.abs(maps[0] - old_map))
                        ),
                    }
                    record["paired_parity"]["prediction_tolerance"] = {
                        "absolute": 1e-6,
                        "relative": 1e-6,
                    }
                    atomic_json(output / "paired-patchcore-parity.json", record["paired_parity"])
                    if (
                        not exact
                        or not runtime.np.allclose(scores, old_scores, atol=1e-6, rtol=1e-6)
                        or not runtime.np.allclose(maps[0], old_map, atol=1e-6, rtol=1e-6)
                    ):
                        raise ValueError(
                            "Exact PatchCore memory/backbone or numeric prediction parity failed"
                        )
                atomic_json(output / f"patchcore-{name}" / "performance.json", record)
                if name == "future":
                    del old_memory, old_weights, old_scores, old_map
                pc_records[name] = record
                del predictor, memory, weights, scores, maps
                gc.collect()
                runtime.torch.cuda.empty_cache()
            report["stages"].append({"stage": "paired_patchcore", "profiles": pc_records})
            del reference
            gc.collect()
            for resolution in RESOLUTIONS:
                audit_records = {}
                audit_reference_weights = None
                numeric_difference = None
                for name in ("original", "future"):
                    progress(
                        "short_data_rng_audit",
                        profile=name,
                        resolution=list(resolution),
                        fit_directory=str(output / f"audit-{resolution[0]}-{name}"),
                    )
                    config = {
                        "config_id": f"performance-audit-{resolution[0]}",
                        "family": "supersimplenet",
                        "resolution": list(resolution),
                        "lr_multiplier": 1.0,
                        "epochs": 1,
                    }
                    predictor, record = fit(
                        runtime,
                        variants[name],
                        config,
                        short,
                        seed,
                        output / f"audit-{resolution[0]}-{name}",
                        audit=True,
                    )
                    current_weights = cpu_weights(predictor)
                    if audit_reference_weights is None:
                        audit_reference_weights = current_weights
                    else:
                        numeric_difference = weight_difference(
                            audit_reference_weights, current_weights, runtime
                        )
                        atomic_json(
                            output / f"audit-{resolution[0]}-weight-difference.json",
                            numeric_difference,
                        )
                    audit_records[name] = record
                    del predictor, current_weights
                    gc.collect()
                    runtime.torch.cuda.empty_cache()
                del audit_reference_weights
                keys = (
                    "sample_call_count",
                    "ordered_sample_bytes_sha256",
                    "end_global_rng_sha256",
                    "checkpoint_rng_and_frequencies_sha256",
                )
                comparison = {
                    key: {
                        "original": audit_records["original"][key],
                        "future": audit_records["future"][key],
                        "matches": audit_records["original"][key] == audit_records["future"][key],
                    }
                    for key in keys
                }
                atomic_json(output / f"audit-{resolution[0]}-parity.json", comparison)
                parity = all(item["matches"] for item in comparison.values())
                if not parity:
                    raise ValueError("Short CUDA diagnostic data/order/RNG parity failed")
                report["stages"].append(
                    {
                        "stage": "short_data_rng_audit",
                        "resolution": list(resolution),
                        "exact_data_order_rng": True,
                        "profiles": audit_records,
                        "weights_bitwise_comparison": "Not claimed: adaptive maxpool CUDA backward is nondeterministic",
                        "descriptive_weight_difference": numeric_difference,
                    }
                )
            for resolution in RESOLUTIONS:
                order = ("original", "future") if resolution[0] == 704 else ("future", "original")
                warm_records = {}
                for name in order:
                    progress("prewarm_training_inputs", profile=name, resolution=list(resolution))
                    warmed = prewarm(variants[name], train, resolution, runtime)
                    progress(
                        "warm_ssn_fit",
                        profile=name,
                        resolution=list(resolution),
                        epochs=epochs,
                        fit_directory=str(output / f"warm-{resolution[0]}-{name}"),
                    )
                    config = {
                        "config_id": f"performance-warm-{resolution[0]}",
                        "family": "supersimplenet",
                        "resolution": list(resolution),
                        "lr_multiplier": 1.0,
                        "epochs": epochs,
                    }
                    predictor, record = fit(
                        runtime,
                        variants[name],
                        config,
                        train,
                        seed,
                        output / f"warm-{resolution[0]}-{name}",
                    )
                    record["separate_input_prewarm"] = warmed
                    if name == "future" and resolution[0] == 1056:
                        progress(
                            "cuda_checkpoint_self_roundtrip",
                            profile=name,
                            resolution=list(resolution),
                        )
                        record["checkpoint_self_roundtrip"] = checkpoint_roundtrip(
                            variants[name],
                            predictor,
                            output / f"warm-{resolution[0]}-{name}",
                            seed,
                            runtime,
                        )
                    atomic_json(
                        output / f"warm-{resolution[0]}-{name}" / "performance.json", record
                    )
                    warm_records[name] = record
                    del predictor
                    gc.collect()
                    runtime.torch.cuda.empty_cache()
                report["stages"].append(
                    {
                        "stage": "warm_ssn_fit",
                        "resolution": list(resolution),
                        "execution_order": list(order),
                        "profiles": warm_records,
                    }
                )
        report.update(
            state="completed",
            completed_unix=time.time(),
            interpretation="Measured diagnostics only. No whole-study acceleration, calibrated probability or predictive-performance claim follows from this pilot.",
        )
        atomic_json(output / "report.json", report)
        atomic_json(
            output / "progress.json",
            {"scope": SCOPE, "state": "completed", "updated_unix": time.time()},
        )
        return report
    except BaseException as error:
        state = (
            "interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed"
        )
        report.update(state=state, error_type=type(error).__name__)
        atomic_json(output / "report.partial.json", report)
        atomic_json(
            output / "failure.json",
            {
                "scope": SCOPE,
                "state": state,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        atomic_json(
            output / "progress.json",
            {
                "scope": SCOPE,
                "state": state,
                "updated_unix": time.time(),
                "error_type": type(error).__name__,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--future-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    result = run(args.workspace, args.future_models, args.output, args.epochs, args.seed)
    print(
        json.dumps({"state": result["state"], "scope": result["scope"], "output": str(args.output)})
    )
