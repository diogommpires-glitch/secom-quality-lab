"""Build a cloud-only deterministic preprocessing cache into a verified handoff ZIP."""

import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

MODELS_PATH = "src/secom_quality/vision/models.py"
CACHE_MAX_BYTES = 24 * 1024**3
PRIORITY_SOURCE = Path(__file__).resolve().parents[2] / "experiments/speed/priority_reservoir.py"
PREFETCH_SOURCE = Path(__file__).resolve().parents[2] / "experiments/speed/ordered_prefetch.py"
CACHE_SOURCE = """

# Cloud-only optimization: cache deterministic, unflipped preprocessing only.
# All original sampling, flips, batching, model arithmetic and RNG calls are preserved.
from collections import OrderedDict as _TrainingSampleOrderedDict

_TRAINING_SAMPLE_CACHE_MAX_BYTES = 24 * 1024**3
_TRAINING_SAMPLE_CACHE = _TrainingSampleOrderedDict()
_TRAINING_SAMPLE_CACHE_BYTES = 0


def _training_sample(row: Any, resolution: tuple[int, int], flip: int):
    global _TRAINING_SAMPLE_CACHE_BYTES

    def signature(path):
        path = Path(path).resolve()
        status = path.stat()
        return (str(path), status.st_size, status.st_mtime_ns)

    label = int(row.label)
    key = (
        signature(row.image_path),
        signature(row.mask_path) if label else None,
        label,
        tuple(resolution),
    )
    base = _TRAINING_SAMPLE_CACHE.get(key)
    if base is None:
        base = _training_sample_uncached(row, resolution, 0)
        for array in base:
            array.flags.writeable = False
        size = sum(array.nbytes for array in base)
        if size <= _TRAINING_SAMPLE_CACHE_MAX_BYTES:
            while _TRAINING_SAMPLE_CACHE_BYTES + size > _TRAINING_SAMPLE_CACHE_MAX_BYTES:
                _, evicted = _TRAINING_SAMPLE_CACHE.popitem(last=False)
                _TRAINING_SAMPLE_CACHE_BYTES -= sum(array.nbytes for array in evicted)
            _TRAINING_SAMPLE_CACHE[key] = base
            _TRAINING_SAMPLE_CACHE_BYTES += size
    else:
        _TRAINING_SAMPLE_CACHE.move_to_end(key)

    values, mask, valid, weights = base
    if flip in {1, 3}:
        values, mask, valid, weights = (
            np.flip(values, axis=1),
            np.flip(mask, axis=0),
            np.flip(valid, axis=0),
            np.flip(weights, axis=0),
        )
    if flip in {2, 3}:
        values, mask, valid, weights = (
            np.flip(values, axis=2),
            np.flip(mask, axis=1),
            np.flip(valid, axis=1),
            np.flip(weights, axis=1),
        )
    return tuple(array.copy() for array in (values, mask, valid, weights))
"""


def patch_models(body):
    source = body.decode("utf-8")
    tree = ast.parse(source)
    names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    signature = "def _training_sample(row: Any, resolution: tuple[int, int], flip: int):"
    if names.count("_training_sample") != 1 or "_training_sample_uncached" in names:
        raise ValueError("Expected one original training-sample function; refusing repeated patch")
    if source.count(signature) != 1:
        raise ValueError(
            "Training-sample signature changed; review the optimization before patching"
        )
    source = source.replace(
        signature, signature.replace("_training_sample(", "_training_sample_uncached("), 1
    )
    result = (source.rstrip() + CACHE_SOURCE + "\n").encode("utf-8")
    ast.parse(result)
    return result


def patch_priority_reservoir(body, candidate_body):
    """Inline a reviewed future candidate so the scientific model hash covers it."""
    source = body.decode("utf-8").replace("\r\n", "\n")
    candidate = candidate_body.decode("utf-8").replace("\r\n", "\n")
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fit_patchcore"
    ]
    if len(functions) != 1 or any(
        isinstance(node, ast.ClassDef) and node.name == "PriorityReservoir" for node in tree.body
    ):
        raise ValueError("Expected one original PatchCore fitter without a priority helper")
    candidate_tree = ast.parse(candidate)
    classes = [node for node in candidate_tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1 or classes[0].name != "PriorityReservoir":
        raise ValueError("Expected one self-contained PriorityReservoir candidate")
    for node in candidate_tree.body:
        if isinstance(node, ast.ClassDef) or (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.Import) and all(
            alias.name == "numpy" and alias.asname == "np" for alias in node.names
        ):
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and all(alias.name == "annotations" for alias in node.names)
        ):
            continue
        raise ValueError("Priority helper must be self-contained with only NumPy imports")
    original = ast.get_source_segment(source, functions[0])
    replacements = [
        (
            "    reservoir: NDArray[np.float32] | None = None\n    priorities: NDArray[np.float64] | None = None",
            "    candidate_pool = PriorityReservoir(MAX_CANDIDATE_PATCHES)",
        ),
        (
            "                if reservoir is not None:\n                    assert priorities is not None\n                    chunk = np.concatenate((reservoir, chunk))\n                    keys = np.concatenate((priorities, keys))\n                if len(chunk) > MAX_CANDIDATE_PATCHES:\n                    keep = np.argpartition(keys, -MAX_CANDIDATE_PATCHES)[-MAX_CANDIDATE_PATCHES:]\n                    chunk, keys = chunk[keep], keys[keep]\n                reservoir, priorities = chunk, keys",
            "                candidate_pool.update(chunk, keys)",
        ),
        (
            "    assert reservoir is not None\n    indices = approximate_coreset(",
            "    reservoir = candidate_pool.values()\n    del candidate_pool\n    assert reservoir is not None\n    indices = approximate_coreset(",
        ),
    ]
    modified = original
    for old, new in replacements:
        if modified.count(old) != 1:
            raise ValueError(
                "PatchCore implementation changed; review future optimization before patching"
            )
        modified = modified.replace(old, new, 1)
    if source.count(original) != 1:
        raise ValueError("Ambiguous PatchCore source segment")
    helper = ast.get_source_segment(candidate, classes[0])
    result = (
        source.replace(original, modified, 1).rstrip()
        + "\n\n# Future candidate: exact logical priority reservoir; new protocol required.\n"
        + helper
        + "\n"
    ).encode("utf-8")
    if result.count(b"NDArray") == 1:
        result = result.replace(b"from numpy.typing import NDArray\n", b"", 1)
    ast.parse(result)
    return result


def patch_ssn_prefetch(body, candidate_body):
    """Overlap pure batch preparation without moving RNG/model work off the main thread."""
    source = body.decode("utf-8").replace("\r\n", "\n")
    candidate = candidate_body.decode("utf-8").replace("\r\n", "\n")
    tree = ast.parse(source)
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_fit_ssn"
    ]
    if len(functions) != 1 or any(
        isinstance(node, ast.FunctionDef) and node.name == "ordered_prefetch" for node in tree.body
    ):
        raise ValueError("Expected one original SSN fitter without a prefetch helper")
    candidate_tree = ast.parse(candidate)
    helpers = [node for node in candidate_tree.body if isinstance(node, ast.FunctionDef)]
    if (
        len(helpers) != 1
        or helpers[0].name != "ordered_prefetch"
        or any(not isinstance(node, (ast.FunctionDef, ast.Expr)) for node in candidate_tree.body)
    ):
        raise ValueError("Expected one self-contained ordered_prefetch function")
    original = ast.get_source_segment(source, functions[0])
    old = """        for start in range(0, len(order), batch_size):
            samples, labels = [], []
            for index in order[start : start + batch_size]:
                row, flip = examples[index]
                samples.append(_training_sample(row, predictor.resolution, flip))
                labels.append(float(row.label))
            arrays = [
                torch.from_numpy(np.stack(items)).to(predictor.device)
                for items in zip(*samples, strict=True)
            ]
            images, masks, valid, weights = arrays"""
    new = """        def prepare_batch(indices):
            samples, labels = [], []
            for index in indices:
                row, flip = examples[index]
                samples.append(_training_sample(row, predictor.resolution, flip))
                labels.append(float(row.label))
            host_arrays = [torch.from_numpy(np.stack(items)) for items in zip(*samples, strict=True)]
            if predictor.device.startswith('cuda'):
                host_arrays = [tensor.pin_memory() for tensor in host_arrays]
            return host_arrays, labels

        batches = (order[start : start + batch_size] for start in range(0, len(order), batch_size))
        for host_arrays, labels in ordered_prefetch(batches, prepare_batch, prefetch=2):
            arrays = [tensor.to(predictor.device, non_blocking=predictor.device.startswith('cuda')) for tensor in host_arrays]
            images, masks, valid, weights = arrays"""
    if original.count(old) != 1:
        raise ValueError("SSN batch preparation changed; review future prefetch before patching")
    helper = ast.get_source_segment(candidate, helpers[0])
    modified = original.replace(old, new, 1)
    loop_start = modified.index("        for host_arrays, labels in ordered_prefetch(")
    loop_end = modified.index("        scheduler.step()", loop_start)
    consumer = modified[loop_start:loop_end].replace(
        "for host_arrays, labels in ordered_prefetch(batches, prepare_batch, prefetch=2):",
        "for host_arrays, labels in prepared_batches:",
        1,
    )
    if "for host_arrays, labels in prepared_batches:" not in consumer:
        raise ValueError("Unexpected future consumer loop")
    wrapped = (
        "        prepared_batches = ordered_prefetch(batches, prepare_batch, prefetch=2)\n"
        "        try:\n"
        + "".join(
            "    " + line if line.strip() else line for line in consumer.splitlines(keepends=True)
        )
        + "        finally:\n            prepared_batches.close()\n"
    )
    modified = modified[:loop_start] + wrapped + modified[loop_end:]
    result = (
        source.replace(original, modified, 1).rstrip()
        + "\n\n# Future candidate: one deterministic producer; model/RNG work stays on main thread.\n"
        + helper
        + "\n"
    ).encode("utf-8")
    ast.parse(result)
    return result


def optimize_handoff(
    input_source, output_cloud, *, future_priority_reservoir=False, future_prefetch=False
):
    input_source, output_cloud = Path(input_source).resolve(), Path(output_cloud).resolve()
    if input_source == output_cloud:
        raise ValueError("Cloud output must be separate from the original handoff")
    sidecar = output_cloud.with_suffix(output_cloud.suffix + ".sha256")
    if output_cloud.exists() or sidecar.exists():
        raise FileExistsError("Cloud output or checksum already exists; choose a fresh output path")
    input_sha256 = hashlib.sha256(input_source.read_bytes()).hexdigest()
    with zipfile.ZipFile(input_source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members are not allowed")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe ZIP member name")
        members = {name: archive.read(name) for name in names}
    try:
        manifest = json.loads(members.pop("handoff-manifest.json"))
    except (KeyError, ValueError) as exc:
        raise ValueError("A valid handoff manifest is required") from exc
    if manifest.get("schema") != "secom-runpod-source-v1":
        raise ValueError("Unsupported source handoff schema")
    expected = manifest.get("files")
    actual = {name: hashlib.sha256(body).hexdigest() for name, body in members.items()}
    if expected != actual:
        raise ValueError("Input handoff manifest member hashes do not match")
    if MODELS_PATH not in members or "migration/cloud-optimizations.json" in members:
        raise ValueError(
            "Expected original scientific source without an existing cloud optimization"
        )
    original = members[MODELS_PATH]
    optimized = patch_models(original)
    cache_only_sha256 = hashlib.sha256(optimized).hexdigest()
    priority_body = PRIORITY_SOURCE.read_bytes() if future_priority_reservoir else None
    if priority_body is not None:
        optimized = patch_priority_reservoir(optimized, priority_body)
    prefetch_body = PREFETCH_SOURCE.read_bytes() if future_prefetch else None
    if prefetch_body is not None:
        optimized = patch_ssn_prefetch(optimized, prefetch_body)
    members[MODELS_PATH] = optimized
    record = {
        "schema": "secom-cloud-optimizations-v1",
        "input_archive_sha256": input_sha256,
        "original_models_sha256": hashlib.sha256(original).hexdigest(),
        "optimized_models_sha256": hashlib.sha256(optimized).hexdigest(),
        "optimization": "Byte-bounded LRU cache of deterministic unflipped training preprocessing",
        "default_cache_max_bytes": CACHE_MAX_BYTES,
        "key": "Resolved image/mask path, size, mtime_ns, label and resolution",
        "data_assumption": "Training source is verified and immutable; stat keys also detect ordinary file changes",
        "preserved_algorithm": "Original preprocessing remains intact; wrapper repeats identical flips and returned copies",
        "preserved_rng": "Cache and keys consume no RNG; sampling, shuffle, batch order and Perlin remain unchanged",
        "scope": "Cloud handoff only, before a fresh cloud protocol freeze; Windows source and results preserved separately",
        "cache_lifetime": "One worker process; released when the fit worker exits",
        "scientific_settings": "Unchanged folds, seeds, epochs, batch size, precision, losses, selection and held-out boundaries",
    }
    if priority_body is not None:
        record["future_priority_reservoir"] = {
            "status": "candidate_requires_fresh_workspace_and_protocol",
            "candidate_source_sha256": hashlib.sha256(priority_body).hexdigest(),
            "cache_only_models_sha256": cache_only_sha256,
            "runtime_code_hashed_in": MODELS_PATH,
            "preservation": "Same largest-priority np.argpartition inputs, returned logical order, feature bytes and RNG consumption",
            "speed_claim_scope": "Synthetic CPU reservoir microbenchmark only until full-fold benchmarking",
            "rollout": "Never overwrite a frozen predictor or pool results across protocol hashes",
        }
        members["migration/priority-reservoir-candidate.py"] = priority_body
    if prefetch_body is not None:
        record["future_ssn_prefetch"] = {
            "status": "candidate_requires_GPU_warm_epoch_and_resume_validation",
            "candidate_source_sha256": hashlib.sha256(prefetch_body).hexdigest(),
            "runtime_code_hashed_in": MODELS_PATH,
            "preservation": "Unchanged sampling/batch order, one pure preprocessing producer, RNG/model/optimizer on main thread",
            "bounded_buffers": "Two host batches, CUDA-only pinning and nonblocking same-stream transfers",
            "rollout": "Fresh research workspace/protocol; never patch an active frozen run",
        }
        members["migration/ordered-prefetch-candidate.py"] = prefetch_body
    members["migration/cloud-optimizations.json"] = json.dumps(
        record, indent=2, sort_keys=True
    ).encode()
    manifest["files"] = {name: hashlib.sha256(body).hexdigest() for name, body in members.items()}
    manifest["cloud_optimizations"] = "migration/cloud-optimizations.json"
    members["handoff-manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode()
    output_cloud.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_cloud, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, body)
    digest = hashlib.sha256(output_cloud.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {output_cloud.name}\n", encoding="utf-8")
    return {
        "archive": str(output_cloud),
        "sha256": digest,
        "sha256_file": str(sidecar),
        "files": len(members),
        "bytes": output_cloud.stat().st_size,
        "input_sha256": input_sha256,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, dest="input_source")
    parser.add_argument("--output", type=Path, required=True, dest="output_cloud")
    parser.add_argument(
        "--future-priority-reservoir",
        action="store_true",
        help="Build an isolated future speed candidate; requires a fresh research protocol",
    )
    parser.add_argument(
        "--future-prefetch",
        action="store_true",
        help="Build an isolated future ordered-prefetch candidate; GPU validation required",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            optimize_handoff(
                args.input_source,
                args.output_cloud,
                future_priority_reservoir=args.future_priority_reservoir,
                future_prefetch=args.future_prefetch,
            ),
            indent=2,
        )
    )
