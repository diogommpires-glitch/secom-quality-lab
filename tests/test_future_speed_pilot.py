"""Isolated performance-pilot boundaries, paired identity and interruption evidence."""

import contextlib
import hashlib
import importlib.util
import json
import random
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "future_speed_pilot", ROOT / "scripts/runpod/future_speed_pilot.py"
)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def write_package(tmp_path):
    current = tmp_path / "current/src/secom_quality/vision/models.py"
    future = tmp_path / "future/src/secom_quality/vision/models.py"
    files = {
        "configs/runpod-requirements.lock": b"same pinned Linux dependencies",
        "configs/vision-requirements.lock": b"historical Windows lock not used by this module probe",
        "src/secom_quality/vision/models.py": b"future candidate source",
        "src/secom_quality/vision/transforms.py": b"same transforms",
        "src/secom_quality/vision/vendor/loss.py": b"same loss",
        "src/secom_quality/vision/vendor/supersimplenet.py": b"same model",
        "migration/priority-reservoir-candidate.py": b"priority candidate",
        "migration/ordered-prefetch-candidate.py": b"prefetch candidate",
    }
    current.parent.mkdir(parents=True)
    current.write_bytes(b"current cache-only source")
    for name, body in files.items():
        path = tmp_path / "future" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        if (
            "/transforms.py" in name
            or "/vendor/" in name
            or name == "configs/runpod-requirements.lock"
        ):
            original = tmp_path / "current" / name
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_bytes(body)
    record = {
        "optimized_models_sha256": hashlib.sha256(
            files["src/secom_quality/vision/models.py"]
        ).hexdigest(),
        "future_priority_reservoir": {
            "cache_only_models_sha256": pilot.file_hash(current),
            "candidate_source_sha256": hashlib.sha256(
                files["migration/priority-reservoir-candidate.py"]
            ).hexdigest(),
        },
        "future_ssn_prefetch": {
            "candidate_source_sha256": hashlib.sha256(
                files["migration/ordered-prefetch-candidate.py"]
            ).hexdigest()
        },
    }
    files["migration/cloud-optimizations.json"] = json.dumps(record).encode()
    (tmp_path / "future/migration/cloud-optimizations.json").write_bytes(
        files["migration/cloud-optimizations.json"]
    )
    manifest = {
        "schema": "secom-runpod-source-v1",
        "files": {name: hashlib.sha256(body).hexdigest() for name, body in files.items()},
        "cloud_optimizations": "migration/cloud-optimizations.json",
    }
    (tmp_path / "future/handoff-manifest.json").write_text(json.dumps(manifest))
    return SimpleNamespace(__file__=str(current)), future


def test_verified_source_pair_and_shared_imports(tmp_path):
    current, future = write_package(tmp_path)
    result = pilot.verify_sources(current, future)
    assert result["current_models_sha256"] != result["future_models_sha256"]
    assert len(result["shared_transform_vendor_sha256"]) == 3
    (future.parent / "vendor/loss.py").write_text("tampered")
    with pytest.raises(ValueError, match="checksum differs"):
        pilot.verify_sources(current, future)


def test_current_transform_mismatch_is_rejected(tmp_path):
    current, future = write_package(tmp_path)
    (Path(current.__file__).parent / "transforms.py").write_text("changed")
    with pytest.raises(ValueError, match="transform or vendor"):
        pilot.verify_sources(current, future)


def test_package_without_both_future_candidates_is_rejected(tmp_path):
    current, future = write_package(tmp_path)
    root = future.parents[3]
    record_path = root / "migration/cloud-optimizations.json"
    record = json.loads(record_path.read_text())
    record.pop("future_ssn_prefetch")
    record_path.write_text(json.dumps(record))
    manifest_path = root / "handoff-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["migration/cloud-optimizations.json"] = pilot.file_hash(record_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="both isolated"):
        pilot.verify_sources(current, future)


def test_wrong_paired_baseline_is_rejected(tmp_path):
    current, future = write_package(tmp_path)
    Path(current.__file__).write_text("different cache-only base")
    with pytest.raises(ValueError, match="paired current/future"):
        pilot.verify_sources(current, future)


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    current_source, future_path = write_package(tmp_path)
    calls = []

    class FakeModule:
        def __init__(self, name):
            self.name = name
            self.__file__ = current_source.__file__
            self._TRAINING_SAMPLE_CACHE_BYTES = 0
            self.bad_memory = False
            self.bad_history = False

        def _training_sample(self, row, resolution, flip):
            self._TRAINING_SAMPLE_CACHE_BYTES += 1
            value = np.full((2, 2), int(row.record_id) + flip, dtype=np.float32)
            return value[None].repeat(3, axis=0), value.copy(), value.copy(), value.copy()

        def _save_checkpoint(self, directory, *args):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "checkpoint.json").write_text("{}")

        def fit_model(self, config, train, seed, directory, device):
            assert device == "cuda:0"
            assert not train.image_path.str.contains("heldout").any()
            assert set(train.fold) == {1, 2}
            calls.append(
                {
                    "profile": self.name,
                    "config": config,
                    "rows": len(train),
                    "positives": int((train.label == 1).sum()),
                    "directory": directory.name,
                    "seed": seed,
                }
            )
            directory.mkdir(parents=True, exist_ok=False)
            if config["family"] == "supersimplenet":
                history = []
                for epoch in range(config["epochs"]):
                    if config["epochs"] == 1:
                        for index in range(8 * int((train.label == 1).sum())):
                            self._training_sample(
                                train.iloc[index % len(train)],
                                tuple(config["resolution"]),
                                index % 4,
                            )
                    self._save_checkpoint(directory, None, None, None, epoch + 1)
                    history.append(
                        {
                            "epoch": epoch + 1,
                            "epochs": config["epochs"],
                            "mean_loss": float("nan") if self.bad_history else 0.5,
                            "seconds": 1.0,
                            "examples": 8 * int((train.label == 1).sum()),
                        }
                    )
                (directory / "training_history.json").write_text(json.dumps(history))
            memory = torch.tensor([[2.0 if self.bad_memory else 1.0]])
            return SimpleNamespace(
                memory=memory,
                training={"batch_size": 8},
                network=SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([3.0])}),
                predict=lambda paths: (np.array([0.25]), [np.full((2, 2), 0.5, dtype=np.float32)]),
            )

    original, future = FakeModule("original"), FakeModule("future")
    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        set_device=lambda device: None,
        synchronize=lambda device: None,
        empty_cache=lambda: None,
        reset_peak_memory_stats=lambda device: None,
        max_memory_allocated=lambda device: 1024,
        max_memory_reserved=lambda device: 2048,
        get_device_name=lambda device: "NVIDIA GeForce RTX 5090",
        get_device_properties=lambda device: SimpleNamespace(total_memory=32 * 1024**3),
        get_rng_state_all=lambda: [],
    )
    runtime = SimpleNamespace(
        np=np,
        torch=SimpleNamespace(
            cuda=cuda,
            equal=torch.equal,
            set_num_threads=lambda value: None,
            get_rng_state=torch.get_rng_state,
        ),
        process=SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1000)),
        models=original,
        threadpool_limits=lambda **kwargs: contextlib.nullcontext(),
    )
    train = pd.DataFrame(
        {
            "record_id": np.arange(1554),
            "image_path": [f"training-{i}" for i in range(1554)],
            "label": [*([1] * 164), *([0] * 1390)],
            "fold": [1, 2] * 777,
        }
    )
    heldout = pd.DataFrame(
        {
            "record_id": [9000, 9001, 9002],
            "image_path": ["heldout-never-open"] * 3,
            "label": [0, 1, 1],
            "fold": [0] * 3,
        }
    )
    protocol = {
        "protocol_hash": "frozen-protocol",
        "seeds": [20261026],
        "candidate_configs": [
            {
                "config_id": "patchcore-h256-c0p01",
                "family": "patchcore",
                "resolution": [704, 256],
                "coreset_fraction": 0.01,
            }
        ],
    }
    load_calls = []
    runtime.checked_protocol = lambda root: protocol

    def load(root):
        load_calls.append(root)
        return pd.concat([train, heldout], ignore_index=True)

    runtime.load_development = load
    monkeypatch.setattr(pilot, "dependencies", lambda: runtime)
    monkeypatch.setattr(pilot, "load_future", lambda path: future)
    monkeypatch.setattr(pilot, "rng_hash", lambda runtime: "unchanged-global-rng")
    monkeypatch.setattr(pilot, "checkpoint_rng_hash", lambda path, runtime: "same-checkpoint-rng")
    monkeypatch.setattr(
        pilot,
        "checkpoint_roundtrip",
        lambda *args: {
            "state": "passed",
            "epoch": 12,
            "model_optimizer_scheduler_rng_tensors_exact": True,
        },
    )
    return SimpleNamespace(
        runtime=runtime,
        original=original,
        future=future,
        future_path=future_path,
        calls=calls,
        load_calls=load_calls,
    )


def test_complete_control_flow_uses_only_training_and_clean_warm_measurements(
    tmp_path, fake_runtime
):
    output = tmp_path / "performance"
    result = pilot.run(tmp_path / "study", fake_runtime.future_path, output)
    assert result["state"] == "completed" and result["scope"] == pilot.SCOPE
    assert result["heldout_or_final_images_used"] is False
    assert len(fake_runtime.load_calls) == 1
    calls = fake_runtime.calls
    assert len(calls) == 10
    assert all(
        row["rows"] == 1554 and row["positives"] == 164
        for row in calls
        if "audit" not in row["directory"]
    )
    assert all(
        row["rows"] == 32 and row["positives"] == 16 for row in calls if "audit" in row["directory"]
    )
    assert [row["directory"] for row in calls[-4:]] == [
        "warm-704-original",
        "warm-704-future",
        "warm-1056-future",
        "warm-1056-original",
    ]
    warm = [stage for stage in result["stages"] if stage["stage"] == "warm_ssn_fit"]
    for stage in warm:
        for record in stage["profiles"].values():
            assert record["warm_10_epoch_numbers"] == list(range(3, 13))
            assert len(record["warm_10_checkpoint_seconds"]) == 10
            assert "ordered_sample_bytes_sha256" not in record
            assert record["separate_input_prewarm"]["global_rng_unchanged"]
    assert (output / "paired-patchcore-parity.json").exists()
    assert json.loads((output / "progress.json").read_text())["state"] == "completed"
    assert not [thread for thread in threading.enumerate() if thread.name == "speed-pilot-rss"]


def test_short_rng_mismatch_stops_before_full_timing_and_preserves_evidence(
    tmp_path, fake_runtime, monkeypatch
):
    monkeypatch.setattr(
        pilot,
        "checkpoint_rng_hash",
        lambda path, runtime: "future" if "future" in path.name else "original",
    )
    output = tmp_path / "failed-audit"
    with pytest.raises(ValueError, match="data/order/RNG parity"):
        pilot.run(tmp_path / "study", fake_runtime.future_path, output)
    assert len(fake_runtime.calls) == 4
    comparison = json.loads((output / "audit-704-parity.json").read_text())
    assert not comparison["checkpoint_rng_and_frequencies_sha256"]["matches"]
    assert json.loads((output / "progress.json").read_text())["state"] == "failed"
    assert json.loads((output / "report.partial.json").read_text())["sources"]
    assert not (output / "report.json").exists()


def test_memory_parity_failure_is_recorded(tmp_path, fake_runtime):
    fake_runtime.future.bad_memory = True
    output = tmp_path / "failed-memory"
    with pytest.raises(ValueError, match="memory/backbone"):
        pilot.run(tmp_path / "study", fake_runtime.future_path, output)
    assert len(fake_runtime.calls) == 2
    assert not json.loads((output / "paired-patchcore-parity.json").read_text())[
        "memory_and_backbone_exact"
    ]
    assert (output / "failure.json").exists()


def test_nonfinite_history_fails_without_relabelling_it_as_timing_success(tmp_path, fake_runtime):
    fake_runtime.future.bad_history = True
    output = tmp_path / "bad-history"
    with pytest.raises(ValueError, match="nonfinite"):
        pilot.run(tmp_path / "study", fake_runtime.future_path, output)
    assert len(fake_runtime.calls) == 4
    assert json.loads((output / "progress.json").read_text())["state"] == "failed"


def test_fresh_output_required_and_short_epoch_plan_rejected(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "evidence.json"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        pilot.run(tmp_path / "study", tmp_path / "unused.py", existing)
    assert marker.read_text() == "preserve"
    with pytest.raises(ValueError, match="twelve"):
        pilot.run(tmp_path / "study", tmp_path / "unused.py", tmp_path / "too-short", epochs=11)
    assert not (tmp_path / "too-short").exists()


@pytest.mark.parametrize(
    "folder",
    ["research", "protected", "images", "source", "ad2", "ad2_downloads", "bundles", "reports"],
)
def test_cannot_write_diagnostics_into_scientific_or_data_trees(tmp_path, folder):
    with pytest.raises(ValueError, match="outside frozen"):
        pilot.run(
            tmp_path / "study", tmp_path / "unused.py", tmp_path / "study/vision" / folder / "pilot"
        )


def test_interrupted_protocol_check_preserves_partial_state(tmp_path, fake_runtime):
    def interrupted(root):
        raise KeyboardInterrupt("operator lease ended")

    fake_runtime.runtime.checked_protocol = interrupted
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        pilot.run(tmp_path / "study", fake_runtime.future_path, output)
    assert not fake_runtime.calls
    assert json.loads((output / "failure.json").read_text())["state"] == "interrupted"
    assert json.loads((output / "report.partial.json").read_text())["sources"]


def test_actual_rng_hash_and_prewarm_detect_stochastic_preparation():
    torch = pytest.importorskip("torch")
    runtime = SimpleNamespace(
        np=np,
        torch=SimpleNamespace(
            get_rng_state=torch.get_rng_state, cuda=SimpleNamespace(get_rng_state_all=lambda: [])
        ),
    )
    before = pilot.rng_hash(runtime)
    rows = pd.DataFrame({"record_id": [1], "image_path": ["synthetic"], "label": [1]})
    module = SimpleNamespace(
        _TRAINING_SAMPLE_CACHE_BYTES=0, _training_sample=lambda *_args: np.ones((2, 2))
    )
    assert pilot.prewarm(module, rows, (2, 2), runtime)["global_rng_unchanged"]
    assert pilot.rng_hash(runtime) == before
    saved = random.getstate()
    module._training_sample = lambda *_args: random.random()
    try:
        with pytest.raises(ValueError, match="consumed global RNG"):
            pilot.prewarm(module, rows, (2, 2), runtime)
    finally:
        random.setstate(saved)


def test_actual_checkpoint_rng_and_self_roundtrip_with_cpu_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors import safe_open

    from secom_quality.vision import models

    class Tiny(torch.nn.Linear):
        def get_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
            return optimizer, torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[1], gamma=0.7
            )

    model = Tiny(3, 1)
    optimizer, scheduler = model.get_optimizers()
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    rng = np.random.default_rng(20261026)
    models._save_checkpoint(tmp_path, model, optimizer, scheduler, 1, rng, np.array([1.0, 2.0]))
    runtime = SimpleNamespace(np=np, torch=torch, safe_open=safe_open)
    before = pilot.checkpoint_rng_hash(tmp_path, runtime)
    result = pilot.checkpoint_roundtrip(
        models, SimpleNamespace(network=model), tmp_path, 20261026, runtime
    )
    assert result["state"] == "passed" and result["epoch"] == 1
    assert pilot.checkpoint_rng_hash(tmp_path / "self-roundtrip", runtime) == before
    with (tmp_path / "checkpoint.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="incomplete or corrupt"):
        pilot.checkpoint_rng_hash(tmp_path, runtime)


def test_mismatched_runtime_linux_lock_is_rejected(tmp_path):
    current, future = write_package(tmp_path)
    lock = Path(current.__file__).parents[3] / "configs/runpod-requirements.lock"
    lock.write_text("different Linux environment")
    with pytest.raises(ValueError, match="Linux dependency locks"):
        pilot.verify_sources(current, future)


def test_host_pinned_memory_counters_are_measured_or_explicitly_unknown():
    assert pilot.pinned_memory_stats(SimpleNamespace())["status"] == "not_measured"
    cuda = SimpleNamespace(
        memory=SimpleNamespace(
            host_memory_stats=lambda: {
                "allocated_bytes.peak": 123,
                "unknown": object(),
                "nonfinite": float("nan"),
            }
        )
    )
    record = pilot.pinned_memory_stats(cuda)
    assert record["status"] == "measured"
    assert record["counters"] == {"allocated_bytes.peak": 123}


def test_weight_difference_is_descriptive_and_rejects_nonfinite_state():
    torch = pytest.importorskip("torch")
    runtime = SimpleNamespace(torch=torch)
    first = {"weight": torch.tensor([1.0, 2.0]), "counter": torch.tensor(3)}
    second = {"weight": torch.tensor([1.0, 2.25]), "counter": torch.tensor(3)}
    report = pilot.weight_difference(first, second, runtime)
    assert report["floating_max_abs_difference"] == 0.25
    assert report["floating_values_changed"] == 1
    assert report["integer_buffers_equal"]
    second["weight"][0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        pilot.weight_difference(first, second, runtime)
