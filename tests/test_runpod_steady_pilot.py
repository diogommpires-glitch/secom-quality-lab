"""Offline checks for the real-training-fold steady pilot; no CUDA fitting is run."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "runpod_steady_pilot", Path(__file__).parents[1] / "scripts/runpod/steady_pilot.py"
)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


@pytest.fixture
def fake_runtime(monkeypatch):
    frame = pd.DataFrame(
        {
            "record_id": ["excluded", "normal", "defect", "normal-2"],
            "image_path": ["never-open.png", "normal.png", "defect.png", "normal-2.png"],
            "label": [1, 0, 1, 0],
            "fold": [0, 1, 2, 2],
            "partition": ["development"] * 4,
        }
    )
    monkeypatch.setattr(
        pilot, "checked_protocol", lambda root: {"seeds": [77], "protocol_hash": "frozen"}
    )
    monkeypatch.setattr(pilot, "load_development", lambda root: frame)
    cuda = SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
        reset_peak_memory_stats=lambda gpu: None,
        synchronize=lambda gpu: None,
        max_memory_allocated=lambda gpu: 12345,
        get_device_name=lambda gpu: "NVIDIA GeForce RTX 5090",
    )
    monkeypatch.setattr(pilot, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(
        pilot.psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=98765)),
    )
    monkeypatch.setattr(pilot.models, "_TRAINING_SAMPLE_CACHE_BYTES", 4567, raising=False)
    times = iter([100.0, 130.0, 200.0, 260.0])
    monkeypatch.setattr(pilot, "perf_counter", lambda: next(times))
    calls, predictions = [], []

    def predict(paths):
        predictions.append(paths)
        return np.array([0.5]), [np.ones((2, 2))]

    def fit(config, train, seed, directory, device):
        calls.append((config, train.copy(), seed, directory, device))
        directory.mkdir(parents=True)
        history = [
            {"epoch": epoch, "epochs": 3, "examples": 8, "seconds": float(epoch + 1)}
            for epoch in range(1, 4)
        ]
        (directory / "training_history.json").write_text(json.dumps(history))
        return SimpleNamespace(predict=predict)

    monkeypatch.setattr(pilot.models, "fit_model", fit)
    return calls, predictions, frame


def test_real_fold_training_rows_and_complete_elapsed_estimate(tmp_path, fake_runtime):
    calls, predictions, _ = fake_runtime
    report = pilot.main(tmp_path)
    assert len(calls) == 2
    for config, train, seed, directory, device in calls:
        assert train.record_id.tolist() == ["normal", "defect", "normal-2"]
        assert not (train.fold == 0).any()
        assert seed == 77 and device == "cuda:0"
        assert config["epochs"] == 3 and config["lr_multiplier"] == 1.0
        assert directory.name == f"steady-{config['resolution'][0]}"
    assert predictions == [[Path("normal.png")], [Path("normal.png")]]
    first, second = report["neural_pilots"]
    assert first["estimated_100_epoch_fold_hours"] == pytest.approx(30 / 3 * 100 / 3600)
    assert second["estimated_100_epoch_fold_hours"] == pytest.approx(60 / 3 * 100 / 3600)
    assert first["epoch_training_seconds_excluding_checkpoint_save"] == [2.0, 3.0, 4.0]
    assert first["training_rows"] == 3 and first["positive_training_rows"] == 1
    assert first["actual_examples_per_epoch"] == [8, 8, 8]
    assert first["fit_peak_cuda_bytes"] == 12345 and first["rss_bytes_at_end"] == 98765
    assert first["preprocessing_cache_bytes_at_end"] == 4567
    assert first["finite_prediction_on_one_training_image"]
    assert json.loads((tmp_path / "vision/runtime/runpod/steady-pilot.json").read_text()) == report


def test_cuda_context_is_initialized_before_peak_memory_reset(
    tmp_path, fake_runtime, monkeypatch
):
    initialized = False
    events = []

    def synchronize(gpu):
        nonlocal initialized
        assert gpu == 0
        initialized = True
        events.append("synchronize")

    def reset_peak_memory_stats(gpu):
        assert gpu == 0
        if not initialized:
            raise RuntimeError("Invalid device argument")
        events.append("reset")

    monkeypatch.setattr(pilot.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(pilot.torch.cuda, "empty_cache", lambda: events.append("empty"))
    monkeypatch.setattr(pilot.torch.cuda, "reset_peak_memory_stats", reset_peak_memory_stats)
    pilot.main(tmp_path)
    assert events[:3] == ["synchronize", "empty", "reset"]
    assert events.count("reset") == 2


def test_existing_pilot_directory_is_never_overwritten(tmp_path, fake_runtime):
    calls, _, _ = fake_runtime
    (tmp_path / "vision/pilots/steady-1056").mkdir(parents=True)
    with pytest.raises(ValueError, match="evidence exists"):
        pilot.main(tmp_path)
    assert not calls


def test_nonfinite_training_prediction_is_rejected_without_success_marker(
    tmp_path, fake_runtime, monkeypatch
):
    original = pilot.models.fit_model

    def fit(*args, **kwargs):
        predictor = original(*args, **kwargs)
        predictor.predict = lambda paths: (np.array([float("nan")]), [np.ones((2, 2))])
        return predictor

    monkeypatch.setattr(pilot.models, "fit_model", fit)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        pilot.main(tmp_path)
    assert not (tmp_path / "vision/runtime/runpod/steady-pilot.json").exists()


def test_missing_training_class_is_rejected_before_fitting(tmp_path, fake_runtime):
    calls, _, frame = fake_runtime
    frame.loc[frame.fold != 0, "label"] = 0
    with pytest.raises(ValueError, match="both classes"):
        pilot.main(tmp_path)
    assert not calls
