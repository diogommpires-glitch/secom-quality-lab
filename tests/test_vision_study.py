"""Independent scientific-artifact and final-exposure regression checks."""

import numpy as np
import pandas as pd
import pytest

from secom_quality.common import read_json, sha256_file, write_json
from secom_quality.vision import models, study
from secom_quality.vision.metrics import image_metrics, localization_metrics, paired_bootstrap


def test_image_metrics_capacity_ceiling_and_binary_contract():
    result = image_metrics(
        np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), np.arange(10)[::-1], [str(i) for i in range(10)]
    )
    assert result["capacity_recall_ceiling"] == 0.5
    assert result["primary"]["true_positives"] == 1
    with pytest.raises(ValueError, match="binary"):
        image_metrics([0, 0.7], [0.1, 0.8], ["a", "b"])


def test_bootstrap_rejects_unknown_groups_instead_of_one_null_cluster():
    frame = pd.DataFrame(
        {"record_id": ["a", "b"], "label": [0, 1], "group_id": [None, None], "score": [0.1, 0.9]}
    )
    with pytest.raises(ValueError, match="groups cannot be missing"):
        paired_bootstrap(frame, ["score"], repeats=10)
    frame["group_id"] = frame.record_id
    result = paired_bootstrap(frame, ["score"], repeats=10)
    assert result["groups"] == 2
    assert result["replicates"] == 10


def test_localization_rejects_per_image_shape_mismatch_and_handles_perfect_map():
    with pytest.raises(ValueError, match="identical original-image shape"):
        localization_metrics([np.zeros((2, 4))], [np.zeros((4, 2))])
    mask = np.array([[0, 0], [0, 1]], dtype=bool)
    result = localization_metrics([mask], [mask.astype(float)])
    assert result["pixel_average_precision"] == 1.0
    assert result["au_pro_30"] == pytest.approx(1.0)


def fit_record(tmp_path):
    directory = tmp_path / "fold"
    directory.mkdir()
    prediction = directory / "predictions.parquet"
    prediction.write_bytes(b"immutable-prediction-fixture")
    config = {"config_id": "fixed", "family": "resnet18_logistic"}
    protocol = {"protocol_hash": "protocol"}
    result = {
        "config": config,
        "protocol_hash": "protocol",
        "seed": 7,
        "fold": 0,
        "status": "completed",
        "files": {"predictions.parquet": sha256_file(prediction)},
    }
    write_json(directory / "result.json", result)
    return directory, protocol, config


def test_completed_fit_validates_binding_and_exact_files(tmp_path):
    directory, protocol, config = fit_record(tmp_path)
    assert study._read_fit(directory, protocol, config, 7, 0)["status"] == "completed"
    with pytest.raises(ValueError, match="frozen protocol"):
        study._read_fit(directory, protocol, config, 8, 0)
    (directory / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected files"):
        study._read_fit(directory, protocol, config, 7, 0)
    (directory / "extra.json").unlink()
    (directory / "predictions.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        study._read_fit(directory, protocol, config, 7, 0)


def final_fixture(tmp_path, monkeypatch):
    research = tmp_path / "vision" / "research"
    research.mkdir(parents=True)
    protocol = {"protocol_hash": "protocol"}
    selection = {"selection_hash": "selection", "config": {"config_id": "fixed"}, "seeds": [7]}
    write_json(research / "selection.json", selection)
    monkeypatch.setattr(study, "checked_protocol", lambda root: protocol)
    monkeypatch.setattr(study, "freeze_selection", lambda root: selection)
    monkeypatch.setattr(
        study, "load_development", lambda root: pd.DataFrame({"record_id": ["train-a"]})
    )
    monkeypatch.setattr(
        models,
        "fit_model",
        lambda *args, **kwargs: pytest.fail("Test-exposed models must not be refitted"),
    )
    monkeypatch.setattr(
        study,
        "load_protected",
        lambda *args, **kwargs: pytest.fail("Integrity must be checked before protected reads"),
    )
    return research, selection


def test_final_resume_refuses_changed_or_missing_models_before_label_access(tmp_path, monkeypatch):
    research, _ = final_fixture(tmp_path, monkeypatch)
    staging = research / "final.staging"
    model = staging / "models" / "7" / "weights.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"original")
    ledger = {
        "selection_hash": "selection",
        "protocol_hash": "protocol",
        "access_recorded": True,
        "model_files": {"models/7/weights.safetensors": sha256_file(model)},
    }
    write_json(research / "final-access.json", ledger)
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="refitting is forbidden"):
        study.evaluate_final(tmp_path)
    model.unlink()
    with pytest.raises(ValueError, match="refitting is forbidden"):
        study.evaluate_final(tmp_path)


def test_cached_final_report_checksum_is_verified(tmp_path, monkeypatch):
    research, _ = final_fixture(tmp_path, monkeypatch)
    final = research / "final"
    final.mkdir()
    report_path = final / "report.json"
    write_json(report_path, {"selection_hash": "selection"})
    write_json(final / "checksums.json", {"report.json": sha256_file(report_path)})
    report_path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        study.evaluate_final(tmp_path)


def test_constant_localization_scores_are_finite_and_region_free_masks_are_explicit():
    mask = np.array([[0, 0], [0, 1]], dtype=bool)
    result = localization_metrics([mask], [np.zeros((2, 2))])
    assert result["pixel_average_precision"] == 0.25
    assert result["au_pro_30"] == pytest.approx(0.15)
    assert localization_metrics([np.zeros((2, 2))], [np.zeros((2, 2))])["au_pro_30"] is None


def test_references_fit_before_access_and_are_immutable_after_access(tmp_path, monkeypatch):
    research, selection = final_fixture(tmp_path, monkeypatch)
    selection["config"] = {"config_id": "selected", "family": "supersimplenet"}
    selection["seeds"] = [7, 8]
    selection["reference_configs"] = [
        {"config_id": "linear", "family": "resnet18_logistic"},
        {"config_id": "patch", "family": "patchcore"},
    ]
    write_json(research / "selection.json", selection)
    fitted = []

    def fit(config, train, seed, output, device="cuda"):
        fitted.append((config["config_id"], seed))
        output.mkdir(parents=True, exist_ok=True)
        (output / "weights.safetensors").write_bytes(b"weights")
        if config["family"] == "patchcore":
            (output / "memory.safetensors").write_bytes(b"memory")
        write_json(
            output / "predictor.json",
            {
                "config": config,
                "training": {"seed": seed, "training_record_ids": train.record_id.tolist()},
            },
        )

    def stop_before_pixels(root, *, selection_hash, authorization):
        assert fitted == [("selected", 7), ("selected", 8), ("linear", 7), ("patch", 7)]
        assert authorization == read_json(research / "final-access.json")
        assert "references/linear/predictor.json" in authorization["model_files"]
        assert "references/patch/memory.safetensors" in authorization["model_files"]
        raise RuntimeError("stopped before protected pixel read")

    monkeypatch.setattr(models, "fit_model", fit)
    monkeypatch.setattr(study, "load_protected", stop_before_pixels)
    with pytest.raises(RuntimeError, match="stopped before protected"):
        study.evaluate_final(tmp_path)
    assert len(fitted) == 4
    reference = research / "final.staging" / "references" / "linear" / "weights.safetensors"
    reference.write_bytes(b"changed")
    with pytest.raises(ValueError, match="refitting is forbidden"):
        study.evaluate_final(tmp_path)
    reference.unlink()
    with pytest.raises(ValueError, match="refitting is forbidden"):
        study.evaluate_final(tmp_path)
    assert len(fitted) == 4


def test_cached_report_binds_reference_bytes_to_first_access_ledger(tmp_path, monkeypatch):
    research, _ = final_fixture(tmp_path, monkeypatch)
    final = research / "final"
    reference = final / "references" / "linear" / "weights.safetensors"
    selected = final / "models" / "7" / "weights.safetensors"
    reference.parent.mkdir(parents=True)
    selected.parent.mkdir(parents=True)
    reference.write_bytes(b"original reference")
    selected.write_bytes(b"original selected")
    write_json(
        research / "final-access.json",
        {
            "selection_hash": "selection",
            "protocol_hash": "protocol",
            "model_files": study._final_model_files(final),
        },
    )
    write_json(final / "report.json", {"selection_hash": "selection"})
    write_json(final / "checksums.json", study._file_hashes(final))
    assert study.evaluate_final(tmp_path)["selection_hash"] == "selection"
    reference.write_bytes(b"changed reference")
    (final / "checksums.json").unlink()
    write_json(final / "checksums.json", study._file_hashes(final))
    with pytest.raises(ValueError, match="first-access ledger"):
        study.evaluate_final(tmp_path)
