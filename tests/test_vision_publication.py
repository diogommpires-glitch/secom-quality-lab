"""Publication must bind the serving predictor to the evaluated model ledger."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from secom_quality.common import fingerprint, read_json, write_json
from secom_quality.vision import bundle, models, report, study


def final_artifacts(root):
    research = root / "vision/research"
    config = {"config_id": "fixture", "family": "resnet18_logistic"}
    protocol = {"seeds": [7], "candidate_configs": [config]}
    protocol["protocol_hash"] = fingerprint(protocol)
    selection = {
        "config": config,
        "seeds": [7],
        "serving_seed": 7,
        "reference_configs": [],
        "protocol_hash": protocol["protocol_hash"],
    }
    selection["selection_hash"] = fingerprint(selection)
    write_json(research / "protocol.json", protocol)
    write_json(research / "selection.json", selection)
    model = research / "final/models/7"
    write_json(
        model / "predictor.json",
        {
            "config": config,
            "training": {"seed": 7, "training_rows": 1, "training_record_ids": ["training-only"]},
        },
    )
    (model / "weights.safetensors").write_bytes(b"synthetic weights, no neural loading")
    final = {
        "selection_hash": selection["selection_hash"],
        "protocol_hash": protocol["protocol_hash"],
        "model_id": "fixture",
        "serving_seed": 7,
        "per_seed": [],
        "scientific_status": "synthetic_fixture",
        "primary": {"random_review_one_sided_pvalue": 0.5},
    }
    write_json(research / "final/report.json", final)
    pd.DataFrame(
        {
            "record_id": [f"image-{i}" for i in range(10)],
            "label": [1] + [0] * 9,
            "score_7": np.linspace(1, 0, 10),
        }
    ).to_parquet(research / "final/predictions.parquet")
    write_json(
        research / "final-access.json",
        {
            "selection_hash": selection["selection_hash"],
            "protocol_hash": protocol["protocol_hash"],
            "access_recorded": True,
            "model_files": study._final_model_files(research / "final"),
        },
    )
    write_json(research / "final/checksums.json", study._file_hashes(research / "final"))
    return research, protocol


@pytest.mark.parametrize("change", ["weights", "predictor", "ledger_missing", "checksums_missing"])
def test_bundle_rejects_changed_evaluation_lineage_before_publication(
    tmp_path, monkeypatch, change
):
    research, _ = final_artifacts(tmp_path)
    monkeypatch.setattr(models, "load_predictor", lambda *_a, **_k: SimpleNamespace(predict=None))
    if change == "weights":
        (research / "final/models/7/weights.safetensors").write_bytes(b"changed model")
    elif change == "predictor":
        path = research / "final/models/7/predictor.json"
        value = read_json(path)
        value["training"]["seed"] = 8
        write_json(path, value)
    elif change == "ledger_missing":
        (research / "final-access.json").unlink()
    else:
        (research / "final/checksums.json").unlink()
    if change in {"weights", "predictor"}:
        # Recomputing local file checksums still cannot replace first-access pins.
        (research / "final/checksums.json").unlink()
        write_json(research / "final/checksums.json", study._file_hashes(research / "final"))
    with pytest.raises(ValueError):
        bundle.build_bundle(tmp_path)
    assert not (tmp_path / "vision/bundles/main").exists()
    assert not (tmp_path / "vision/bundles/main.staging").exists()


def test_report_rejects_changed_model_before_publishing_old_evidence(tmp_path, monkeypatch):
    research, protocol = final_artifacts(tmp_path)
    (research / "final/models/7/weights.safetensors").write_bytes(b"changed model")
    (research / "final/checksums.json").unlink()
    write_json(research / "final/checksums.json", study._file_hashes(research / "final"))
    monkeypatch.setattr(report, "checked_protocol", lambda _: protocol)
    monkeypatch.setattr(report, "comparison", lambda _: [])
    with pytest.raises(ValueError):
        report.build_report(tmp_path)
    assert not (tmp_path / "vision/reports/published-report.json").exists()


def test_bundle_binds_copied_bytes_to_verified_first_access_snapshot(tmp_path, monkeypatch):
    final_artifacts(tmp_path)
    monkeypatch.setattr(models, "load_predictor", lambda *_a, **_k: SimpleNamespace(predict=None))
    copy = bundle.shutil.copy2

    def changed_during_copy(source, destination):
        if source.name == "weights.safetensors":
            source.write_bytes(b"changed after verification but before copying")
        return copy(source, destination)

    monkeypatch.setattr(bundle.shutil, "copy2", changed_during_copy)
    with pytest.raises(ValueError, match="copied predictor differs from verified final snapshot"):
        bundle.build_bundle(tmp_path)
    assert not (tmp_path / "vision/bundles/main").exists()


def test_report_binds_figure_input_before_publishing_final_json(tmp_path, monkeypatch):
    _, protocol = final_artifacts(tmp_path)
    monkeypatch.setattr(report, "checked_protocol", lambda _: protocol)
    monkeypatch.setattr(report, "comparison", lambda _: [])
    figures = report._figures

    def changed_before_figure_read(path, *args):
        frame = pd.read_parquet(path)
        frame["score_7"] = 1 - frame["score_7"]
        frame.to_parquet(path, index=False)
        return figures(path, *args)

    monkeypatch.setattr(report, "_figures", changed_before_figure_read)
    with pytest.raises(ValueError, match="figure input differs from verified final snapshot"):
        report.build_report(tmp_path)
    assert not (tmp_path / "vision/reports/published-report.json").exists()
