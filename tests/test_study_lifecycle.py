"""Exercise holdout lifecycle safeguards using synthetic data only."""

import numpy as np
import pytest

from secom_quality import artifacts, study
from secom_quality.common import read_json, write_json
from secom_quality.contracts import Dataset


@pytest.fixture
def synthetic_study(tmp_path, monkeypatch):
    n = 1567
    data = Dataset(
        X=np.zeros((n, 590)),
        y=np.asarray([int(i % 13 == 0) for i in range(n)], dtype=np.int64),
        timestamps=np.asarray(
            [np.datetime64("2008-07-19") + np.timedelta64(i // 30, "D") for i in range(n)]
        ),
        ids=[f"synthetic-{i}" for i in range(n)],
    )
    selection = {"protocol_hash": "synthetic-protocol", "selection_hash": "synthetic-selection"}
    monkeypatch.setattr(
        study, "_checked_protocol", lambda _: {"protocol_hash": "synthetic-protocol"}
    )
    monkeypatch.setattr(study, "_checked_selection", lambda *_: selection)
    monkeypatch.setattr(study, "_raw_dataset", lambda _: data)
    monkeypatch.setattr(artifacts, "load_verified_model", lambda _: object())
    monkeypatch.setattr(study, "score_rows", lambda _, X: np.linspace(0.01, 0.99, len(X)))
    return tmp_path, selection


def test_final_result_reuses_same_selection_and_detects_mutation(synthetic_study):
    root, selection = synthetic_study
    first = study.evaluate_final(root, bootstrap_samples=10)
    assert study.evaluate_final(root, bootstrap_samples=10) == first
    ledger = read_json(root / "research/final_access_ledger.json")
    assert ledger["status"] == "completed"
    assert len(ledger["events"]) == 2
    path = root / "research/final_evaluation.json"
    changed = read_json(path)
    changed["final_status"] = "fabricated"
    write_json(path, changed)
    with pytest.raises(ValueError, match="changed"):
        study.verify_final_result(root, selection)


def test_interrupted_final_resumes_only_original_protocol(synthetic_study):
    root, selection = synthetic_study
    ledger_path = root / "research/final_access_ledger.json"
    write_json(
        ledger_path,
        {
            **selection,
            "bootstrap_samples": 10,
            "status": "opened",
            "events": [{"event": "final_access_started", "at": "synthetic"}],
        },
    )
    study.evaluate_final(root, bootstrap_samples=10)
    ledger = read_json(ledger_path)
    assert [item["event"] for item in ledger["events"]] == [
        "final_access_started",
        "final_access_resumed",
        "final_evaluation_completed",
    ]
    ledger["selection_hash"] = "changed-selection"
    write_json(ledger_path, ledger)
    with pytest.raises(PermissionError, match="identical"):
        study.evaluate_final(root, bootstrap_samples=10)


def test_final_predictions_are_checked_against_committed_hash(synthetic_study):
    root, selection = synthetic_study
    study.evaluate_final(root, bootstrap_samples=10)
    path = root / "research/final_predictions.parquet"
    with path.open("ab") as file:
        file.write(b"unexpected mutation")
    with pytest.raises(ValueError, match="predictions changed"):
        study.verify_final_result(root, selection)


def test_final_lock_excludes_a_second_reader(tmp_path):
    with (
        study._final_lock(tmp_path),
        pytest.raises(PermissionError, match="Another final"),
        study._final_lock(tmp_path),
    ):
        pytest.fail("A second final reader acquired the lock")


def test_abandoned_bootstrap_cannot_change_on_resume(synthetic_study):
    root, selection = synthetic_study
    write_json(
        root / "research/final_access_ledger.json",
        {
            **selection,
            "bootstrap_samples": 10,
            "status": "opened",
            "events": [],
        },
    )
    with pytest.raises(ValueError, match="original bootstrap"):
        study.evaluate_final(root, bootstrap_samples=20)
