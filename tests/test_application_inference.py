"""Behavioral boundary checks using a deliberately simple trusted test predictor."""

import copy

import numpy as np
import pytest

from secom_quality.common import measurement_fingerprint
from secom_quality.contracts import FEATURE_NAMES, LoadedBundle
from secom_quality.inference import (
    ApplicationError,
    parse_csv,
    process_batch,
    rank_predictions,
    record_from_values,
    records_csv,
    score_records,
    validate_record,
)
from secom_quality.storage import Store


class TestPredictor:
    __test__ = False
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        values = np.nan_to_num(X[:, 0], nan=0)
        positive = 1 / (1 + np.exp(-values))
        return np.column_stack([1 - positive, positive])


def make_bundle(tmp_path):
    return LoadedBundle(
        path=tmp_path,
        manifest={"bundle_id": "test-main", "model_id": "test-lr", "purpose": "main"},
        predictor=TestPredictor(),
        quality={
            "retained_indices": list(range(590)),
            "training_missingness_q99": 0.1,
            "training_min": [-10.0] * 590,
            "training_max": [10.0] * 590,
            "training_std": [1.0] * 590,
            "abstain_missingness": 0.5,
            "protected_ids": ["final-1"],
            "protected_fingerprints": [],
            "top_sensor_indices": [0, 1, 2, 3, 4],
        },
        schema={"feature_names": list(FEATURE_NAMES), "n_features": 590},
        policy={"capacity_fraction": 0.1, "tie_break": "sha256-record-id"},
        report={"audit": {"rows": 1096}, "fold_results": [], "final_status": "test-fixture"},
    )


@pytest.fixture
def bundle(tmp_path):
    return make_bundle(tmp_path)


def json_record(record_id="sample-1"):
    return {"record_id": record_id, "sensors": dict.fromkeys(FEATURE_NAMES, 0.0)}


@pytest.mark.parametrize("bad", [True, "1", float("inf"), float("nan")])
def test_json_rejects_coercion_and_nonfinite_values(bad):
    record = json_record()
    record["sensors"]["sensor_001"] = bad
    with pytest.raises(ApplicationError):
        validate_record(record)


def test_missing_measurement_is_distinct_from_absent_column():
    record = json_record()
    record["sensors"]["sensor_001"] = None
    assert validate_record(record)["values"][1] is None
    del record["sensors"]["sensor_001"]
    with pytest.raises(ApplicationError, match="590"):
        validate_record(record)


def test_labels_are_not_inference_inputs():
    record = json_record()
    record["label"] = 1
    with pytest.raises(ApplicationError, match="Labels"):
        validate_record(record)


def test_huge_json_integer_is_a_boundary_error():
    record = json_record()
    record["sensors"]["sensor_000"] = 10**1000
    with pytest.raises(ApplicationError) as error:
        validate_record(record)
    assert error.value.status == 422


def test_csv_canonicalization_and_duplicate_rejection():
    record = record_from_values("sample-1", np.zeros(590))
    record["values"][3] = None
    parsed = parse_csv(records_csv([record]))
    assert parsed == [record]
    with pytest.raises(ApplicationError, match="unique"):
        parse_csv(records_csv([record, record]))


def test_protected_id_and_measurement_fingerprint_are_both_denied(bundle):
    record = record_from_values("final-1", np.zeros(590))
    with pytest.raises(ApplicationError) as error:
        score_records([record], bundle)
    assert error.value.status == 403
    bundle.quality["protected_fingerprints"] = [measurement_fingerprint(np.zeros(590))]
    record["record_id"] = "renamed"
    with pytest.raises(ApplicationError):
        score_records([record], bundle)


def test_abstention_and_exact_capacity_are_separate(bundle):
    records = [record_from_values(f"row-{i}", np.zeros(590)) for i in range(21)]
    records[-1]["values"][:296] = [None] * 296
    predictions = score_records(records, bundle)
    result = rank_predictions(predictions, bundle)
    assert result["eligible_count"] == 20
    assert result["review_capacity"] == 2
    assert result["abstain_count"] == 1
    assert sum(item["review_recommended"] for item in predictions) == 2
    assert predictions[-1]["risk_score"] is None
    assert all(item["calibrated_probability"] is None for item in predictions)
    reversed_result = rank_predictions(list(reversed(copy.deepcopy(predictions))), bundle)
    assert {p["record_id"] for p in reversed_result["predictions"] if p["review_recommended"]} == {
        p["record_id"] for p in predictions if p["review_recommended"]
    }


def test_single_or_tiny_batch_does_not_force_review(bundle):
    predictions = score_records([validate_record(json_record())], bundle)
    assert rank_predictions(predictions, bundle)["review_capacity"] == 0


def test_real_pipeline_numeric_overflow_abstains_only_affected_record(bundle):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.random.default_rng(19).normal(scale=0.01, size=(30, 590))
    predictor = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=200))
    predictor.fit(X, np.arange(30) % 2)
    actual = LoadedBundle(
        bundle.path,
        bundle.manifest,
        predictor,
        bundle.quality,
        bundle.schema,
        bundle.policy,
        bundle.report,
    )
    records = [
        record_from_values("normal", X[0]),
        record_from_values("extreme", np.full(590, 1e308)),
    ]
    result = score_records(records, actual)
    assert result[0]["risk_score"] is not None
    assert result[1]["risk_score"] is None
    assert result[1]["quality_reason_codes"] == ["numerical_failure", "outside_training_range"]


def test_batch_retry_is_idempotent_and_conflicts_are_visible(tmp_path, bundle):
    records = [validate_record(json_record())]
    first = process_batch(tmp_path, records, "request-1", bundle)
    retry = process_batch(tmp_path, records, "request-1", bundle)
    assert first["batch_id"] == retry["batch_id"]
    assert retry["replayed"] is True
    records[0]["values"][0] = 2
    with pytest.raises(ApplicationError) as error:
        process_batch(tmp_path, records, "request-1", bundle)
    assert error.value.status == 409
    assert len(Store(tmp_path).list_batches()) == 1


def test_sqlite_transaction_rolls_back(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(RuntimeError), store.connection() as connection:
        connection.execute("INSERT INTO streams VALUES (?, ?, ?)", ("x", "now", "{}"))
        raise RuntimeError("interrupted request")
    with pytest.raises(ApplicationError):
        store.get_stream("x")
