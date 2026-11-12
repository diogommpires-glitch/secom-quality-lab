"""Scientific invariants independent of the final holdout."""

from copy import deepcopy

import numpy as np
import pytest

from secom_quality.artifacts import load_verified_model, save_verified_model
from secom_quality.data import immutable_json, load_dataset
from secom_quality.evaluation import (
    capacity_metrics,
    paired_day_bootstrap,
    random_review_baseline,
    rank_indices,
)
from secom_quality.modeling import (
    candidate_configs,
    fit_predictor,
    quality_reference,
    reference_config,
    score_rows,
)


def training_fixture():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(80, 590))
    X[rng.random(X.shape) < 0.04] = np.nan
    y = np.asarray([0] * 60 + [1] * 20)
    return X, y


def test_capacity_is_exact_deterministic_and_tie_order_independent():
    ids = [f"r{i}" for i in range(21)]
    y = np.asarray([1, 0, 1] + [0] * 18)
    scores = np.ones(21)
    selected = [ids[i] for i in rank_indices(scores, ids)[:2]]
    reversed_ids = list(reversed(ids))
    assert selected == [reversed_ids[i] for i in rank_indices(scores, reversed_ids)[:2]]
    assert capacity_metrics(y, scores, ids)["review_count"] == 2
    assert capacity_metrics(y, scores, ids, 0)["precision"] is None


def test_random_review_baseline_matches_pinned_final_combinatorics():
    baseline = random_review_baseline(471, 26, 47)
    assert baseline["expected_true_positives"] == pytest.approx(2.59447983014862)
    assert baseline["central_95pct_count_range"] == [0, 6]


def test_missing_failures_are_undefined_not_fabricated():
    metrics = capacity_metrics(np.zeros(20, dtype=int), np.arange(20), [str(i) for i in range(20)])
    assert metrics["recall"] is None
    assert metrics["lift"] is None


def test_candidate_budget_and_diagnostics_cannot_be_promoted():
    configs = candidate_configs()
    assert len(configs) == 18
    assert sum(c["promotable"] for c in configs) == 16
    assert len({c["config_id"] for c in configs}) == 18
    assert all(not c["promotable"] for c in configs if c["config_id"].startswith("diagnostic"))


def test_preprocessing_is_fitted_only_to_training_and_preserves_raw_schema():
    X, y = training_fixture()
    predictor = fit_predictor(reference_config(), X, y)
    before = deepcopy(
        predictor.named_steps["features"]
        .named_transformers_["values"]
        .named_steps["impute"]
        .statistics_
    )
    future = np.full((3, 590), 1e9)
    scores = score_rows(predictor, future)
    assert len(scores) == 3
    np.testing.assert_array_equal(
        before,
        predictor.named_steps["features"]
        .named_transformers_["values"]
        .named_steps["impute"]
        .statistics_,
    )
    assert predictor.n_features_in_ == 590
    quality = quality_reference(predictor, X)
    assert len(quality["retained_indices"]) == 590
    assert len(quality["top_sensor_indices"]) == 5


@pytest.mark.parametrize("family", ["logistic", "histogram"])
def test_skops_roundtrip_preserves_scores(tmp_path, family):
    X, y = training_fixture()
    config = next(c for c in candidate_configs() if c["family"] == family)
    predictor = fit_predictor(config, X, y)
    path = tmp_path / "model.skops"
    save_verified_model(predictor, path)
    restored = load_verified_model(path)
    np.testing.assert_allclose(
        score_rows(restored, X[:8]), score_rows(predictor, X[:8]), rtol=0, atol=1e-14
    )


def test_boosting_ignores_training_empty_sensor_without_filling_other_missingness():
    X, y = training_fixture()
    X[:, 0] = np.nan
    config = next(c for c in candidate_configs() if c["family"] == "histogram")
    predictor = fit_predictor(config, X, y)
    retained = predictor.named_steps["observed"].transformers_[0][2]
    assert 0 not in retained
    future = X[:5].copy()
    future[:, 0] = 1e9
    np.testing.assert_array_equal(score_rows(predictor, future), score_rows(predictor, X[:5]))
    transformed = predictor.named_steps["observed"].transform(X)
    assert np.isnan(transformed).any()


def test_public_dataset_access_never_opens_final(tmp_path):
    with pytest.raises(PermissionError, match="Final rows"):
        load_dataset(tmp_path, "final")
    with pytest.raises(PermissionError):
        load_dataset(tmp_path, "all")


def test_scientific_record_cannot_be_overwritten(tmp_path):
    path = tmp_path / "record.json"
    immutable_json(path, {"fixed": 1})
    immutable_json(path, {"fixed": 1})
    with pytest.raises(ValueError, match="Immutable"):
        immutable_json(path, {"fixed": 2})


def test_paired_bootstrap_identical_predictors_have_zero_difference():
    y = np.asarray([0, 1, 0, 0, 1, 0] * 4)
    scores = np.linspace(0, 1, len(y))
    dates = np.repeat(np.arange("2008-01-01", "2008-01-05", dtype="datetime64[D]"), 6)
    result = paired_day_bootstrap(
        y, scores, scores, dates, [str(i) for i in range(len(y))], samples=25
    )
    assert result["defined_samples"] == 25
    assert result["intervals"]["delta_recall"] == [0.0, 0.0]
    assert result["intervals"]["delta_average_precision"] == [0.0, 0.0]
