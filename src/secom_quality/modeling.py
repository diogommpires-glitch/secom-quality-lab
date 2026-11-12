"""The finite candidate set and fitted preprocessing used by research and serving."""

import warnings
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import SEED


def candidate_configs() -> list[dict[str, Any]]:
    configs = []
    for C in (0.01, 0.1, 1.0):
        for balanced in (False, True):
            for indicators in (False, True):
                configs.append(
                    {
                        "config_id": f"lr-c{C:g}-w{int(balanced)}-m{int(indicators)}",
                        "family": "logistic",
                        "C": C,
                        "balanced": balanced,
                        "indicators": indicators,
                        "promotable": True,
                    }
                )
    for leaves in (7, 15):
        for balanced in (False, True):
            configs.append(
                {
                    "config_id": f"hgb-l{leaves}-w{int(balanced)}",
                    "family": "histogram",
                    "leaves": leaves,
                    "balanced": balanced,
                    "promotable": True,
                }
            )
    configs.extend(
        [
            {
                "config_id": "diagnostic-missingness",
                "family": "logistic",
                "C": 0.1,
                "balanced": True,
                "indicators": True,
                "missingness_only": True,
                "promotable": False,
            },
            {
                "config_id": "diagnostic-top40",
                "family": "logistic",
                "C": 0.1,
                "balanced": True,
                "indicators": True,
                "top_k": 40,
                "promotable": False,
            },
        ]
    )
    return configs


def reference_config() -> dict:
    return next(c for c in candidate_configs() if c["config_id"] == "lr-c0.1-w1-m1")


def observed_sensor_columns(X: np.ndarray) -> list[int]:
    """Exclude training-empty columns while preserving native missingness elsewhere."""
    columns = np.flatnonzero(~np.isnan(X).all(axis=0)).tolist()
    if not columns:
        raise ValueError("Training data contains no observed sensor measurements")
    return columns


def make_predictor(config: dict) -> Any:
    if config["family"] == "histogram":
        boosting = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            max_leaf_nodes=config["leaves"],
            random_state=SEED,
            class_weight="balanced" if config["balanced"] else None,
        )
        return Pipeline(
            [
                (
                    "observed",
                    ColumnTransformer(
                        [
                            ("sensors", "passthrough", observed_sensor_columns),
                        ]
                    ),
                ),
                ("boosting", boosting),
            ]
        )
    columns = list(range(590))
    branches = []
    if not config.get("missingness_only", False):
        values = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("variance", VarianceThreshold()),
                ("scale", StandardScaler()),
            ]
        )
        branches.append(("values", values, columns))
    if config.get("indicators", False):
        missing = Pipeline(
            [("indicator", MissingIndicator(features="all")), ("variance", VarianceThreshold())]
        )
        branches.append(("missing", missing, columns))
    steps = [("features", ColumnTransformer(branches, sparse_threshold=0))]
    if config.get("top_k"):
        steps.append(("select", SelectKBest(f_classif, k=config["top_k"])))
    steps.append(
        (
            "model",
            LogisticRegression(
                C=config["C"],
                solver="lbfgs",
                max_iter=5000,
                tol=1e-4,
                random_state=SEED,
                class_weight="balanced" if config["balanced"] else None,
            ),
        )
    )
    return Pipeline(steps)


def fit_predictor(config: dict, X: np.ndarray, y: np.ndarray) -> Any:
    if len(np.unique(y)) != 2:
        raise ValueError("Training partition must contain both pass and fail labels")
    predictor = make_predictor(config)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        predictor.fit(X, y)
    return predictor


def score_rows(predictor: Any, X: np.ndarray) -> np.ndarray:
    classes = np.asarray(predictor.classes_)
    positions = np.flatnonzero(classes == 1)
    if len(positions) != 1 or set(classes.tolist()) != {0, 1}:
        raise ValueError("Predictor classes must be exactly pass=0 and fail=1")
    scores = np.asarray(predictor.predict_proba(X)[:, int(positions[0])], dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Predictor emitted invalid risk scores")
    return scores


def quality_reference(predictor: Any, X: np.ndarray) -> dict:
    retained = [j for j in range(X.shape[1]) if len(np.unique(X[~np.isnan(X[:, j]), j])) > 1]
    if not retained:
        raise ValueError("Training data contains no variable measured sensors")
    missingness = np.isnan(X[:, retained]).mean(axis=1)
    minima, maxima, deviations = [], [], []
    for column in X.T:
        observed = column[~np.isnan(column)]
        minima.append(float(observed.min()) if len(observed) else None)
        maxima.append(float(observed.max()) if len(observed) else None)
        deviations.append(float(observed.std()) if len(observed) else None)
    top_indices = []
    if isinstance(predictor, Pipeline) and "model" in predictor.named_steps:
        features = predictor.named_steps["features"].get_feature_names_out()
        if "select" in predictor.named_steps:
            features = features[predictor.named_steps["select"].get_support()]
        coefficients = predictor.named_steps["model"].coef_[0]
        ranked = sorted(zip(features, coefficients, strict=True), key=lambda pair: -abs(pair[1]))
        for name, _ in ranked:
            if name.startswith("values__x"):
                top_indices.append(int(name.removeprefix("values__x")))
            if len(top_indices) == 5:
                break
    return {
        "retained_indices": retained,
        "training_missingness_q99": float(np.quantile(missingness, 0.99)),
        "training_min": minima,
        "training_max": maxima,
        "training_std": deviations,
        "abstain_missingness": 0.5,
        "top_sensor_indices": top_indices,
        "quality_policy": "Abstain above 50% missing retained original sensors; warn above training 99th percentile. Demonstration policy, not a validated safety threshold.",
    }
