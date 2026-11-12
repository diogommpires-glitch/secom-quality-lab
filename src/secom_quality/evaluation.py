"""Capacity decisions and uncertainty on frozen out-of-sample predictions."""

import hashlib

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.metrics import average_precision_score, precision_recall_curve

from .contracts import SEED


def rank_indices(scores: np.ndarray, ids: list[str]) -> np.ndarray:
    if len(scores) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Scores require equal-length unique record identities")
    if not np.isfinite(scores).all():
        raise ValueError("Risk scores must be finite")
    ties = [hashlib.sha256(record.encode()).hexdigest() for record in ids]
    return np.asarray(
        sorted(range(len(scores)), key=lambda i: (-float(scores[i]), ties[i])), dtype=int
    )


def capacity_metrics(
    y: np.ndarray, scores: np.ndarray, ids: list[str], fraction: float = 0.1
) -> dict:
    if not 0 <= fraction <= 1 or len(y) == 0:
        raise ValueError("Capacity requires a nonempty batch and fraction in [0, 1]")
    n, positives = len(y), int(y.sum())
    count = int(np.floor(fraction * n))
    selected = rank_indices(scores, ids)[:count]
    tp = int(y[selected].sum())
    precision = tp / count if count else None
    prevalence = positives / n
    return {
        "n_samples": n,
        "n_failures": positives,
        "review_count": count,
        "true_positives": tp,
        "false_positives": count - tp,
        "false_negatives": positives - tp,
        "true_negatives": n - positives - count + tp,
        "review_rate": count / n,
        "recall": tp / positives if positives else None,
        "precision": precision,
        "lift": precision / prevalence if precision is not None and prevalence else None,
    }


def random_review_baseline(n: int, positives: int, k: int) -> dict:
    distribution = hypergeom(n, positives, k)
    return {
        "expected_true_positives": k * positives / n,
        "expected_recall": k / n if positives else None,
        "expected_precision": positives / n if k else None,
        "central_95pct_count_range": [int(distribution.ppf(0.025)), int(distribution.ppf(0.975))],
        "meaning": "Conditional reference distribution for uniform random review, not a model confidence interval.",
    }


def evaluate_scores(y: np.ndarray, scores: np.ndarray, ids: list[str]) -> dict:
    primary = capacity_metrics(y, scores, ids)
    positives = int(y.sum())
    pvalue = (
        float(
            hypergeom.sf(primary["true_positives"] - 1, len(y), positives, primary["review_count"])
        )
        if positives
        else None
    )
    return {
        "primary": primary,
        "secondary20": capacity_metrics(y, scores, ids, 0.2),
        "average_precision": float(average_precision_score(y, scores)) if positives else None,
        "baselines": {
            "all_pass_accuracy": float((y == 0).mean()),
            "all_pass_recall": 0.0 if positives else None,
            "constant_score_average_precision": float(y.mean()),
            "random_review": random_review_baseline(len(y), positives, primary["review_count"]),
        },
        "random_review_one_sided_pvalue": pvalue,
    }


def metric_curves(y: np.ndarray, scores: np.ndarray, ids: list[str]) -> dict:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    fractions = np.linspace(0.01, 0.5, 50)
    metrics = [capacity_metrics(y, scores, ids, float(q)) for q in fractions]
    return {
        "precision_recall": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "capacity": {
            "fractions": fractions.tolist(),
            "recall": [m["recall"] for m in metrics],
            "precision": [m["precision"] for m in metrics],
        },
    }


def paired_day_bootstrap(
    y: np.ndarray,
    scores: np.ndarray,
    reference: np.ndarray,
    timestamps: np.ndarray,
    ids: list[str],
    samples: int = 2000,
) -> dict:
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    days = pd.DatetimeIndex(timestamps).normalize().to_numpy()
    unique = np.unique(days)
    groups = [np.flatnonzero(days == day) for day in unique]
    rng = np.random.default_rng(SEED)
    values: dict[str, list[float]] = {
        key: []
        for key in ("recall", "average_precision", "delta_recall", "delta_average_precision")
    }
    undefined = 0
    for _ in range(samples):
        indices = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        labels = y[indices]
        if labels.sum() == 0:
            undefined += 1
            continue
        # Occurrences remain distinct when a day is resampled more than once.
        sampled_ids = [f"{ids[index]}:{occurrence}" for occurrence, index in enumerate(indices)]
        a = capacity_metrics(labels, scores[indices], sampled_ids)["recall"]
        b = capacity_metrics(labels, reference[indices], sampled_ids)["recall"]
        ap = float(average_precision_score(labels, scores[indices]))
        ref_ap = float(average_precision_score(labels, reference[indices]))
        for key, value in (
            ("recall", a),
            ("average_precision", ap),
            ("delta_recall", a - b),
            ("delta_average_precision", ap - ref_ap),
        ):
            values[key].append(value)
    return {
        "samples": samples,
        "defined_samples": samples - undefined,
        "undefined_samples": undefined,
        "day_clusters": len(groups),
        "seed": SEED,
        "intervals": {
            key: np.quantile(value, [0.025, 0.975]).tolist() if value else None
            for key, value in values.items()
        },
        "assumption": "Calendar days are exchangeable clusters; serial dependence and the short historical window limit interpretation. These intervals do not establish future-domain coverage.",
    }
