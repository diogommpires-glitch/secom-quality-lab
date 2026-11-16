"""Image-level inspection metrics and paired, image/group-level uncertainty."""

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import average_precision_score, roc_auc_score

from secom_quality.evaluation import evaluate_scores

PRO_METRIC_REVISION = "FPR-boundary-left-limit-v1"
# Float64 histogram sums of inverse region sizes can span up to20million pixels.
PRO_ROUNDOFF_TOLERANCE = 1e-8


def partial_pro_area(xs, ys, limit=0.3):
    """Normalise trapezoidal PRO area; vertical segments at the cutoff add zero area."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if (
        xs.ndim != 1
        or xs.shape != ys.shape
        or not len(xs)
        or not np.isfinite(xs).all()
        or not np.isfinite(ys).all()
        or not 0 < limit <= 1
        or xs[0] != 0
        or xs[-1] < limit
        or (np.diff(xs) < 0).any()
    ):
        raise ValueError(
            "PRO integration requires an aligned finite ordered curve covering its FPR limit"
        )
    selected = xs <= limit
    x, y = xs[selected], ys[selected]
    if x[-1] < limit:
        upper = int(np.searchsorted(xs, limit, side="left"))
        lower = upper - 1
        endpoint = ys[lower] + (ys[upper] - ys[lower]) * (
            (limit - xs[lower]) / (xs[upper] - xs[lower])
        )
        x, y = np.r_[x, limit], np.r_[y, endpoint]
    area = float(np.trapezoid(y, x) / limit)
    if not -PRO_ROUNDOFF_TOLERANCE <= area <= 1 + PRO_ROUNDOFF_TOLERANCE:
        raise ArithmeticError(f"Normalised PRO area outside [0,1]: {area}")
    return min(1.0, max(0.0, area))


def image_metrics(y, scores, ids):
    original_y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if (
        original_y.ndim != 1
        or scores.shape != original_y.shape
        or not np.isin(original_y, [0, 1]).all()
    ):
        raise ValueError("Image metrics require aligned binary labels and scores")
    y = original_y.astype(int)
    result = evaluate_scores(y, scores, list(ids))
    result["auroc"] = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None
    k = result["primary"]["review_count"]
    result["capacity_recall_ceiling"] = min(1.0, k / int(y.sum())) if y.sum() else None
    return result


def paired_bootstrap(frame, score_columns, repeats=2000, seed=20261026):
    """Resample entire verified groups; unique IDs include bootstrap occurrence."""
    if repeats < 1 or len(frame) == 0 or not score_columns:
        raise ValueError(
            "Bootstrap requires nonempty predictions, score columns and positive replicates"
        )
    if frame.group_id.isna().any():
        raise ValueError(
            "Bootstrap groups cannot be missing; use record_id for explicitly independent images"
        )
    rng = np.random.default_rng(seed)
    groups = frame["group_id"].astype(str).unique()
    blocks = {g: np.flatnonzero(frame.group_id.astype(str).to_numpy() == g) for g in groups}
    values: dict[str, list[list[float]]] = {name: [] for name in score_columns}
    differences = []
    undefined = 0
    for _ in range(repeats):
        chosen = rng.choice(groups, len(groups), replace=True)
        indices = np.concatenate([blocks[g] for g in chosen])
        sample = frame.iloc[indices]
        y = sample.label.to_numpy(dtype=int)
        if not y.sum():
            undefined += 1
            continue
        ids = [f"{row_id}:draw:{i}" for i, row_id in enumerate(sample.record_id)]
        current = []
        for name in score_columns:
            metrics = image_metrics(y, sample[name].to_numpy(), ids)
            pair = [metrics["primary"]["recall"], metrics["average_precision"]]
            values[name].append(pair)
            current.append(pair)
        if len(current) == 2:
            differences.append(np.asarray(current[0]) - current[1])

    def interval(rows):
        return np.quantile(rows, [0.025, 0.975], axis=0).T.tolist() if rows else None

    return {
        "replicates": repeats,
        "undefined_replicates": undefined,
        "resampling_unit": "verified group, or image when physical groups are unavailable",
        "groups": len(groups),
        "metric_order": ["recall_at_10pct", "average_precision"],
        "intervals": {name: interval(rows) for name, rows in values.items()},
        "paired_difference": interval(differences),
        "assumption": "Observed groups approximate independent sampling units; this does not establish future factory performance.",
    }


def localization_metrics(masks, maps):
    """Exact pixel AP; AU-PRO integrates overlap up to 30% normal-pixel FPR.

    Threshold integration is an explicitly documented 201-quantile approximation.
    Empty normal images contribute false-positive pixels; padding has been removed.
    """
    if not masks or len(masks) != len(maps):
        raise ValueError("Localization requires paired nonempty masks/maps")
    for mask, score in zip(masks, maps, strict=True):
        if np.asarray(mask).ndim != 2 or np.asarray(mask).shape != np.asarray(score).shape:
            raise ValueError("Each localization mask/map must have identical original-image shape")
    flat_y = np.concatenate([np.asarray(mask, bool).ravel() for mask in masks])
    flat_s = np.concatenate([np.asarray(score, float).ravel() for score in maps])
    if not np.isfinite(flat_s).all() or len(flat_y) != len(flat_s):
        raise ValueError("Localization masks and finite maps must align")
    ap = float(average_precision_score(flat_y, flat_s)) if flat_y.any() else None
    normal_count = int((~flat_y).sum())
    components = [label(np.asarray(mask, bool)) for mask in masks]
    regions = sum(count for _, count in components)
    if not regions or not normal_count:
        return {
            "pixel_average_precision": ap,
            "au_pro_30": None,
            "au_pro_30_legacy": None,
            "metric_revision": PRO_METRIC_REVISION,
        }
    # Subsample only to choose thresholds; evaluate every pixel at each threshold.
    thresholds = np.unique(
        np.quantile(flat_s[:: max(1, len(flat_s) // 1_000_000)], np.linspace(0, 1, 201))
    )
    curve = [(0.0, 0.0)]
    for threshold in thresholds[::-1]:
        fp, overlaps = 0, []
        for mask, score, (cc, count) in zip(masks, maps, components, strict=True):
            predicted = np.asarray(score) >= threshold
            fp += int((predicted & ~np.asarray(mask, bool)).sum())
            if count:
                sizes = np.bincount(cc.ravel(), minlength=count + 1)[1:]
                caught = np.bincount(cc[predicted].ravel(), minlength=count + 1)[1:]
                overlaps.extend((caught / sizes).tolist())
        curve.append((fp / normal_count, float(np.mean(overlaps))))
    curve.append((1.0, 1.0))
    xs, ys = np.asarray(sorted(curve)).T
    selected = xs < 0.3
    x = np.r_[xs[selected], 0.3]
    y = np.r_[ys[selected], np.interp(0.3, xs, ys)]
    return {
        "pixel_average_precision": ap,
        "au_pro_30": partial_pro_area(xs, ys),
        "au_pro_30_legacy": float(np.trapezoid(y, x) / 0.3),
        "metric_revision": PRO_METRIC_REVISION,
        "au_pro_method": "201 empirical-quantile thresholds; original-size pixels; checked FPR range [0,0.3]; legacy cutoff interpolation separately retained",
    }
