"""Frozen development experiments and a single, recorded final evaluation."""

import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .common import fingerprint, read_json, sha256_file, utc_now, write_json
from .contracts import DEVELOPMENT_END, FOLDS, LIMITATION, SEED
from .data import _raw_dataset, audit_data, immutable_json, load_dataset, slice_dataset
from .evaluation import evaluate_scores, metric_curves, paired_day_bootstrap, rank_indices
from .modeling import (
    candidate_configs,
    fit_predictor,
    quality_reference,
    reference_config,
    score_rows,
)

SCIENTIFIC_FILES = (
    "contracts.py",
    "common.py",
    "data.py",
    "modeling.py",
    "evaluation.py",
    "study.py",
    "artifacts.py",
)


def dependency_versions() -> dict:
    return {
        name: version(name)
        for name in ("numpy", "pandas", "scipy", "scikit-learn", "skops", "pyarrow")
    }


def _scientific_hashes() -> dict:
    return {name: sha256_file(Path(__file__).parent / name) for name in SCIENTIFIC_FILES}


def freeze_protocol(root: Path) -> dict:
    root = Path(root)
    audit = audit_data(root)
    protocol = {
        "protocol_version": "secom-study-v1.1",
        "revision_reason": "Before any final access, all HGB development fits exposed a sklearn 1.9.1 all-missing-column binning failure. A training-only empty-column selector preserves native missing values elsewhere. The same candidate grid is rerun; protocol v1 and its source/results remain in evidence/protocol_v1.",
        "seed": SEED,
        "development_end": DEVELOPMENT_END,
        "folds": [{"training": [0, start], "validation": [start, end]} for start, end in FOLDS],
        "primary_capacity_fraction": 0.1,
        "secondary_capacity_fraction": 0.2,
        "rounding": "floor",
        "tie_break": "SHA-256 of permanent record ID, ascending",
        "candidate_configs": candidate_configs(),
        "selection_rule": "Mean forward recall at 10% capacity; mean average precision; frozen configuration order (simpler first).",
        "calibration": "None; predict_proba output is an uncalibrated, model-version-specific ranking score.",
        "exploratory_analysis": "After immutable selection: reference/selected models on the two development folds; top10 reference-derived sensor probes, five paired permutations; missing10/missing25 and +1 training standard-deviation stresses. At most four diagnostic refits; these results cannot alter selection.",
        "final_access": "Only evaluate_final after immutable selection; never choose a replacement using final results.",
        "planning_audit_disclosure": "Raw structure, timestamp and class counts were inspected to establish split feasibility; no predictive final results were computed during planning.",
        "source_hashes": audit["source_hashes"],
        "scientific_code_hashes": _scientific_hashes(),
        "dependencies": dependency_versions(),
        "lockfile_sha256": _lockfile_hash(),
        "limitation": LIMITATION,
    }
    protocol["protocol_hash"] = fingerprint(protocol)
    immutable_json(root / "research" / "protocol.json", protocol)
    return protocol


def _checked_protocol(root: Path) -> dict:
    protocol = read_json(Path(root) / "research" / "protocol.json")
    claimed = protocol["protocol_hash"]
    if fingerprint({k: v for k, v in protocol.items() if k != "protocol_hash"}) != claimed:
        raise ValueError("Frozen protocol hash is invalid")
    if protocol["scientific_code_hashes"] != _scientific_hashes():
        raise ValueError(
            "Scientific source changed after protocol freeze; use a new versioned workspace"
        )
    if protocol["dependencies"] != dependency_versions():
        raise ValueError("Dependencies changed after protocol freeze")
    if protocol.get("lockfile_sha256") != _lockfile_hash():
        raise ValueError("Dependency lockfile changed after protocol freeze")
    audit = audit_data(root)
    if protocol["source_hashes"] != audit["source_hashes"]:
        raise ValueError("Source data changed after protocol freeze")
    return protocol


def _lockfile_hash() -> str | None:
    lockfile = Path(__file__).resolve().parents[2] / "uv.lock"
    return sha256_file(lockfile) if lockfile.is_file() else None


def _fit_fold(task: tuple) -> tuple[dict, pd.DataFrame]:
    config, fold, start, end, X, y, timestamps, ids = task
    started = perf_counter()
    base = {
        "config_id": config["config_id"],
        "family": config["family"],
        "promotable": config["promotable"],
        "fold": fold,
        "train_size": start,
        "validation_size": end - start,
        "n_failures": int(y[start:end].sum()),
    }
    try:
        with threadpool_limits(limits=1):
            predictor = fit_predictor(config, X[:start], y[:start])
            scores = score_rows(predictor, X[start:end])
        metrics = evaluate_scores(y[start:end], scores, ids[start:end])
    except (ConvergenceWarning, ValueError, FloatingPointError) as error:
        return {
            **base,
            "status": "invalid",
            "reason": str(error),
            "fit_seconds": perf_counter() - started,
        }, pd.DataFrame()
    primary = metrics["primary"]
    row = {
        **base,
        "status": "ok",
        "recall_at_capacity": primary["recall"],
        "precision_at_capacity": primary["precision"],
        "review_count": primary["review_count"],
        "average_precision": metrics["average_precision"],
        "fit_seconds": perf_counter() - started,
    }
    predictions = pd.DataFrame(
        {
            "record_id": ids[start:end],
            "timestamp": timestamps[start:end],
            "label": y[start:end],
            "risk_score": scores,
            "config_id": config["config_id"],
            "fold": fold,
        }
    )
    return row, predictions


def run_experiments(root: Path, max_workers: int = 4) -> dict:
    root = Path(root)
    if not 1 <= max_workers <= 4:
        raise ValueError("max_workers must be between 1 and 4")
    protocol = _checked_protocol(root)
    destination = root / "research" / "experiments.json"
    if destination.exists():
        existing = read_json(destination)
        if existing["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("Existing experiments belong to a different protocol")
        return existing
    if (root / "research" / "final_access_ledger.json").exists():
        raise PermissionError("Development cannot run after final access")
    data = load_dataset(root)
    tasks = [
        (config, fold, start, end, data.X, data.y, data.timestamps, data.ids)
        for config in protocol["candidate_configs"]
        for fold, (start, end) in enumerate(FOLDS, 1)
    ]
    if max_workers == 1:
        results = [_fit_fold(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_fit_fold, tasks))
    rows = [row for row, _ in results]
    predictions = [table for _, table in results if len(table)]
    prediction_path = root / "research" / "development_predictions.parquet"
    pd.concat(predictions, ignore_index=True).to_parquet(
        prediction_path, index=False
    ) if predictions else pd.DataFrame().to_parquet(prediction_path, index=False)
    result = {
        "protocol_hash": protocol["protocol_hash"],
        "created_at": utc_now(),
        "fold_results": rows,
        "planned_fits": 36,
        "completed_fits": len(rows),
        "invalid_fits": sum(row["status"] != "ok" for row in rows),
        "predictions_sha256": sha256_file(prediction_path),
        "selection_scope": "Development only; both diagnostic configurations are excluded from promotion.",
    }
    immutable_json(destination, result)
    return result


def compare_experiments(root: Path) -> dict:
    root = Path(root)
    protocol = _checked_protocol(root)
    experiments = read_json(root / "research" / "experiments.json")
    prediction_path = root / "research" / "development_predictions.parquet"
    if (
        experiments["protocol_hash"] != protocol["protocol_hash"]
        or sha256_file(prediction_path) != experiments["predictions_sha256"]
    ):
        raise ValueError("Experiment provenance or predictions changed")
    candidates = []
    for order, config in enumerate(protocol["candidate_configs"]):
        rows = [
            row for row in experiments["fold_results"] if row["config_id"] == config["config_id"]
        ]
        valid = len(rows) == 2 and all(
            row["status"] == "ok" and row.get("recall_at_capacity") is not None for row in rows
        )
        candidates.append(
            {
                "config_id": config["config_id"],
                "family": config["family"],
                "promotable": config["promotable"],
                "valid": valid,
                "simplicity_order": order,
                "mean_recall_at_capacity": float(np.mean([r["recall_at_capacity"] for r in rows]))
                if valid
                else None,
                "mean_average_precision": float(np.mean([r["average_precision"] for r in rows]))
                if valid
                else None,
            }
        )
    eligible = [c for c in candidates if c["promotable"] and c["valid"]]
    if not eligible:
        raise ValueError("No promotable configuration completed both forward folds")
    eligible.sort(
        key=lambda c: (
            -c["mean_recall_at_capacity"],
            -c["mean_average_precision"],
            c["simplicity_order"],
        )
    )
    comparison = {
        "protocol_hash": protocol["protocol_hash"],
        "candidates": candidates,
        "selected_config_id": eligible[0]["config_id"],
        "selection_rule": protocol["selection_rule"],
        "caveat": "These are tuned development results, not unbiased final performance estimates.",
    }
    immutable_json(root / "research" / "comparison.json", comparison)
    return comparison


def freeze_selection(root: Path) -> dict:
    from .artifacts import save_verified_model

    root = Path(root)
    protocol = _checked_protocol(root)
    comparison = compare_experiments(root)
    selection_path = root / "research" / "selection.json"
    if selection_path.exists():
        selection = read_json(selection_path)
        if (
            selection["protocol_hash"] != protocol["protocol_hash"]
            or selection["config_id"] != comparison["selected_config_id"]
        ):
            raise ValueError("Frozen selection differs from the frozen development comparison")
        analyze_development(root, selection)
        return selection
    if (root / "research" / "final_access_ledger.json").exists():
        raise PermissionError("Selection cannot be changed after final access")
    config = next(
        c for c in candidate_configs() if c["config_id"] == comparison["selected_config_id"]
    )
    data = load_dataset(root)
    with threadpool_limits(limits=1):
        selected = fit_predictor(config, data.X, data.y)
        reference = fit_predictor(reference_config(), data.X, data.y)
    models = root / "research" / "models"
    selected_hash = save_verified_model(selected, models / "selected.skops")
    reference_hash = save_verified_model(reference, models / "reference.skops")
    chosen = next(c for c in comparison["candidates"] if c["config_id"] == config["config_id"])
    selection = {
        "protocol_hash": protocol["protocol_hash"],
        "config_id": config["config_id"],
        "model_config": config,
        "training_rows": DEVELOPMENT_END,
        "mean_recall_at_capacity": chosen["mean_recall_at_capacity"],
        "mean_average_precision": chosen["mean_average_precision"],
        "selection_rule": protocol["selection_rule"],
        "selected_model_sha256": selected_hash,
        "reference_model_sha256": reference_hash,
        "created_at": utc_now(),
        "status": "frozen_before_final_access",
    }
    selection["selection_hash"] = fingerprint(selection)
    immutable_json(selection_path, selection)
    analyze_development(root, selection)
    return selection


def analyze_development(root: Path, selection: dict) -> dict:
    """Bounded, explicitly exploratory diagnostics never select or replace a model."""
    destination = Path(root) / "research" / "exploratory_analysis.json"
    if destination.exists():
        existing = read_json(destination)
        if existing["selection_hash"] != selection["selection_hash"]:
            raise ValueError("Exploratory analysis belongs to another selection")
        return existing
    data = load_dataset(root)
    stress_rows, permutation_rows, coefficient_features = [], [], {}
    fits = 0
    for fold, (start, end) in enumerate(FOLDS, 1):
        X_train, y_train = data.X[:start], data.y[:start]
        X_valid, y_valid, ids = data.X[start:end], data.y[start:end], data.ids[start:end]
        with threadpool_limits(limits=1):
            reference = fit_predictor(reference_config(), X_train, y_train)
            fits += 1
            selected = reference
            if selection["config_id"] != reference_config()["config_id"]:
                selected = fit_predictor(selection["model_config"], X_train, y_train)
                fits += 1
            quality = quality_reference(reference, X_train)
            feature_names = reference.named_steps["features"].get_feature_names_out()
            coefficients = reference.named_steps["model"].coef_[0]
            ranked = sorted(
                zip(feature_names, coefficients, strict=True), key=lambda item: -abs(item[1])
            )
            probes = [
                int(name.removeprefix("values__x"))
                for name, _ in ranked
                if name.startswith("values__x")
            ][:10]
            coefficient_features[str(fold)] = probes
            deviations = np.asarray([0.0 if v is None else v for v in quality["training_std"]])
            rng = np.random.default_rng(SEED + fold)
            missing10 = X_valid.copy()
            missing10[rng.random(X_valid.shape) < 0.1] = np.nan
            missing25 = X_valid.copy()
            missing25[rng.random(X_valid.shape) < 0.25] = np.nan
            shifted = X_valid.copy()
            shifted[:, probes] += deviations[probes]
            scenarios = {
                "unaltered": X_valid,
                "additional_missing10": missing10,
                "additional_missing25": missing25,
                "probe_shift_plus1_training_std": shifted,
            }
            retained = quality["retained_indices"]
            for role, predictor in (("reference", reference), ("selected", selected)):
                baseline_scores = score_rows(predictor, X_valid)
                baseline_ap = float(
                    evaluate_scores(y_valid, baseline_scores, ids)["average_precision"]
                )
                for sensor in probes:
                    drops = []
                    for repeat in range(5):
                        perturbed = X_valid.copy()
                        permutation_rng = np.random.default_rng(
                            SEED + fold * 10000 + sensor * 10 + repeat
                        )
                        perturbed[:, sensor] = perturbed[
                            permutation_rng.permutation(len(perturbed)), sensor
                        ]
                        ap = evaluate_scores(y_valid, score_rows(predictor, perturbed), ids)[
                            "average_precision"
                        ]
                        drops.append(baseline_ap - ap)
                    permutation_rows.append(
                        {
                            "fold": fold,
                            "role": role,
                            "sensor_index": sensor,
                            "average_precision_drop_mean": float(np.mean(drops)),
                            "average_precision_drop_std": float(np.std(drops)),
                            "repeats": 5,
                        }
                    )
                for name, X_scenario in scenarios.items():
                    fractions = np.isnan(X_scenario[:, retained]).mean(axis=1)
                    scorable = (fractions <= 0.5) & (fractions < 1)
                    indices = np.flatnonzero(scorable)
                    if len(indices):
                        metrics = evaluate_scores(
                            y_valid[indices],
                            score_rows(predictor, X_scenario[indices]),
                            [ids[i] for i in indices],
                        )
                        primary = metrics["primary"]
                        recall, ap = primary["recall"], metrics["average_precision"]
                    else:
                        recall, ap = None, None
                    stress_rows.append(
                        {
                            "fold": fold,
                            "role": role,
                            "condition": name,
                            "n_records": len(X_scenario),
                            "scorable_count": len(indices),
                            "abstain_count": int((~scorable).sum()),
                            "recall_at_capacity": recall,
                            "average_precision": ap,
                        }
                    )
    first, second = set(coefficient_features["1"]), set(coefficient_features["2"])
    result = {
        "selection_hash": selection["selection_hash"],
        "diagnostic_refits": fits,
        "scope": "Development only, after frozen selection; no model promotion is permitted from these diagnostics.",
        "feature_probe_rule": "Top10 absolute scaled value coefficients of the training-only reference logistic model, excluding indicators. Probes are not an exhaustive importance ranking for boosted models.",
        "feature_stability": {
            "reference_top10_by_fold": coefficient_features,
            "jaccard_overlap": len(first & second) / len(first | second),
        },
        "permutation_importance": permutation_rows,
        "stress_results": stress_rows,
        "limitation": "Synthetic corruption and permutation sensitivity are exploratory and neither establish sensor causes nor new-domain generalisation.",
    }
    immutable_json(destination, result)
    return result


def _checked_selection(root: Path, protocol: dict) -> dict:
    selection = read_json(Path(root) / "research" / "selection.json")
    if (
        fingerprint({k: v for k, v in selection.items() if k != "selection_hash"})
        != selection["selection_hash"]
    ):
        raise ValueError("Selection fingerprint mismatch")
    if selection["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("Selection belongs to another protocol")
    for name, key in (
        ("selected", "selected_model_sha256"),
        ("reference", "reference_model_sha256"),
    ):
        if sha256_file(Path(root) / "research" / "models" / f"{name}.skops") != selection[key]:
            raise ValueError(f"Frozen {name} model changed")
    return selection


@contextmanager
def _final_lock(root: Path):
    """An OS lock survives concurrent commands and is released after process failure."""
    path = Path(root) / "research" / "final_evaluation.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise PermissionError(
                "Another final evaluation currently owns the holdout lock"
            ) from error
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def verify_final_result(root: Path, selection: dict | None = None) -> dict:
    root = Path(root)
    selection = selection or read_json(root / "research" / "selection.json")
    result_path = root / "research" / "final_evaluation.json"
    result = read_json(result_path)
    ledger = read_json(root / "research" / "final_access_ledger.json")
    completed = [
        event for event in ledger["events"] if event["event"] == "final_evaluation_completed"
    ]
    if ledger["status"] != "completed" or not completed:
        raise ValueError("Final result has no completed access-ledger record")
    if sha256_file(result_path) != completed[-1]["result_sha256"]:
        raise ValueError("Final result changed after its access-ledger commitment")
    if any(record["selection_hash"] != selection["selection_hash"] for record in (result, ledger)):
        raise ValueError("Final result or ledger selection mismatch")
    if any(record["protocol_hash"] != selection["protocol_hash"] for record in (result, ledger)):
        raise ValueError("Final result or ledger protocol mismatch")
    if sha256_file(root / "research" / "final_predictions.parquet") != result["predictions_sha256"]:
        raise ValueError("Frozen final predictions changed")
    return result


def evaluate_final(root: Path, bootstrap_samples: int = 2000) -> dict:
    """Resume only the same frozen evaluation, preserving every access event."""
    with _final_lock(Path(root)):
        return _evaluate_final_locked(Path(root), bootstrap_samples)


def _evaluate_final_locked(root: Path, bootstrap_samples: int) -> dict:
    from .artifacts import load_verified_model

    root = Path(root)
    if not 1 <= bootstrap_samples <= 10000:
        raise ValueError("bootstrap_samples must be between 1 and 10000")
    protocol = _checked_protocol(root)
    selection = _checked_selection(root, protocol)
    result_path = root / "research" / "final_evaluation.json"
    ledger_path = root / "research" / "final_access_ledger.json"
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        if (
            ledger["protocol_hash"] != protocol["protocol_hash"]
            or ledger["selection_hash"] != selection["selection_hash"]
        ):
            raise PermissionError(
                "Final resume requires the identical frozen protocol and selection"
            )
        if ledger["bootstrap_samples"] != bootstrap_samples:
            raise ValueError("Resume must use the original bootstrap sample count")
        if ledger["status"] == "completed":
            return verify_final_result(root, selection)
        ledger["events"].append({"event": "final_access_resumed", "at": utc_now()})
        write_json(ledger_path, ledger)
        if result_path.exists():
            result = read_json(result_path)
            if (
                result["selection_hash"] != selection["selection_hash"]
                or result["protocol_hash"] != protocol["protocol_hash"]
            ):
                raise ValueError("Interrupted result does not belong to this frozen evaluation")
            if (
                sha256_file(root / "research" / "final_predictions.parquet")
                != result["predictions_sha256"]
            ):
                raise ValueError("Interrupted final predictions changed")
            ledger["events"].append(
                {
                    "event": "final_evaluation_completed",
                    "at": utc_now(),
                    "result_sha256": sha256_file(result_path),
                    "recovered_after_interruption": True,
                }
            )
            ledger["status"] = "completed"
            write_json(ledger_path, ledger)
            return verify_final_result(root, selection)
    else:
        if result_path.exists():
            raise ValueError("Final result exists without an access ledger")
        ledger = {
            "protocol_hash": protocol["protocol_hash"],
            "selection_hash": selection["selection_hash"],
            "bootstrap_samples": bootstrap_samples,
            "events": [{"event": "final_access_started", "at": utc_now()}],
            "status": "opened",
        }
        write_json(ledger_path, ledger)
    data = slice_dataset(_raw_dataset(root), DEVELOPMENT_END, 1567)
    selected = load_verified_model(root / "research" / "models" / "selected.skops")
    reference = load_verified_model(root / "research" / "models" / "reference.skops")
    with threadpool_limits(limits=1):
        scores = score_rows(selected, data.X)
        reference_scores = score_rows(reference, data.X)
    metrics = evaluate_scores(data.y, scores, data.ids)
    metrics["reference_model"] = evaluate_scores(data.y, reference_scores, data.ids)
    metrics["bootstrap"] = paired_day_bootstrap(
        data.y, scores, reference_scores, data.timestamps, data.ids, bootstrap_samples
    )
    primary = metrics["primary"]
    expected = metrics["baselines"]["random_review"]["expected_true_positives"]
    if primary["true_positives"] <= expected:
        status = "no_improvement"
    elif (
        metrics["random_review_one_sided_pvalue"] is not None
        and metrics["random_review_one_sided_pvalue"] <= 0.05
    ):
        status = "supported"
    else:
        status = "inconclusive"
    predictions = root / "research" / "final_predictions.parquet"
    pd.DataFrame(
        {
            "record_id": data.ids,
            "timestamp": data.timestamps,
            "label": data.y,
            "risk_score": scores,
            "reference_score": reference_scores,
        }
    ).to_parquet(predictions, index=False)
    result = {
        "protocol_hash": protocol["protocol_hash"],
        "selection_hash": selection["selection_hash"],
        "created_at": utc_now(),
        "final_metrics": metrics,
        "final_status": status,
        "curves": metric_curves(data.y, scores, data.ids),
        "predictions_sha256": sha256_file(predictions),
        "interpretation": "Supported refers only to the frozen comparison with uniform random review in this historical test batch. Engineering release and future deployment validity are separate.",
        "limitation": LIMITATION,
    }
    immutable_json(result_path, result)
    ledger["events"].append(
        {
            "event": "final_evaluation_completed",
            "at": utc_now(),
            "result_sha256": sha256_file(result_path),
        }
    )
    ledger["status"] = "completed"
    write_json(ledger_path, ledger)
    return result


def report_data(root: Path) -> dict:
    root = Path(root)
    audit = read_json(root / "data" / "audit.json")
    public_audit = {k: v for k, v in audit.items() if not k.startswith("protected_")}
    research = root / "research"
    experiments = (
        read_json(research / "experiments.json") if (research / "experiments.json").exists() else {}
    )
    selection = (
        read_json(research / "selection.json") if (research / "selection.json").exists() else {}
    )
    final = (
        verify_final_result(root, selection)
        if (research / "final_evaluation.json").exists()
        else {}
    )
    exploratory = (
        read_json(research / "exploratory_analysis.json")
        if (research / "exploratory_analysis.json").exists()
        else {}
    )
    return {
        "limitation": LIMITATION,
        "audit": public_audit,
        "fold_results": experiments.get("fold_results", []),
        "selection": selection,
        "final_metrics": final.get("final_metrics", {}),
        "final_status": final.get("final_status", "pending"),
        "curves": final.get("curves", {}),
        "exploratory_analysis": exploratory,
        "conditions": [
            "Raw measurements are anonymous and historical.",
            "No lot identifiers, sensor semantics or label delays are supplied.",
            "Risk scores are uncalibrated; inspection capacity is hypothetical.",
            "No final-driven model replacement is permitted.",
        ],
    }


def build_report(root: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    result = report_data(root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "study_report.json", result)
    lines = [
        "# SECOM Quality Lab: frozen study",
        "",
        LIMITATION,
        "",
        "## Source and protocol",
        "",
        "UCI SECOM, CC BY 4.0: https://archive.ics.uci.edu/dataset/179/secom",
        "",
        "Pinned source: 1,567 production records, 590 anonymous measurements, 104 failures. "
        "The metadata's 591-feature description disagrees with the raw 590-column file.",
        "",
        "Development: first 1,096 rows (78 failures). Final: last 471 rows (26 failures). "
        "Two forward validation windows each contain 329 rows and 11 failures. "
        "All learned transforms are fitted within training partitions.",
        "",
        "## Development experiments",
        "",
        "| Configuration | Fold | Recall at capacity | Average precision | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in result["fold_results"]:
        recall = (
            f"{row['recall_at_capacity']:.4f}"
            if row.get("recall_at_capacity") is not None
            else "undefined"
        )
        ap = (
            f"{row['average_precision']:.4f}"
            if row.get("average_precision") is not None
            else "undefined"
        )
        lines.append(f"| {row['config_id']} | {row['fold']} | {recall} | {ap} | {row['status']} |")
    lines += [
        "",
        "Development scores were used for selection and are not unbiased final estimates.",
        "",
    ]
    exploratory = result.get("exploratory_analysis", {})
    if exploratory:
        lines += [
            "## Development-only diagnostics",
            "",
            exploratory["scope"],
            "",
            f"Diagnostic refits: {exploratory['diagnostic_refits']}. "
            f"Reference top-10 sensor Jaccard overlap between folds: "
            f"{exploratory['feature_stability']['jaccard_overlap']:.3f}. "
            "These sensor associations do not establish physical causes.",
            "",
            "| Fold | Model role | Synthetic condition | Scorable | Recall at capacity | Average precision |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in exploratory["stress_results"]:
            lines.append(
                f"| {row['fold']} | {row['role']} | {row['condition']} | "
                f"{row['scorable_count']}/{row['n_records']} | "
                f"{row['recall_at_capacity']} | {row['average_precision']} |"
            )
        lines += ["", exploratory["limitation"], ""]
    lines += ["## Final evidence", "", f"Scientific outcome: **{result['final_status']}**.", ""]
    if result["final_metrics"]:
        metrics = result["final_metrics"]
        primary = metrics["primary"]
        lines += [
            f"Selected configuration: `{result['selection']['config_id']}`. "
            f"At {primary['review_count']} reviews, the model catches {primary['true_positives']} of "
            f"{primary['n_failures']} failures; recall={primary['recall']:.4f}, precision={primary['precision']:.4f}. "
            f"Average precision={metrics['average_precision']:.4f}.",
            "",
            "The random-review hypergeometric range is a comparator distribution, not a future-performance interval. "
            "Paired calendar-day bootstrap intervals assume exchangeable days and may not cover serial dependence.",
            "",
        ]
        random = metrics["baselines"]["random_review"]
        secondary = metrics["secondary20"]
        reference = metrics["reference_model"]["primary"]
        bootstrap = metrics["bootstrap"]
        lines += [
            "| Comparison | Inspections | Failures caught | Recall | Precision |",
            "|---|---:|---:|---:|---:|",
            f"| Selected model, primary | {primary['review_count']} | {primary['true_positives']} | {primary['recall']:.4f} | {primary['precision']:.4f} |",
            f"| Fixed reference logistic | {reference['review_count']} | {reference['true_positives']} | {reference['recall']:.4f} | {reference['precision']:.4f} |",
            f"| Selected model, 20% sensitivity | {secondary['review_count']} | {secondary['true_positives']} | {secondary['recall']:.4f} | {secondary['precision']:.4f} |",
            f"| Uniform random review, expected | {primary['review_count']} | {random['expected_true_positives']:.4f} | {random['expected_recall']:.4f} | {random['expected_precision']:.4f} |",
            "",
            f"Missed failures: {primary['false_negatives']}; false reviews: {primary['false_positives']}. "
            f"Lift over final prevalence: {primary['lift']:.3f}. "
            f"Exact one-sided random-review p-value: {metrics['random_review_one_sided_pvalue']:.6g}. "
            f"Always-pass final accuracy: {metrics['baselines']['all_pass_accuracy']:.4%}, with zero failure recall.",
            "",
            f"Bootstrap: {bootstrap['defined_samples']} defined of {bootstrap['samples']} resamples; "
            f"{bootstrap['undefined_samples']} undefined, {bootstrap['day_clusters']} observed day clusters.",
            "",
            "| Conditional bootstrap quantity | 2.5th percentile | 97.5th percentile |",
            "|---|---:|---:|",
        ]
        for name, interval in bootstrap["intervals"].items():
            if interval:
                lines.append(f"| {name} | {interval[0]:.4f} | {interval[1]:.4f} |")
        lines += ["", bootstrap["assumption"], ""]
        curve = result["curves"]["precision_recall"]
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.step(curve["recall"], curve["precision"], where="post")
        axis.axhline(
            primary["n_failures"] / primary["n_samples"],
            linestyle="--",
            color="gray",
            label="Test prevalence",
        )
        axis.set(
            xlabel="Failure recall",
            ylabel="Precision",
            title="Frozen final precision-recall curve",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.legend()
        fig.tight_layout()
        fig.savefig(reports / "precision_recall.png", dpi=160)
        plt.close(fig)
        lines += ["![Final precision-recall curve](precision_recall.png)", ""]
        curve = result["curves"]["capacity"]
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.plot(curve["fractions"], curve["recall"], label="Frozen selected model")
        axis.plot(
            curve["fractions"],
            curve["fractions"],
            "--",
            color="gray",
            label="Random review expectation",
        )
        axis.axvline(0.1, color="#277464", alpha=0.6, label="Primary capacity")
        axis.set(
            xlabel="Requested review fraction",
            ylabel="Failure recall",
            title="Prespecified capacity sensitivity",
            ylim=(0, 1),
        )
        axis.legend()
        fig.tight_layout()
        fig.savefig(reports / "capacity_sensitivity.png", dpi=160)
        plt.close(fig)
        lines += ["![Capacity sensitivity](capacity_sensitivity.png)", ""]
        predictions = pd.read_parquet(root / "research" / "final_predictions.parquet")
        order = rank_indices(predictions.risk_score.to_numpy(), predictions.record_id.tolist())
        predictions["selected_for_review"] = False
        predictions.loc[order[: primary["review_count"]], "selected_for_review"] = True
        errors = predictions[(predictions.label == 1) != predictions.selected_for_review]
        errors.to_csv(reports / "final_error_analysis.csv", index=False)
        lines += [
            "## Frozen error review",
            "",
            "These examples are examined after model selection and final evaluation. They may not be used to retune this study.",
            "",
            "| Record | Historical label | Risk score | Selected for review |",
            "|---|---:|---:|---|",
        ]
        examples = pd.concat(
            [
                errors[errors.label == 1].head(3),
                errors[errors.label == 0].head(3),
            ]
        )
        for row in examples.itertuples():
            lines.append(
                f"| {row.record_id} | {row.label} | {row.risk_score:.6f} | {row.selected_for_review} |"
            )
        lines += [
            "",
            "Full offline errors are in final_error_analysis.csv. No physical cause can be inferred from anonymous measurements.",
            "",
        ]
    lines += [
        "## Release boundaries",
        "",
        "A reproducible implementation can be released with negative or inconclusive scientific results. "
        "This study establishes neither early warning nor physical causes, profits, operational safety or real factory performance. "
        "Post-final repairs must disclose that the holdout has been exposed.",
        "",
    ]
    destination = reports / "study_report.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
