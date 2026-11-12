"""Resumable vision experiments with a separate, immutable final-access ledger."""

import gc
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from secom_quality.common import fingerprint, read_json, sha256_file, utc_now, write_json

from .data import load_development, load_protected
from .metrics import image_metrics, localization_metrics, paired_bootstrap


@contextmanager
def exclusive(path):
    """OS releases the byte lock after interruption; the lock file remains harmless."""
    import sys

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def checked_protocol(root):
    from .protocol import checked_protocol as check

    return check(Path(root))


def _research(root):
    return Path(root) / "vision" / "research"


def _file_hashes(directory):
    return {
        p.relative_to(directory).as_posix(): sha256_file(p)
        for p in sorted(Path(directory).rglob("*"))
        if p.is_file()
    }


def _verify_files(directory, expected, *, excluded=()):
    directory = Path(directory).resolve()
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(expected) | set(excluded):
        raise ValueError("Completed artifact has missing or unexpected files")
    for name, digest in expected.items():
        file = (directory / name).resolve()
        if (
            not file.is_relative_to(directory)
            or (directory / name).is_symlink()
            or sha256_file(file) != digest
        ):
            raise ValueError("Completed artifact checksum mismatch")


def _read_fit(destination, protocol, config, seed, fold):
    result = read_json(Path(destination) / "result.json")
    if (
        result.get("protocol_hash") != protocol["protocol_hash"]
        or result.get("config") != config
        or result.get("seed") != seed
        or result.get("fold") != fold
    ):
        raise ValueError(
            "Completed experiment does not match frozen protocol/configuration/seed/fold"
        )
    _verify_files(destination, result.get("files", {}), excluded=("result.json",))
    return result


def run_fit(root, config, seed, fold, device="cuda"):
    from .models import fit_model

    protocol = checked_protocol(root)
    if (
        config not in protocol["candidate_configs"]
        or seed not in protocol["seeds"]
        or fold not in range(3)
    ):
        raise ValueError("Experiment is outside frozen configuration/seed/fold matrix")
    destination = _research(root) / "experiments" / config["config_id"] / str(seed) / f"fold-{fold}"
    result_file = destination / "result.json"
    if result_file.exists():
        return _read_fit(destination, protocol, config, seed, fold)
    data = load_development(Path(root))
    train, validation = data[data.fold != fold].copy(), data[data.fold == fold].copy()
    if set(train.group_id) & set(validation.group_id):
        raise ValueError("Training/validation group leakage")
    staging = destination.with_name(destination.name + ".staging")
    request = {
        "protocol_hash": protocol["protocol_hash"],
        "config": config,
        "seed": seed,
        "fold": fold,
        "training_ids_hash": fingerprint(train.record_id.tolist()),
        "validation_ids_hash": fingerprint(validation.record_id.tolist()),
    }
    request_path = staging / "fit-request.json"
    if staging.exists():
        if not request_path.exists():
            staging.rename(staging.with_name(staging.name + ".interrupted-" + uuid.uuid4().hex))
        elif read_json(request_path) != request:
            raise ValueError("Interrupted checkpoint does not match frozen fit request")
        elif (staging / "result.json").exists():
            completed = _read_fit(staging, protocol, config, seed, fold)
            staging.rename(destination)
            return completed
    staging.mkdir(parents=True, exist_ok=True)
    write_json(request_path, request)
    result = {
        "config": config,
        "seed": seed,
        "fold": fold,
        "protocol_hash": protocol["protocol_hash"],
        "started_at": utc_now(),
        "training_ids_hash": fingerprint(train.record_id.tolist()),
        "validation_ids_hash": fingerprint(validation.record_id.tolist()),
    }
    started = perf_counter()
    try:
        model_dir = staging / "model"
        # Tensor-only checkpoints resume this exact protocol/configuration/seed.
        with threadpool_limits(limits=4):
            predictor = fit_model(config, train, seed, model_dir, device=device)
            result["training_seconds"] = perf_counter() - started
            inference = perf_counter()
            scores, _ = predictor.predict([Path(p) for p in validation.image_path])
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(validation),) or not np.isfinite(scores).all():
            raise ValueError("Invalid validation scores")
        result["inference_seconds"] = perf_counter() - inference
        predictions = validation[["record_id", "label", "group_id", "fold"]].copy()
        predictions["risk_score"] = scores
        predictions.to_parquet(staging / "predictions.parquet", index=False)
        result["metrics"] = image_metrics(validation.label, scores, validation.record_id)
        result["status"] = "completed"
        del predictor
    except Exception as exc:
        result.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    result["elapsed_seconds"] = perf_counter() - started
    result["finished_at"] = utc_now()
    result["files"] = {
        p.relative_to(staging).as_posix(): sha256_file(p) for p in staging.rglob("*") if p.is_file()
    }
    write_json(staging / "result.json", result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return result


def comparison(root, seeds=None):
    protocol = checked_protocol(root)
    seeds = seeds or [protocol["seeds"][0]]
    rows = []
    for order, config in enumerate(protocol["candidate_configs"]):
        results = []
        for seed in seeds:
            for fold in range(3):
                path = (
                    _research(root)
                    / "experiments"
                    / config["config_id"]
                    / str(seed)
                    / f"fold-{fold}"
                    / "result.json"
                )
                if path.exists():
                    results.append(_read_fit(path.parent, protocol, config, seed, fold))
        valid = len(results) == len(seeds) * 3 and all(r["status"] == "completed" for r in results)
        rows.append(
            {
                "config": config,
                "order": order,
                "valid": valid,
                "completed_fits": len(results),
                "results": results,
                "mean_recall": float(np.mean([r["metrics"]["primary"]["recall"] for r in results]))
                if valid
                else None,
                "mean_ap": float(np.mean([r["metrics"]["average_precision"] for r in results]))
                if valid
                else None,
                "mean_inference_seconds": float(np.mean([r["inference_seconds"] for r in results]))
                if valid
                else None,
            }
        )
    return rows


def _ordering(row):
    return (-row["mean_recall"], -row["mean_ap"], row["mean_inference_seconds"], row["order"])


def run_experiments(root, device="cuda", config_id=None):
    root = Path(root)
    with exclusive(_research(root) / "experiment.lock"):
        protocol = checked_protocol(root)
        configs = protocol["candidate_configs"]
        if config_id:
            configs = [c for c in configs if c["config_id"] == config_id]
            if not configs:
                raise ValueError("Unknown frozen configuration")
        for config in configs:
            for fold in range(3):
                result = run_fit(root, config, protocol["seeds"][0], fold, device)
                print(f"{config['config_id']} fold={fold} {result['status']}", flush=True)
        rows = comparison(root)
        write_json(_research(root) / "development-progress.json", rows)
        if config_id:
            return rows
        if any(r["completed_fits"] != 3 for r in rows):
            raise ValueError("Initial matrix is incomplete")
        leaders = sorted([r for r in rows if r["valid"]], key=_ordering)[:2]
        if not leaders:
            raise ValueError("No valid model configurations")
        for row in leaders:
            for seed in protocol["seeds"][1:]:
                for fold in range(3):
                    result = run_fit(root, row["config"], seed, fold, device)
                    print(
                        f"{row['config']['config_id']} seed={seed} fold={fold} {result['status']}",
                        flush=True,
                    )
        result = {
            "protocol_hash": protocol["protocol_hash"],
            "initial": rows,
            "finalists": [
                r
                for r in comparison(root, protocol["seeds"])
                if r["config"]["config_id"] in {x["config"]["config_id"] for x in leaders}
            ],
        }
        destination = _research(root) / "development.json"
        if destination.exists() and fingerprint(read_json(destination)) != fingerprint(result):
            raise ValueError("Completed development results are immutable")
        write_json(destination, result)
        return result


def freeze_selection(root):
    protocol = checked_protocol(root)
    research = _research(root)
    development = read_json(research / "development.json")
    if development.get("protocol_hash") != protocol["protocol_hash"]:
        raise ValueError("Development comparison belongs to a different protocol")
    initial = comparison(root, [protocol["seeds"][0]])
    if development.get("initial") != initial:
        raise ValueError("Initial development matrix differs from verified experiments")
    expected_finalists = {
        row["config"]["config_id"]
        for row in sorted([row for row in initial if row["valid"]], key=_ordering)[:2]
    }
    if {row["config"]["config_id"] for row in development["finalists"]} != expected_finalists:
        raise ValueError("Finalist set differs from frozen initial selection rule")
    fresh = {row["config"]["config_id"]: row for row in comparison(root, protocol["seeds"])}
    for row in development["finalists"]:
        if row != fresh.get(row["config"]["config_id"]):
            raise ValueError("Frozen development comparison differs from verified experiments")
    valid = [r for r in development["finalists"] if r["valid"]]
    if not valid:
        raise ValueError("No valid five-seed finalist")
    winner = sorted(valid, key=_ordering)[0]
    reference_rows = []
    for family in ("resnet18_logistic", "patchcore"):
        candidates = [
            r for r in development["initial"] if r["valid"] and r["config"]["family"] == family
        ]
        if candidates and family != winner["config"]["family"]:
            reference_rows.append(sorted(candidates, key=_ordering)[0]["config"])
    selection = {
        "protocol_hash": protocol["protocol_hash"],
        "config": winner["config"],
        "seeds": protocol["seeds"],
        "development_hash": sha256_file(research / "development.json"),
        "reference_configs": reference_rows,
        "selection_rule": "Mean recall@10%, mean AP, measured inference time, frozen order",
        "serving_seed": protocol["seeds"][0],
    }
    selection["selection_hash"] = fingerprint(selection)
    path = research / "selection.json"
    if path.exists():
        if read_json(path) != selection:
            raise ValueError("Frozen selection cannot change")
    else:
        write_json(path, selection)
    return selection


def _final_model_files(directory):
    """Bind both selected-seed models and fixed baseline references before access."""
    return {
        f"{tree}/{name}": digest
        for tree in ("models", "references")
        for name, digest in _file_hashes(Path(directory) / tree).items()
    }


def _final_fit_requests(selection):
    requests = [
        (selection["config"], seed, Path("models") / str(seed)) for seed in selection["seeds"]
    ]
    requests.extend(
        (config, selection["seeds"][0], Path("references") / config["config_id"])
        for config in selection.get("reference_configs", [])
    )
    return requests


def evaluate_final(root, device="cuda"):
    from .models import fit_model, load_predictor

    root = Path(root)
    protocol = checked_protocol(root)
    research = _research(root)
    selection = read_json(research / "selection.json")
    if selection != freeze_selection(root):
        raise ValueError("Selection no longer matches development")
    with exclusive(research / "final.lock"):
        target = research / "final"
        if (target / "report.json").exists():
            checksums = read_json(target / "checksums.json")
            _verify_files(target, checksums, excluded=("checksums.json",))
            report = read_json(target / "report.json")
            ledger = read_json(research / "final-access.json")
            if (
                report["selection_hash"] != selection["selection_hash"]
                or ledger.get("selection_hash") != selection["selection_hash"]
                or ledger.get("protocol_hash") != protocol["protocol_hash"]
            ):
                raise ValueError(
                    "Final test has already been exposed to another selection/protocol"
                )
            expected_models = ledger.get("model_files")
            actual_models = _final_model_files(target)
            if not expected_models or expected_models != actual_models:
                raise ValueError("Final frozen model bytes differ from first-access ledger")
            return report
        staging = target.with_name("final.staging")
        staging.mkdir(parents=True, exist_ok=True)
        train = load_development(root)
        ledger_file = research / "final-access.json"
        if ledger_file.exists():
            authorization = read_json(ledger_file)
            if (
                authorization.get("selection_hash") != selection["selection_hash"]
                or authorization.get("protocol_hash") != protocol["protocol_hash"]
            ):
                raise ValueError("Test-exposed selection/protocol cannot be replaced")
            actual_models = _final_model_files(staging)
            if (
                not authorization.get("model_files")
                or actual_models != authorization["model_files"]
            ):
                raise ValueError(
                    "Test-exposed final models are missing or changed; refitting is forbidden"
                )
        else:
            # Fit and bind every final model before the first protected-label read.
            for config, seed, relative_path in _final_fit_requests(selection):
                model_dir = staging / relative_path
                metadata = model_dir / "predictor.json"
                if not metadata.exists():
                    fit_model(config, train, seed, model_dir, device=device)
                required = ["weights.safetensors"]
                if config["family"] == "patchcore":
                    required.append("memory.safetensors")
                if any(not (model_dir / name).is_file() for name in required):
                    raise ValueError(
                        "Final predictor weights are incomplete before protected access"
                    )
                fitted = read_json(metadata)
                training = fitted.get("training", {})
                if (
                    fitted.get("config") != config
                    or training.get("seed") != seed
                    or training.get("training_record_ids") != train.record_id.astype(str).tolist()
                ):
                    raise ValueError("Final model metadata does not match frozen full-training fit")
            model_files = _final_model_files(staging)
            authorization = {
                "selection_hash": selection["selection_hash"],
                "protocol_hash": protocol["protocol_hash"],
                "access_recorded": True,
                "recorded_at": utc_now(),
                "model_files": model_files,
            }
            write_json(ledger_file, authorization)
        test = load_protected(
            root, selection_hash=selection["selection_hash"], authorization=authorization
        )
        predictions = test[["record_id", "label", "group_id"]].copy()
        per_seed, localization = [], None
        for seed in selection["seeds"]:
            predictor = load_predictor(staging / "models" / str(seed), device=device)
            started = perf_counter()
            scores, maps = predictor.predict([Path(p) for p in test.image_path])
            metrics = image_metrics(test.label, scores, test.record_id)
            metrics["inference_seconds"] = perf_counter() - started
            per_seed.append({"seed": seed, "metrics": metrics})
            predictions[f"score_{seed}"] = scores
            if seed == selection["seeds"][0]:
                masks = [np.asarray(Image.open(p).convert("L")) > 0 for p in test.mask_path]
                localization = localization_metrics(masks, maps)
                del masks
            del predictor, maps
            gc.collect()
        primary = per_seed[0]["metrics"]
        references = []
        for config in selection["reference_configs"]:
            predictor = load_predictor(staging / "references" / config["config_id"], device=device)
            scores, maps = predictor.predict([Path(p) for p in test.image_path])
            predictions[config["config_id"]] = scores
            references.append(
                {
                    "model_id": config["config_id"],
                    "metrics": image_metrics(test.label, scores, test.record_id),
                    "paired_uncertainty": paired_bootstrap(
                        predictions, [f"score_{selection['seeds'][0]}", config["config_id"]]
                    ),
                }
            )
            del predictor, maps
        caught = primary["primary"]["true_positives"]
        expected = primary["baselines"]["random_review"]["expected_true_positives"]
        status = (
            "no_observed_improvement"
            if caught <= expected
            else (
                "supported_on_benchmark"
                if primary["random_review_one_sided_pvalue"] <= 0.05
                else "inconclusive"
            )
        )
        report = {
            "selection_hash": selection["selection_hash"],
            "protocol_hash": protocol["protocol_hash"],
            "model_id": selection["config"]["config_id"],
            "scientific_status": status,
            "serving_seed": selection["seeds"][0],
            "per_seed": per_seed,
            "primary": primary,
            "localization": localization,
            "references": references,
            "uncertainty": paired_bootstrap(predictions, [f"score_{selection['seeds'][0]}"]),
            "claim": "Retrospective visual inspection benchmark; not factory deployment validation.",
            "calibration_status": "not_validated",
            "completed_at": utc_now(),
        }
        predictions.to_parquet(staging / "predictions.parquet", index=False)
        write_json(staging / "report.json", report)
        write_json(
            staging / "checksums.json",
            {
                p.relative_to(staging).as_posix(): sha256_file(p)
                for p in staging.rglob("*")
                if p.is_file()
            },
        )
        staging.rename(target)
        return report
