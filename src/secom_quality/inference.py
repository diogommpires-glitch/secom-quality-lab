"""Strict input validation and the shared scoring/review implementation."""

import csv
import hashlib
import io
import math
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .common import fingerprint, measurement_fingerprint
from .contracts import FEATURE_NAMES, LIMITATION, SCHEMA_VERSION, LoadedBundle

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_BATCH_ROWS = 2000
RECORD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class ApplicationError(Exception):
    """An expected boundary failure with a stable public explanation."""

    def __init__(
        self, code: str, message: str, status: int = 422, issues: list[dict[str, Any]] | None = None
    ):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        self.issues = issues or []

    def detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "issues": self.issues}


def validate_record(record: dict[str, Any], row_index: int = 0) -> dict[str, Any]:
    """JSON numbers are intentionally not coerced from strings or booleans."""
    if not isinstance(record, dict):
        raise ApplicationError("invalid_record", "A record must be an object.")
    extra = set(record) - {"record_id", "observed_at", "sensors"}
    record_id = record.get("record_id")
    if extra:
        raise ApplicationError("unknown_fields", "Labels and unknown metadata are not accepted.")
    if not isinstance(record_id, str) or not RECORD_ID.fullmatch(record_id):
        raise ApplicationError("invalid_record_id", "record_id must be a safe, unique identifier.")
    sensors = record.get("sensors")
    if not isinstance(sensors, dict) or set(sensors) != set(FEATURE_NAMES):
        missing = (
            sorted(set(FEATURE_NAMES) - set(sensors or {})) if isinstance(sensors, dict) else []
        )
        raise ApplicationError(
            "sensor_schema",
            "Exactly the 590 canonical sensor columns are required.",
            issues=[
                {
                    "row_index": row_index,
                    "record_id": record_id,
                    "field": "sensors",
                    "reason_code": "sensor_schema",
                    "message": f"Missing columns: {', '.join(missing[:5])}",
                }
            ],
        )
    values: list[float | None] = []
    for name in FEATURE_NAMES:
        value = sensors[name]
        if value is None:
            values.append(None)
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ApplicationError(
                "invalid_measurement",
                "Sensor values must be finite numbers or null.",
                issues=[
                    {
                        "row_index": row_index,
                        "record_id": record_id,
                        "field": name,
                        "reason_code": "invalid_measurement",
                    }
                ],
            )
        else:
            try:
                converted = float(value)
            except OverflowError as exc:
                raise ApplicationError(
                    "nonfinite_measurement", "A measurement exceeds numeric limits."
                ) from exc
            if not math.isfinite(converted):
                raise ApplicationError(
                    "nonfinite_measurement", "Infinity and NaN are not JSON measurements."
                )
            values.append(converted)
    observed_at = record.get("observed_at")
    if observed_at is not None:
        try:
            parsed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timezone is required")
        except (ValueError, TypeError, AttributeError) as exc:
            raise ApplicationError(
                "invalid_timestamp", "observed_at must include a timezone."
            ) from exc
    return {"record_id": record_id, "observed_at": observed_at, "values": values}


def parse_csv(content: bytes) -> list[dict[str, Any]]:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ApplicationError("upload_too_large", "The CSV limit is 20 MiB.", 413)
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ApplicationError("encoding", "CSV must be UTF-8 text.") from exc
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader)
        expected = {"record_id", *FEATURE_NAMES}
        if (
            len(header) != len(set(header))
            or not expected.issubset(header)
            or set(header) - expected - {"observed_at"}
        ):
            raise ApplicationError(
                "csv_schema",
                "CSV needs record_id and all 590 canonical sensors; observed_at is optional.",
            )
        records = []
        seen: set[str] = set()
        for row_index, values in enumerate(reader, 1):
            if row_index > MAX_BATCH_ROWS:
                raise ApplicationError("too_many_rows", "A batch is limited to 2,000 rows.", 413)
            if len(values) != len(header):
                raise ApplicationError(
                    "csv_row_width", f"CSV row {row_index} has an incorrect number of cells."
                )
            raw = dict(zip(header, values, strict=True))
            sensors: dict[str, float | None] = {}
            for name in FEATURE_NAMES:
                token = raw[name].strip()
                try:
                    sensors[name] = None if token in {"", "NaN"} else float(token)
                except ValueError as exc:
                    raise ApplicationError(
                        "invalid_measurement", f"Row {row_index}: {name} is not numeric."
                    ) from exc
            record: dict[str, Any] = {"record_id": raw["record_id"], "sensors": sensors}
            if raw.get("observed_at"):
                record["observed_at"] = raw["observed_at"]
            validated = validate_record(record, row_index)
            if validated["record_id"] in seen:
                raise ApplicationError(
                    "duplicate_id", "record_id values must be unique within a batch."
                )
            seen.add(validated["record_id"])
            records.append(validated)
    except (csv.Error, StopIteration) as exc:
        raise ApplicationError(
            "invalid_csv", "A nonempty, correctly quoted CSV is required."
        ) from exc
    if not records:
        raise ApplicationError("empty_batch", "The CSV contains no data rows.")
    return records


def record_from_values(record_id: str, values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "record_id": record_id,
        "observed_at": None,
        "values": [None if np.isnan(value) else float(value) for value in array],
    }


def score_records(records: list[dict[str, Any]], bundle: LoadedBundle) -> list[dict[str, Any]]:
    """Accept only canonical validated records. Every caller uses this implementation."""
    protected_ids = set(bundle.quality.get("protected_ids", []))
    protected_hashes = set(bundle.quality.get("protected_fingerprints", []))
    retained = np.asarray(bundle.quality["retained_indices"], dtype=int)
    if not retained.size:
        raise ApplicationError("invalid_bundle", "The model has no retained sensors.", 503)
    X = np.asarray([record["values"] for record in records], dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ApplicationError("sensor_schema", "A record must contain 590 measurements.")
    hashes = [measurement_fingerprint(row) for row in X]
    for record, row_hash in zip(records, hashes, strict=True):
        if record["record_id"] in protected_ids or row_hash in protected_hashes:
            raise ApplicationError(
                "protected_test_record",
                "Known final-test records cannot be served or replayed.",
                403,
            )
    missingness = np.mean(np.isnan(X[:, retained]), axis=1)
    abstain_limit = float(bundle.quality.get("abstain_missingness", 0.5))
    eligible = (missingness <= abstain_limit) & (missingness < 1)
    scores = np.full(len(records), np.nan)
    numerical_failures: set[int] = set()
    if eligible.any():
        classes = np.asarray(bundle.predictor.classes_)
        failure_columns = np.flatnonzero(classes == 1)
        if len(failure_columns) != 1:
            raise ApplicationError(
                "invalid_classes", "Bundle failure-class mapping is invalid.", 503
            )

        def predict(values: np.ndarray) -> np.ndarray:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = np.asarray(
                    bundle.predictor.predict_proba(values)[:, int(failure_columns[0])]
                )
            if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
                raise FloatingPointError("nonfinite model output")
            return result

        eligible_indices = np.flatnonzero(eligible)
        try:
            scores[eligible] = predict(X[eligible])
        except (FloatingPointError, OverflowError, ValueError):
            # Isolate numerical outliers without dropping otherwise valid rows in the batch.
            for row_index in eligible_indices:
                try:
                    scores[row_index] = float(predict(X[row_index : row_index + 1])[0])
                except (FloatingPointError, OverflowError):
                    numerical_failures.add(int(row_index))
                    eligible[row_index] = False
                except ValueError as exc:
                    if not any(
                        token in str(exc).lower()
                        for token in ("finite", "infinity", "too large", "overflow", "nan")
                    ):
                        raise ApplicationError(
                            "invalid_model_output",
                            "The verified model could not score this input.",
                            503,
                        ) from exc
                    numerical_failures.add(int(row_index))
                    eligible[row_index] = False
    minimum = np.array([np.nan if x is None else x for x in bundle.quality["training_min"]])
    maximum = np.array([np.nan if x is None else x for x in bundle.quality["training_max"]])
    q99 = float(bundle.quality["training_missingness_q99"])
    results = []
    for i, record in enumerate(records):
        reasons = []
        if i in numerical_failures:
            reasons.append("numerical_failure")
        elif not eligible[i]:
            reasons.append("excessive_missingness")
        elif missingness[i] > q99:
            reasons.append("missingness_above_training_q99")
        range_count = int(np.sum((X[i] < minimum) | (X[i] > maximum)))
        if range_count:
            reasons.append("outside_training_range")
        results.append(
            {
                "prediction_id": str(uuid.uuid4()),
                "record_id": record["record_id"],
                "risk_score": None if not eligible[i] else float(scores[i]),
                "calibrated_probability": None,
                "score_kind": "uncalibrated",
                "calibration_status": "not_validated",
                "feature_schema_version": SCHEMA_VERSION,
                "quality_status": "abstain"
                if not eligible[i]
                else ("warning" if reasons else "valid"),
                "quality_reason_codes": reasons,
                "missingness": float(missingness[i]),
                "range_violation_count": range_count,
                "input_fingerprint": hashes[i],
                "bundle_id": bundle.manifest["bundle_id"],
                "model_id": bundle.manifest["model_id"],
            }
        )
    return results


def rank_predictions(predictions: list[dict[str, Any]], bundle: LoadedBundle) -> dict[str, Any]:
    eligible = [item for item in predictions if item["risk_score"] is not None]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -item["risk_score"],
            hashlib.sha256(item["record_id"].encode()).hexdigest(),
        ),
    )
    capacity = math.floor(float(bundle.policy.get("capacity_fraction", 0.1)) * len(ranked))
    for item in predictions:
        item.update(rank=None, review_recommended=False)
    for rank, item in enumerate(ranked, 1):
        item.update(rank=rank, review_recommended=rank <= capacity)
    return {
        "eligible_count": len(eligible),
        "review_capacity": capacity,
        "abstain_count": len(predictions) - len(eligible),
        "predictions": predictions,
        "queue_policy_id": fingerprint(bundle.policy)[:16],
    }


def csv_template() -> bytes:
    return (",".join(["record_id", *FEATURE_NAMES]) + "\r\n").encode()


def records_csv(records: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["record_id", *FEATURE_NAMES])
    for record in records:
        writer.writerow([record["record_id"], *["" if x is None else x for x in record["values"]]])
    return output.getvalue().encode()


def predictions_csv(result: dict[str, Any]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    fields = [
        "record_id",
        "risk_score",
        "quality_status",
        "rank",
        "review_recommended",
        "bundle_id",
    ]
    writer.writerow(fields)
    for item in result["predictions"]:
        cells = [item.get(field, "") for field in fields]
        writer.writerow(
            [
                "'" + cell
                if isinstance(cell, str) and cell.startswith(("=", "+", "-", "@"))
                else cell
                for cell in cells
            ]
        )
    return output.getvalue().encode()


def model_metadata(bundle: LoadedBundle) -> dict[str, Any]:
    return {
        "bundle_id": bundle.manifest["bundle_id"],
        "model_id": bundle.manifest["model_id"],
        "purpose": bundle.manifest["purpose"],
        "score_kind": "uncalibrated",
        "limitation": LIMITATION,
        "policy": bundle.policy,
        "quality_summary": {
            "abstain_missingness": 0.5,
            "retained_sensor_count": len(bundle.quality["retained_indices"]),
            "training_missingness_q99": bundle.quality["training_missingness_q99"],
        },
    }


def process_batch(
    root: Path, records: list[dict[str, Any]], client_request_id: str, bundle: LoadedBundle
) -> dict[str, Any]:
    from .storage import Store

    if not RECORD_ID.fullmatch(client_request_id):
        raise ApplicationError("invalid_request_id", "client_request_id must be a safe identifier.")
    payload_hash = fingerprint(
        {"records": records, "bundle_id": bundle.manifest["bundle_id"], "policy": bundle.policy}
    )
    result = {
        "batch_id": str(uuid.uuid4()),
        "row_count": len(records),
        "bundle_id": bundle.manifest["bundle_id"],
        "model_id": bundle.manifest["model_id"],
        "validation_issues": [],
        **rank_predictions(score_records(records, bundle), bundle),
    }
    return Store(root).save_batch(client_request_id, payload_hash, result)


def predict_csv(
    root: Path, input_path: Path, bundle_path: Path | None = None, capacity: float = 0.1
) -> dict[str, Any]:
    from .artifacts import load_bundle

    if capacity != 0.1:
        raise ApplicationError("frozen_policy", "The qualified review capacity is frozen at 10%.")
    bundle = load_bundle(bundle_path or root / "bundles" / "main")
    if input_path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ApplicationError("upload_too_large", "The CSV limit is 20 MiB.", 413)
    records = parse_csv(input_path.read_bytes())
    request_id = "cli-" + fingerprint({"records": records, "bundle": bundle.manifest["bundle_id"]})
    return process_batch(root, records, request_id, bundle)
