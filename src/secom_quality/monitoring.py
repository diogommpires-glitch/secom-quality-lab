"""A separate development-only replay with explicit virtual outcome availability."""

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score

from .common import utc_now
from .contracts import FEATURE_NAMES, LIMITATION, SEED, Dataset, LoadedBundle
from .inference import ApplicationError, record_from_values, score_records
from .storage import Store

SCENARIOS = {"baseline", "missingness", "sensor_shift"}
BATCH_SIZE = 20
REFERENCE_START, STREAM_START, STREAM_END = 438, 538, 1096


def public_stream(state: dict[str, Any]) -> dict[str, Any]:
    hidden = {"history", "recent_features", "reference_features", "reference_scores", "windows"}
    return {key: value for key, value in state.items() if key not in hidden}


class ReplayService:
    def __init__(
        self, root: Path, bundle: LoadedBundle | None = None, dataset: Dataset | None = None
    ):
        self.root = Path(root)
        self.store = Store(self.root)
        if bundle is None:
            from .artifacts import load_bundle

            bundle = load_bundle(self.root / "bundles" / "monitoring_demo")
        if bundle.manifest["purpose"] != "monitoring_demo":
            raise ApplicationError(
                "demo_bundle_required", "Replay requires the isolated monitoring demo bundle.", 503
            )
        self.bundle = bundle
        if dataset is None:
            from .artifacts import load_development_fixture

            dataset = load_development_fixture(bundle)
        if len(dataset.ids) != STREAM_END:
            raise ApplicationError(
                "development_partition",
                "Replay requires exactly the frozen 1,096 development rows.",
                503,
            )
        self.dataset = dataset

    def create(self, scenario: str = "baseline") -> dict[str, Any]:
        if scenario not in SCENARIOS:
            raise ApplicationError(
                "unknown_scenario", "Choose baseline, missingness or sensor_shift."
            )
        reference = self.dataset.X[REFERENCE_START:STREAM_START]
        reference_records = [
            record_from_values(self.dataset.ids[i], self.dataset.X[i])
            for i in range(REFERENCE_START, STREAM_START)
        ]
        reference_predictions = score_records(reference_records, self.bundle)
        state: dict[str, Any] = {
            "stream_id": str(uuid.uuid4()),
            "scenario": scenario,
            "status": "running",
            "cursor": 0,
            "total_records": STREAM_END - STREAM_START,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "label_delay_batches": 2,
            "virtual_batch": 0,
            "created_at": utc_now(),
            "bundle_id": self.bundle.manifest["bundle_id"],
            "model_id": self.bundle.manifest["model_id"],
            "limitation": LIMITATION
            + " Monitoring is a development-only sensitivity demonstration; altered covariates retain historical outcomes.",
            "reference_features": [
                record_from_values("reference", row)["values"] for row in reference
            ],
            "reference_scores": [
                item["risk_score"]
                for item in reference_predictions
                if item["risk_score"] is not None
            ],
            "recent_features": [],
            "history": [],
            "windows": [],
            "latest_window": None,
            "consecutive_warning_windows": 0,
            "last_advance": None,
        }
        self.store.create_stream(state)
        return public_stream(state)

    def snapshot(self, stream_id: str) -> dict[str, Any]:
        return public_stream(self.store.get_stream(stream_id))

    def advance(
        self, stream_id: str, expected_cursor: int, steps: int = BATCH_SIZE
    ) -> dict[str, Any]:
        if steps != BATCH_SIZE:
            raise ApplicationError(
                "fixed_batch_size", "Replay advances exactly 20 records per virtual batch."
            )
        state = self.store.get_stream(stream_id)
        if state["bundle_id"] != self.bundle.manifest["bundle_id"]:
            raise ApplicationError(
                "bundle_mismatch", "This stream belongs to a different frozen demo bundle.", 409
            )
        if state["cursor"] != expected_cursor:
            if state.get("last_advance") == {"expected_cursor": expected_cursor, "steps": steps}:
                return public_stream(state)
            raise ApplicationError(
                "cursor_conflict", "Refresh the stream before advancing its current cursor.", 409
            )
        if state["status"] == "completed":
            return public_stream(state)
        old_cursor = state["cursor"]
        state["status"] = "running"
        state["virtual_batch"] += 1
        end = (
            min(old_cursor + steps, state["total_records"])
            if old_cursor < state["total_records"]
            else old_cursor + steps
        )
        records = []
        for sequence in range(old_cursor, min(end, state["total_records"])):
            source_index = STREAM_START + sequence
            values = self.dataset.X[source_index].copy()
            generator = np.random.default_rng(SEED + sequence)
            if state["scenario"] == "missingness":
                values[generator.random(len(values)) < 0.25] = np.nan
            elif state["scenario"] == "sensor_shift":
                for feature in self.bundle.quality.get("top_sensor_indices", [])[:5]:
                    standard_deviation = self.bundle.quality["training_std"][feature]
                    if standard_deviation is not None and np.isfinite(values[feature]):
                        values[feature] += 2 * float(standard_deviation)
            records.append(record_from_values(self.dataset.ids[source_index], values))
            state["recent_features"].append(record_from_values("window", values)["values"])
        predictions = score_records(records, self.bundle) if records else []
        for offset, prediction in enumerate(predictions):
            sequence = old_cursor + offset
            # Historical outcomes remain unavailable to monitoring until the declared virtual time.
            state["history"].append(
                {
                    "sequence": sequence,
                    "prediction": prediction,
                    "label_available_batch": state["virtual_batch"] + 2,
                    "source_index": STREAM_START + sequence,
                }
            )
        state["cursor"] = end
        state["recent_features"] = state["recent_features"][-100:]
        if end >= state["total_records"]:
            state["status"] = (
                "completed" if end >= state["total_records"] + 2 * BATCH_SIZE else "flushing"
            )
        self._calculate_window(state)
        state["last_advance"] = {"expected_cursor": expected_cursor, "steps": steps}
        saved = self.store.update_stream(stream_id, expected_cursor, state)
        return public_stream(saved)

    def pause(self, stream_id: str) -> dict[str, Any]:
        state = self.store.get_stream(stream_id)
        if state["status"] != "completed":
            state["status"] = "paused"
            self.store.update_stream(stream_id, state["cursor"], state)
        return public_stream(state)

    def _calculate_window(self, state: dict[str, Any]) -> None:
        retained = np.asarray(self.bundle.quality["retained_indices"], dtype=int)
        current = np.asarray(state["recent_features"], dtype=float)
        reference = np.asarray(state["reference_features"], dtype=float)
        history = state["history"]
        scores = [
            item["prediction"]["risk_score"]
            for item in history[-100:]
            if item["prediction"]["risk_score"] is not None
        ]
        full = len(current) == 100
        ref_missingness = float(np.isnan(reference[:, retained]).mean())
        missingness = float(np.isnan(current[:, retained]).mean()) if len(current) else 0.0
        ks = (
            float(ks_2samp(state["reference_scores"], scores).statistic)
            if scores and state["reference_scores"]
            else None
        )
        signal = bool(
            full and (missingness - ref_missingness >= 0.10 or (ks is not None and ks > 0.2))
        )
        state["consecutive_warning_windows"] = (
            state["consecutive_warning_windows"] + 1 if signal else 0
        )
        mature = [
            item for item in history if item["label_available_batch"] <= state["virtual_batch"]
        ]
        scorable = [item for item in mature if item["prediction"]["risk_score"] is not None][-100:]
        y = np.asarray([self.dataset.y[item["source_index"]] for item in scorable], dtype=int)
        positives, negatives = int(np.sum(y == 1)), int(np.sum(y == 0))
        metrics = None
        if len(scorable) >= 100 and positives >= 5 and negatives >= 5:
            risk = np.asarray([item["prediction"]["risk_score"] for item in scorable])
            metrics = {
                "average_precision": float(average_precision_score(y, risk)),
                "failure_fraction": float(y.mean()),
            }
        feature_summaries = []
        for index in self.bundle.quality.get("top_sensor_indices", [])[:5]:
            a, b = reference[:, index], current[:, index]
            feature_summaries.append(
                {
                    "sensor": FEATURE_NAMES[index],
                    "reference_mean": float(np.nanmean(a)) if np.isfinite(a).any() else None,
                    "current_mean": float(np.nanmean(b)) if np.isfinite(b).any() else None,
                }
            )
        window = {
            "cursor": state["cursor"],
            "arrived_count": len(history),
            "virtual_batch": state["virtual_batch"],
            "matured_label_count": len(mature),
            "pending_label_count": len(history) - len(mature),
            "abstain_count": sum(item["prediction"]["risk_score"] is None for item in history),
            "matured_abstain_count": sum(
                item["prediction"]["risk_score"] is None for item in mature
            ),
            "matured_abstain_failures": sum(
                int(self.dataset.y[item["source_index"]] == 1)
                for item in mature
                if item["prediction"]["risk_score"] is None
            ),
            "feature_window": {
                "count": len(current),
                "complete": full,
                "missingness": missingness,
                "reference_missingness": ref_missingness,
                "missingness_delta": missingness - ref_missingness,
                "score_ks": ks,
                "warning_signal": signal,
                "consecutive_warning_windows": state["consecutive_warning_windows"],
                "warning": state["consecutive_warning_windows"] >= 2,
                "warning_definition": "Demonstration heuristic, not a significance test or concept-drift proof.",
                "feature_summaries": feature_summaries,
            },
            "performance_window": {
                "count": len(scorable),
                "positive_count": positives,
                "negative_count": negatives,
                "coverage": len(mature) / len(history) if history else 0.0,
                "metrics": metrics,
                "status": "available" if metrics else "insufficient_mature_support",
                "first_sequence": scorable[0]["sequence"] if scorable else None,
                "last_sequence": scorable[-1]["sequence"] if scorable else None,
                "definition": "Latest 100 scorable records with outcomes available; this window lags input monitoring.",
            },
        }
        state["latest_window"] = window
        state["windows"].append(window)

    def windows(self, stream_id: str) -> list[dict[str, Any]]:
        return self.store.get_stream(stream_id)["windows"]
