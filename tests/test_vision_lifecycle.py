"""Fast lifecycle acceptance using synthetic images and an explicit stub model.

This verifies research orchestration, not model quality. No production dataset,
pretrained network, GPU training or actual protected benchmark is accessed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from secom_quality.common import read_json, write_json
from secom_quality.vision import bundle, models, report, robustness, study
from secom_quality.vision.metrics import paired_bootstrap
from secom_quality.vision.protocol import candidate_configs


def _synthetic_images(root: Path, prefix: str, count: int, folds: bool) -> pd.DataFrame:
    root.mkdir(parents=True)
    rows = []
    for index in range(count):
        label = int(index % 5 == 0)
        image_path = root / f"{prefix}-{index}.png"
        mask_path = root / f"{prefix}-{index}-mask.png"
        image = np.full((24, 12, 3), 240 if label else 20, dtype=np.uint8)
        mask = np.zeros((24, 12), dtype=np.uint8)
        if label:
            mask[8:16, 4:8] = 255
        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)
        record_id = f"{prefix}-{index}"
        rows.append(
            {
                "record_id": record_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "label": label,
                "group_id": record_id,
                "fold": index % 3 if folds else -1,
            }
        )
    return pd.DataFrame(rows)


class SyntheticPredictor:
    """A deliberately obvious test double; never a scientific model result."""

    def __init__(self, config):
        self.config = config

    def predict(self, paths):
        scores, maps = [], []
        for path in paths:
            with Image.open(path) as image:
                values = np.asarray(image.convert("RGB"))
            positive = float(values.mean() > 150)
            if self.config["family"] == "supersimplenet":
                scores.append(positive)
            elif self.config["family"] == "patchcore":
                scores.append(-positive)
            else:
                scores.append(1 - positive)
            spatial = np.zeros(values.shape[:2], dtype=np.float32)
            if positive:
                spatial[8:16, 4:8] = 1
            maps.append(spatial)
        return np.asarray(scores), maps


def test_complete_synthetic_research_lifecycle_including_interrupted_final_resume(
    tmp_path, monkeypatch
):
    safetensors = pytest.importorskip("safetensors.numpy")
    train = _synthetic_images(tmp_path / "synthetic-development", "development", 30, True)
    protected = _synthetic_images(tmp_path / "synthetic-protected", "protected", 20, False)
    configs = candidate_configs()
    assert len(configs) == 13
    protocol = {
        "protocol_hash": "synthetic-acceptance-only",
        "candidate_configs": configs,
        "seeds": [20261026 + offset for offset in range(5)],
    }
    fitted, loaded = [], []

    def fit(config, supplied_train, seed, output, device="cuda"):
        assert device == "cpu"
        assert set(supplied_train.record_id) <= set(train.record_id)
        assert not set(supplied_train.record_id) & set(protected.record_id)
        fitted.append((config["config_id"], seed, len(supplied_train)))
        output.mkdir(parents=True, exist_ok=True)
        safetensors.save_file(
            {"fixture": np.array([1.0], dtype=np.float32)}, output / "weights.safetensors"
        )
        if config["family"] == "patchcore":
            safetensors.save_file(
                {"memory": np.zeros((1, 1), dtype=np.float32)}, output / "memory.safetensors"
            )
        write_json(
            output / "predictor.json",
            {
                "fixture_only": True,
                "config": config,
                "training": {
                    "seed": seed,
                    "training_record_ids": supplied_train.record_id.tolist(),
                },
            },
        )
        return SyntheticPredictor(config)

    def load(path, device="cpu"):
        assert device == "cpu"
        loaded.append(Path(path))
        return SyntheticPredictor(read_json(Path(path) / "predictor.json")["config"])

    monkeypatch.setattr(models, "fit_model", fit)
    monkeypatch.setattr(models, "load_predictor", load)
    monkeypatch.setattr(study, "checked_protocol", lambda _root: protocol)
    monkeypatch.setattr(study, "load_development", lambda _root: train.copy())
    monkeypatch.setattr(robustness, "checked_protocol", lambda _root: protocol)
    monkeypatch.setattr(robustness, "load_development", lambda _root: train.copy())
    monkeypatch.setattr(report, "checked_protocol", lambda _root: protocol)
    # Exercise paired resampling itself with a small deterministic acceptance budget.
    monkeypatch.setattr(
        study,
        "paired_bootstrap",
        lambda frame, columns: paired_bootstrap(frame, columns, repeats=24),
    )

    development = study.run_experiments(tmp_path, device="cpu")
    assert len(fitted) == 13 * 3 + 2 * 4 * 3 == 63
    assert all(row["valid"] and row["completed_fits"] == 3 for row in development["initial"])
    assert len(development["finalists"]) == 2
    assert all(row["valid"] and row["completed_fits"] == 15 for row in development["finalists"])
    assert all(row["config"]["family"] == "supersimplenet" for row in development["finalists"])
    assert study.run_experiments(tmp_path, device="cpu") == development
    assert len(fitted) == 63  # immutable completed fits are verified and reused
    stress = robustness.run_stress(tmp_path, device="cpu")
    assert stress["development_only"] is True
    assert len(stress["results"]) == 3 * len(robustness.SCENARIOS)
    assert not (tmp_path / "vision/research/stress-inputs").exists()
    selection = study.freeze_selection(tmp_path)
    assert selection["config"]["family"] == "supersimplenet"
    assert {c["family"] for c in selection["reference_configs"]} == {
        "resnet18_logistic",
        "patchcore",
    }
    assert study.freeze_selection(tmp_path) == selection

    research = tmp_path / "vision/research"
    protected_reads = []

    def protected_load(_root, *, selection_hash, authorization):
        assert selection_hash == selection["selection_hash"]
        assert read_json(research / "selection.json") == selection
        assert authorization == read_json(research / "final-access.json")
        assert len(fitted) == 70  # all five selected seeds and both references fit first
        assert "references/linear-c0p01/predictor.json" in authorization["model_files"] or any(
            name.startswith("references/linear-") for name in authorization["model_files"]
        )
        assert any(
            name.startswith("references/patchcore-") and name.endswith("memory.safetensors")
            for name in authorization["model_files"]
        )
        protected_reads.append(authorization)
        if len(protected_reads) == 1:
            raise RuntimeError("synthetic interruption immediately after access ledger")
        return protected.copy()

    monkeypatch.setattr(study, "load_protected", protected_load)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        study.evaluate_final(tmp_path, device="cpu")
    assert not (research / "final/report.json").exists()
    final = study.evaluate_final(tmp_path, device="cpu")
    assert len(fitted) == 70  # ledger-bound resume loads; it cannot refit
    assert len(final["per_seed"]) == 5
    assert len(final["references"]) == 2
    assert final["localization"]["pixel_average_precision"] == pytest.approx(1)
    assert final["primary"]["primary"]["review_count"] == 2
    assert final["uncertainty"]["replicates"] == 24
    assert final["calibration_status"] == "not_validated"
    assert study.evaluate_final(tmp_path, device="cpu") == final
    assert len(protected_reads) == 2  # completed final uses checked artifacts, no third read

    published = report.build_report(tmp_path)
    assert published["final_evaluated"] is True
    assert (tmp_path / "vision/reports/final_curves.png").is_file()
    assert read_json(tmp_path / "vision/reports/published-report.json") == final
    bundle._load_predictor_cached.cache_clear()
    manifest = bundle.build_bundle(tmp_path)
    bundle_path = tmp_path / "vision/bundles/main"
    verified = bundle.load_verified_bundle(bundle_path)
    assert verified["manifest"] == manifest
    assert verified["report"] == final
    assert not any(p.suffix in {".png", ".parquet", ".pt", ".pkl"} for p in bundle_path.rglob("*"))
    assert all(not name.startswith("references/") for name in manifest["files"])
    loads_before_corruption = len(loaded)
    (bundle_path / "predictor/weights.safetensors").write_bytes(b"synthetic corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        bundle.load_verified_bundle(bundle_path)
    assert len(loaded) == loads_before_corruption  # cached network never bypasses verification
    bundle._load_predictor_cached.cache_clear()
