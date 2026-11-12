"""Verified, explicitly selected, tensor-only vision serving bundles."""

import importlib.metadata
import shutil
from functools import lru_cache
from pathlib import Path

from secom_quality.common import fingerprint, read_json, sha256_file, utc_now, write_json


def dependency_versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "numpy", "scikit-learn", "pillow", "safetensors")
    }


def build_bundle(root: Path):
    root = Path(root)
    research = root / "vision" / "research"
    selection = read_json(research / "selection.json")
    final = read_json(research / "final" / "report.json")
    if final["selection_hash"] != selection["selection_hash"]:
        raise ValueError("Final report and frozen selection differ")
    target = root / "vision" / "bundles" / "main"
    if target.exists():
        return load_verified_bundle(target)["manifest"]
    staging = target.with_name("main.staging")
    staging.mkdir(parents=True, exist_ok=True)
    predictor = research / "final" / "models" / str(selection["seeds"][0])
    (staging / "predictor").mkdir(exist_ok=True)
    for name in ("predictor.json", "weights.safetensors", "memory.safetensors"):
        if (predictor / name).exists():
            shutil.copy2(predictor / name, staging / "predictor" / name)
    # Published aggregates only. No paths, image data, labels or per-image errors.
    public = {
        key: value
        for key, value in final.items()
        if key not in {"predictions", "test_manifest", "error_analysis"}
    }
    write_json(staging / "report.json", public)
    write_json(
        staging / "model_card.json",
        {
            "intended_use": "Retrospective research demonstration of visual defect inspection ranking.",
            "limitations": [
                "Single production-image benchmark; unknown physical grouping.",
                "Uncalibrated score; localisation is model evidence, not a physical cause.",
                "Not validated factory deployment software.",
            ],
            "dataset_license": "KolektorSDD2 CC BY-NC-SA 4.0",
            "dataset_source": "https://www.vicos.si/resources/kolektorsdd2/",
        },
    )
    files = {
        p.relative_to(staging).as_posix(): sha256_file(p)
        for p in sorted(staging.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "schema_version": "industrial-vision-bundle-v1",
        "task": "visual_defect_detection",
        "model_id": selection["config"]["config_id"],
        "selection_hash": selection["selection_hash"],
        "created_at": utc_now(),
        "calibration_status": "not_validated",
        "score_kind": "uncalibrated",
        "policy": {"capacities": [0.1, 0.2], "rounding": "floor", "ties": "sha256(record_id)"},
        "dependencies": dependency_versions(),
        "files": files,
        "provenance": {"dataset": "KolektorSDD2", "protocol_hash": selection["protocol_hash"]},
    }
    manifest["bundle_id"] = "vision-" + fingerprint(manifest)[:16]
    write_json(staging / "manifest.json", manifest)
    load_verified_bundle(staging)
    staging.rename(target)
    return manifest


def load_verified_bundle(path: Path):
    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    if manifest.get("schema_version") != "industrial-vision-bundle-v1":
        raise ValueError("Unsupported vision bundle schema")
    files = manifest.get("files", {})
    if (
        not files
        or not {"predictor/predictor.json", "report.json", "model_card.json"} <= files.keys()
    ):
        raise ValueError("Vision bundle is incomplete")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("Vision bundle has missing or unexpected files")
    for name, digest in files.items():
        file = (path / name).resolve()
        if (
            not file.is_relative_to(path)
            or file.is_symlink()
            or file.suffix not in {".json", ".safetensors"}
        ):
            raise ValueError("Unsafe vision bundle member")
        if sha256_file(file) != digest:
            raise ValueError(f"Vision checksum mismatch: {name}")
    if manifest["dependencies"] != dependency_versions():
        raise ValueError("Vision bundle dependencies differ from its recorded environment")
    if (
        manifest["bundle_id"]
        != "vision-" + fingerprint({k: v for k, v in manifest.items() if k != "bundle_id"})[:16]
    ):
        raise ValueError("Vision bundle identity mismatch")
    return _load_predictor_cached(str(path), manifest["bundle_id"])


@lru_cache(maxsize=2)
def _load_predictor_cached(path_string, identity):
    # The caller verifies every byte before each access. Cache expensive network
    # reconstruction, never readiness verification or a file-stat trust shortcut.
    path = Path(path_string)
    manifest = read_json(path / "manifest.json")
    if manifest["bundle_id"] != identity:
        raise ValueError("Vision bundle changed while loading")
    from .models import load_predictor

    return {
        "manifest": manifest,
        "predictor": load_predictor(path / "predictor", device="cpu"),
        "report": read_json(path / "report.json"),
    }
