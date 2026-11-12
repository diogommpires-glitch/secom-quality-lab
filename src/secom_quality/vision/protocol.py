"""Frozen, bounded vision study protocol and deterministic candidate ordering."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..common import fingerprint, read_json, sha256_file
from .data import SEED, audit_data, immutable_json, vision_root

RESOLUTIONS = ([704, 256], [1056, 384])  # tensor height,width; portrait production items
SEEDS = [SEED + offset for offset in range(5)]


def candidate_configs() -> list[dict]:
    candidates = []
    for strength in (0.01, 0.1, 1.0):
        token = str(strength).replace(".", "p")
        candidates.append(
            {
                "config_id": f"linear-c{token}",
                "family": "resnet18_logistic",
                "resolution": RESOLUTIONS[0],
                "C": strength,
                "class_weight": "balanced",
            }
        )
    for resolution in RESOLUTIONS:
        for fraction in (0.01, 0.05):
            token = str(fraction).replace(".", "p")
            candidates.append(
                {
                    "config_id": f"patchcore-h{resolution[1]}-c{token}",
                    "family": "patchcore",
                    "resolution": resolution,
                    "coreset_fraction": fraction,
                    "max_memory_patches": 20000,
                    "prepool_max_patches": 100000,
                    "projection_dimensions": 128,
                    "coreset_algorithm": "seeded randompriority reservoir then projected approximate greedy kcenter,10random anchors",
                }
            )
    for resolution in RESOLUTIONS:
        for multiplier in (0.5, 1.0, 2.0):
            token = str(multiplier).replace(".", "p")
            candidates.append(
                {
                    "config_id": f"ssn-h{resolution[1]}-lr{token}",
                    "family": "supersimplenet",
                    "resolution": resolution,
                    "lr_multiplier": multiplier,
                    "epochs": 100,
                }
            )
    return candidates


def code_hashes() -> dict:
    directory = Path(__file__).parent
    scientific = {
        "data.py",
        "protocol.py",
        "models.py",
        "transforms.py",
        "metrics.py",
        "study.py",
        "bundle.py",
        "vendor.py",
        "robustness.py",
    }
    result = {
        path.name: sha256_file(path)
        for path in sorted(directory.glob("*.py"))
        if path.name in scientific
    }
    for name in ("common.py", "evaluation.py", "contracts.py"):
        result[f"shared/{name}"] = sha256_file(directory.parent / name)
    vendor = directory / "vendor"
    if vendor.is_dir():
        result.update(
            {
                str(path.relative_to(directory)).replace("\\", "/"): sha256_file(path)
                for path in sorted(vendor.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            }
        )
    return result


def dependency_versions() -> dict:
    dependencies = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pyarrow",
        "Pillow",
        "torch",
        "torchvision",
        "tifffile",
        "safetensors",
    ):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "not_installed"
    return dependencies


def freeze_protocol(root: Path) -> dict:
    audit = audit_data(root)
    protocol = {
        "protocol_version": "ksdd2-vision-v1",
        "dataset": "KolektorSDD2",
        "source_fingerprint": audit["source_fingerprint"],
        "audit_fingerprint": audit["audit_fingerprint"],
        "train_manifest_sha256": audit["train_manifest_sha256"],
        "protected_manifest_sha256": audit["protected_manifest_sha256"],
        "seeds": SEEDS,
        "folds": audit["development"]["folds"],
        "fold_policy": "Three stratified folds; verified duplicate groups never cross folds; no verified physical/session metadata.",
        "candidate_configs": candidate_configs(),
        "resolution_order": "height,width; preserve portrait aspect ratio via letterboxing; no crop",
        "primary_capacity": 0.1,
        "secondary_capacity": 0.2,
        "rounding": "floor",
        "tie_break": "SHA256 UTF8 permanent record_id ascending",
        "selection_rule": "Mean development fold recall at10%; mean image AP; measured inference cost; frozen candidate order. Compare two finalists over five fixed seeds, counting initial seed.",
        "numerical_precision": "SuperSimpleNet CUDA BF16autocast forward with FP32 master weights and losses; CPU FP32; inference FP32. Seeded deterministic settings with warned nondeterministic CUDA backward operations; no cross-GPU bitwise claim.",
        "training": "Full supervised training100epochs; no outer-validation checkpoint selection; PatchCore learns normal-training images only; linear scaling and classifier use training rows only.",
        "test_policy": "Freeze selection,training,preprocessing,dependencies and serving seed before durable final-access record; five frozen models evaluated once; no test-based reselection.",
        "serving_seed": SEED,
        "calibration_status": "not_validated",
        "uncertainty": {
            "replicates": 2000,
            "paired": True,
            "unit": "verified physical groups if available, otherwise images",
            "recompute_queue": True,
            "limitation": "Image independence is an approximation; pixels are never sampling units.",
        },
        "confirmatory_test": "Selected model vs capacity-matched uniform random inspection; exact one-sided hypergeometric tail,alpha0.05.",
        "patchcore_amendment": "Before training: bounded100000patch uniform prepool then projected128d approximategreedy coreset; fraction uses prepool denominator, actualfullpoolcountandretainedcountreported. This differs from all-patch author PatchCore and is disclosed.",
        "stress_tests": ["brightness", "blur", "jpeg_compression", "translation"],
        "no_test_trained_initialization": True,
        "scientific_code_hashes": code_hashes(),
        "dependencies": dependency_versions(),
        "intended_claim": "Retrospective image-based prioritisation on the named benchmark; no semiconductor sensor improvement, root cause or factory deployment claim.",
    }
    lockfile = Path(__file__).resolve().parents[3] / "configs" / "vision-requirements.lock"
    protocol["vision_lockfile_sha256"] = sha256_file(lockfile) if lockfile.exists() else None
    protocol["protocol_hash"] = fingerprint(protocol)
    immutable_json(vision_root(root) / "research" / "protocol.json", protocol)
    return protocol


def checked_protocol(root: Path) -> dict:
    protocol = read_json(vision_root(root) / "research" / "protocol.json")
    expected = protocol["protocol_hash"]
    if (
        fingerprint({key: value for key, value in protocol.items() if key != "protocol_hash"})
        != expected
    ):
        raise ValueError("Vision protocol checksum failed")
    if protocol["scientific_code_hashes"] != code_hashes():
        raise ValueError(
            "Vision source changed after freeze; use a new versioned research workspace"
        )
    if protocol["dependencies"] != dependency_versions():
        raise ValueError("Vision dependency versions changed after protocol freeze")
    lockfile = Path(__file__).resolve().parents[3] / "configs" / "vision-requirements.lock"
    actual_lock = sha256_file(lockfile) if lockfile.exists() else None
    if protocol["vision_lockfile_sha256"] != actual_lock:
        raise ValueError("Vision dependency lock changed after protocol freeze")
    audit = audit_data(root)
    if protocol["audit_fingerprint"] != audit["audit_fingerprint"]:
        raise ValueError("Vision data changed after protocol freeze")
    return protocol
