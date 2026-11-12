"""Checksummed, dependency-bound skops bundles for the same research predictors."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .common import fingerprint, read_json, sha256_file, write_json
from .contracts import (
    DEVELOPMENT_END,
    FEATURE_NAMES,
    LIMITATION,
    SCHEMA_VERSION,
    Dataset,
    LoadedBundle,
)
from .data import immutable_json, load_dataset
from .evaluation import evaluate_scores
from .modeling import fit_predictor, quality_reference, reference_config, score_rows

SAFE_SKOPS_TYPES = {
    "secom_quality.modeling.observed_sensor_columns",
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn._loss.link.Interval",
    "sklearn._loss.link.LogitLink",
    "sklearn._loss.loss.HalfBinomialLoss",
    "sklearn._loss._loss.CyHalfBinomialLoss",
}


def save_verified_model(predictor, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Frozen predictor already exists: {path}")
    sio.dump(predictor, path)
    load_verified_model(path)
    return sha256_file(path)


def load_verified_model(path: Path):
    path = Path(path)
    if path.stat().st_size > 100_000_000:
        raise ValueError("Model exceeds the 100 MB local artifact limit")
    unknown = set(sio.get_untrusted_types(file=path))
    if not unknown <= SAFE_SKOPS_TYPES:
        raise ValueError(f"Unapproved serialized types: {sorted(unknown - SAFE_SKOPS_TYPES)}")
    predictor = sio.load(path, trusted=sorted(unknown))
    if not isinstance(predictor, (Pipeline, HistGradientBoostingClassifier)):
        raise ValueError("Bundle must contain a supported fitted sklearn predictor")
    if getattr(predictor, "n_features_in_", None) != 590:
        raise ValueError("Predictor must accept exactly 590 original measurements")
    return predictor


def _write_bundle(
    root: Path,
    purpose: str,
    predictor,
    config: dict,
    quality: dict,
    report: dict,
    protocol: dict,
    source_model: Path | None = None,
) -> dict:
    from .study import dependency_versions

    destination = root / "bundles" / purpose
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        existing = load_bundle(destination)
        if (
            existing.manifest["protocol_hash"] != protocol["protocol_hash"]
            or existing.manifest["model_config"] != config
        ):
            raise ValueError(f"Existing {purpose} bundle belongs to another frozen study")
        return existing.manifest
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.skops"
    if source_model:
        shutil.copyfile(source_model, model_path)
        load_verified_model(model_path)
    else:
        save_verified_model(predictor, model_path)
    audit = read_json(root / "data" / "audit.json")
    quality = {
        **quality,
        "protected_ids": audit["protected_ids"],
        "protected_fingerprints": audit["protected_fingerprints"],
    }
    schema = {
        "schema_version": SCHEMA_VERSION,
        "version": SCHEMA_VERSION,
        "n_features": 590,
        "feature_names": list(FEATURE_NAMES),
        "missing_representation": "JSON null / numeric NaN",
        "timestamp_and_label_are_features": False,
    }
    policy = {
        "capacity_fraction": 0.1,
        "rounding": "floor",
        "tie_break": "sha256-record-id",
        "score_kind": "uncalibrated",
        "higher_score": "failure",
        "abstain_missingness": 0.5,
        "closed_batch_only": True,
        "action": "review_priority",
        "limitation": LIMITATION,
    }
    for name, value in (
        ("quality.json", quality),
        ("schema.json", schema),
        ("policy.json", policy),
        ("report.json", report),
    ):
        write_json(destination / name, value)
    data_card = (
        "# SECOM data card\n\nUCI SECOM, Michael McCann and Adrian Johnston, 2008, "
        "CC BY 4.0. https://doi.org/10.24432/C54305\n\n"
        "1,567 rows and 590 anonymous measurement columns; 104 failed in-house tests. "
        "Metadata says 591 columns; the pinned measurement file has 590. "
        "Timestamps are timezone-unspecified test points. No lot identifiers, physical sensor "
        "semantics or label delays are documented.\n\n" + LIMITATION + "\n"
    )
    (destination / "data_card.md").write_text(data_card, encoding="utf-8")
    model_card = (
        f"# Model card: {purpose}\n\nConfiguration: {config['config_id']}. "
        f"Scientific status: {report.get('final_status', 'pending')}.\n\n"
        "Scores rank retrospective records and are not calibrated failure probabilities. "
        "The 10% inspection budget and missingness guard are demonstration assumptions. "
        "No final-driven replacement is allowed.\n\n" + LIMITATION + "\n"
    )
    (destination / "model_card.md").write_text(model_card, encoding="utf-8")
    if purpose == "monitoring_demo":
        development = load_dataset(root)
        frame = pd.DataFrame(development.X, columns=FEATURE_NAMES)
        frame.insert(0, "record_id", development.ids)
        frame["timestamp"] = development.timestamps
        frame["label"] = development.y
        frame.to_parquet(destination / "development.parquet", index=False)
    files = {
        path.name: sha256_file(path) for path in sorted(destination.iterdir()) if path.is_file()
    }
    dependencies = dependency_versions()
    manifest = {
        "bundle_format": "secom-bundle-v1",
        "purpose": purpose,
        "protocol_hash": protocol["protocol_hash"],
        "model_config": config,
        "model_id": f"{config['config_id']}-{files['model.skops'][:12]}",
        "schema_version": SCHEMA_VERSION,
        "files": files,
        "dependencies": dependencies,
        "dependencies_hash": fingerprint(dependencies),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "source_hashes": protocol["source_hashes"],
        "scientific_code_hashes": protocol["scientific_code_hashes"],
        "lockfile_sha256": protocol.get("lockfile_sha256"),
        "limitation": LIMITATION,
    }
    manifest["bundle_id"] = f"{purpose}-{fingerprint(manifest)[:16]}"
    manifest["manifest_hash"] = fingerprint(manifest)
    immutable_json(manifest_path, manifest)
    load_bundle(destination)
    return manifest


def build_bundles(root: Path) -> dict:
    from .study import _checked_protocol, _checked_selection, report_data

    root = Path(root)
    protocol = _checked_protocol(root)
    development = load_dataset(root)
    result = {}
    if (root / "research" / "final_evaluation.json").exists():
        selection = _checked_selection(root, protocol)
        source = root / "research" / "models" / "selected.skops"
        predictor = load_verified_model(source)
        result["main"] = _write_bundle(
            root,
            "main",
            predictor,
            selection["model_config"],
            quality_reference(predictor, development.X),
            report_data(root),
            protocol,
            source,
        )
    else:
        result["main"] = {"status": "pending_final_evaluation"}
    demo_path = root / "bundles" / "monitoring_demo" / "manifest.json"
    if demo_path.exists():
        result["monitoring_demo"] = load_bundle(demo_path.parent).manifest
        return result
    config = reference_config()
    with threadpool_limits(limits=1):
        demo = fit_predictor(config, development.X[:438], development.y[:438])
        reference_scores = score_rows(demo, development.X[438:538])
    audit = read_json(root / "data" / "audit.json")
    report = {
        "limitation": LIMITATION,
        "audit": {k: v for k, v in audit.items() if not k.startswith("protected_")},
        "fold_results": [],
        "selection": {
            "config_id": config["config_id"],
            "role": "fixed monitoring demonstration, never selected on stream",
        },
        "final_metrics": {},
        "final_status": "not_applicable",
        "curves": {},
        "monitoring": {
            "training": [0, 438],
            "reference": [438, 538],
            "stream": [538, 1096],
            "reference_metrics": evaluate_scores(
                development.y[438:538], reference_scores, development.ids[438:538]
            ),
            "label_delay": "Simulation assumption; source supplies no real label-delay metadata.",
        },
    }
    result["monitoring_demo"] = _write_bundle(
        root,
        "monitoring_demo",
        demo,
        config,
        quality_reference(demo, development.X[:438]),
        report,
        protocol,
    )
    return result


def load_bundle(path: Path) -> LoadedBundle:
    from .study import dependency_versions

    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    if fingerprint({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest.get(
        "manifest_hash"
    ):
        raise ValueError("Bundle manifest fingerprint mismatch")
    if (
        manifest.get("bundle_format") != "secom-bundle-v1"
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported bundle format or feature schema")
    dependencies = dependency_versions()
    if manifest["dependencies"] != dependencies or manifest["dependencies_hash"] != fingerprint(
        dependencies
    ):
        raise ValueError("Bundle dependencies do not match this environment")
    if manifest["python"] != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise ValueError("Bundle Python major/minor mismatch")
    required = {
        "model.skops",
        "quality.json",
        "schema.json",
        "policy.json",
        "report.json",
        "data_card.md",
        "model_card.md",
    }
    if manifest["purpose"] == "monitoring_demo":
        required.add("development.parquet")
    if set(manifest["files"]) != required:
        raise ValueError("Bundle file manifest is incomplete or contains unexpected paths")
    for name, expected in manifest["files"].items():
        file = path / name
        if file.is_symlink() or file.resolve().parent != path or sha256_file(file) != expected:
            raise ValueError(f"Bundle integrity check failed: {name}")
    schema = read_json(path / "schema.json")
    if schema["feature_names"] != list(FEATURE_NAMES) or schema["n_features"] != 590:
        raise ValueError("Bundle sensor order differs from the pinned schema")
    predictor = load_verified_model(path / "model.skops")
    return LoadedBundle(
        path,
        manifest,
        predictor,
        read_json(path / "quality.json"),
        schema,
        read_json(path / "policy.json"),
        read_json(path / "report.json"),
    )


def load_development_fixture(bundle: LoadedBundle) -> Dataset:
    """The offline demo needs no raw archive and contains no final-test records."""
    if bundle.manifest["purpose"] != "monitoring_demo":
        raise ValueError("Development fixtures belong only to the monitoring demonstration")
    path = bundle.path / "development.parquet"
    if sha256_file(path) != bundle.manifest["files"]["development.parquet"]:
        raise ValueError("Development fixture changed after bundle verification")
    frame = pd.read_parquet(path)
    expected_columns = {"record_id", "timestamp", "label", *FEATURE_NAMES}
    expected_ids = [f"secom-r{i + 1:06d}" for i in range(DEVELOPMENT_END)]
    if set(frame.columns) != expected_columns or frame["record_id"].tolist() != expected_ids:
        raise ValueError("Demo fixture must contain exactly the frozen development rows")
    X = frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    y = frame["label"].to_numpy(dtype=np.int64)
    timestamps = frame["timestamp"].to_numpy()
    if np.isinf(X).any() or not set(y.tolist()) <= {0, 1} or int(y.sum()) != 78:
        raise ValueError("Development fixture measurements or labels are invalid")
    if frame["timestamp"].isna().any() or not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Development fixture timestamps are invalid")
    return Dataset(X, y, timestamps, expected_ids)
