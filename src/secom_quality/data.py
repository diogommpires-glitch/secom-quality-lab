"""Pinned source ingestion and the development/final data boundary."""

from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .common import fingerprint, measurement_fingerprint, read_json, sha256_file, write_json
from .contracts import ARCHIVE_SHA256, ARCHIVE_URL, DEVELOPMENT_END, FOLDS, Dataset

FILE_HASHES = {
    "secom.data": "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c",
    "secom_labels.data": "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03",
}


def immutable_json(path: Path, value: dict) -> None:
    """A changed scientific record requires a new workspace, never an overwrite."""
    if path.exists():
        if fingerprint(read_json(path)) != fingerprint(value):
            raise ValueError(f"Immutable record differs: {path}; use a new workspace")
        return
    write_json(path, value)


def fetch_data(root: Path) -> dict:
    root = Path(root)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "secom.zip"
    if not archive.exists():
        request = Request(ARCHIVE_URL, headers={"User-Agent": "SECOM-Quality-Lab/1.0"})
        with urlopen(request, timeout=60) as response:
            content = response.read(10_000_001)
        if len(content) > 10_000_000:
            raise ValueError("SECOM archive exceeded the 10 MB download limit")
        import hashlib

        if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("SECOM archive SHA-256 mismatch; nothing was extracted")
        temporary = archive.with_suffix(".download")
        temporary.write_bytes(content)
        temporary.replace(archive)
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("Existing archive SHA-256 mismatch")
    with ZipFile(archive) as source:
        expected = {"secom.data", "secom_labels.data", "secom.names"}
        if set(source.namelist()) != expected:
            raise ValueError("Archive must contain exactly the three named SECOM files")
        if sum(info.file_size for info in source.infolist()) > 8_000_000:
            raise ValueError("SECOM archive exceeds the 8 MB extraction limit")
        for name in sorted(expected):
            destination = raw / name
            content = source.read(name)
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError(f"Existing pinned source differs: {name}")
            if not destination.exists():
                destination.write_bytes(content)
    return verify_data(root)


def verify_data(root: Path) -> dict:
    raw = Path(root) / "data" / "raw"
    if sha256_file(raw / "secom.zip") != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 mismatch")
    hashes = {name: sha256_file(raw / name) for name in (*FILE_HASHES, "secom.names")}
    for name, expected in FILE_HASHES.items():
        if hashes[name] != expected:
            raise ValueError(f"Pinned file SHA-256 mismatch: {name}")
    return {"archive_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA256, "files": hashes}


def _raw_dataset(root: Path) -> Dataset:
    verify_data(root)
    raw = Path(root) / "data" / "raw"
    X = np.loadtxt(raw / "secom.data", dtype=np.float64)
    labels = pd.read_csv(
        raw / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["label", "timestamp"],
        quotechar='"',
    )
    if X.shape != (1567, 590) or len(labels) != 1567:
        raise ValueError(f"Expected measurements (1567, 590) and 1567 labels, got {X.shape}")
    if np.isinf(X).any() or not set(labels.label.unique()) <= {-1, 1}:
        raise ValueError("Infinite measurements or unexpected label encoding")
    timestamps = pd.to_datetime(labels.timestamp, format="%d/%m/%Y %H:%M:%S", errors="raise")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Pinned timestamps must be complete and chronological")
    y = (labels.label.to_numpy() == 1).astype(np.int64)
    ids = [f"secom-r{i + 1:06d}" for i in range(len(X))]
    return Dataset(X, y, timestamps.to_numpy(), ids)


def slice_dataset(data: Dataset, start: int, end: int) -> Dataset:
    return Dataset(
        data.X[start:end].copy(),
        data.y[start:end].copy(),
        data.timestamps[start:end].copy(),
        data.ids[start:end],
    )


def load_dataset(root: Path, partition: str = "development") -> Dataset:
    """Public consumers only receive development rows; final access is ledgered by study."""
    if partition != "development":
        raise PermissionError(
            "Final rows are available only through evaluate_final after selection freeze"
        )
    return slice_dataset(_raw_dataset(root), 0, DEVELOPMENT_END)


def _partition(data: Dataset, start: int, end: int) -> dict:
    return {
        "n_samples": end - start,
        "n_failures": int(data.y[start:end].sum()),
        "start": str(data.timestamps[start]),
        "end": str(data.timestamps[end - 1]),
        "start_index": start,
        "end_index": end,
    }


def audit_data(root: Path) -> dict:
    """Structural and class-count audit; no predictive test exploration is performed."""
    root = Path(root)
    data = _raw_dataset(root)
    _, tie_sizes = np.unique(data.timestamps, return_counts=True)
    row_hashes = [measurement_fingerprint(row) for row in data.X]
    duplicate_groups = len(row_hashes) - len(set(row_hashes))
    if duplicate_groups:
        raise ValueError("Unexpected duplicate vectors require provenance review before splitting")
    for boundary in (438, 767, DEVELOPMENT_END):
        if data.timestamps[boundary - 1] >= data.timestamps[boundary]:
            raise ValueError(f"Timestamp group crosses fixed boundary {boundary}")
    result = {
        "n_samples": 1567,
        "n_features": 590,
        "class_counts": {"pass": int((data.y == 0).sum()), "fail": int(data.y.sum())},
        "missing_cells": int(np.isnan(data.X).sum()),
        "missingness_by_feature": np.isnan(data.X[:DEVELOPMENT_END]).mean(axis=0).tolist(),
        "missingness_by_feature_scope": "development rows 0:1096 only",
        "timestamp_min": str(data.timestamps[0]),
        "timestamp_max": str(data.timestamps[-1]),
        "timezone_status": "unspecified",
        "unique_timestamps": len(tie_sizes),
        "timestamp_tie_groups": int((tie_sizes > 1).sum()),
        "max_timestamp_tie": int(tie_sizes.max()),
        "exact_duplicate_groups": duplicate_groups,
        "metadata_discrepancy": "UCI description says 591 features; pinned measurement file has 590 columns; labels and dates are separate.",
        "partitions": {
            "development": _partition(data, 0, DEVELOPMENT_END),
            "final": _partition(data, DEVELOPMENT_END, len(data.y)),
        },
        "folds": [
            {
                "training_end": start,
                "validation_end": end,
                "train_size": start,
                "validation_size": end - start,
                "train_failures": int(data.y[:start].sum()),
                "validation_failures": int(data.y[start:end].sum()),
            }
            for start, end in FOLDS
        ],
        "source_hashes": verify_data(root),
        "audit_scope": "Schema, integrity, chronology and class-count feasibility; no predictive final results.",
        "protected_ids": data.ids[DEVELOPMENT_END:],
        "protected_fingerprints": row_hashes[DEVELOPMENT_END:],
    }
    immutable_json(root / "data" / "audit.json", result)
    return result
