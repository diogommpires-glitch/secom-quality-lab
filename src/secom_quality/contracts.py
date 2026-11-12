"""Shared identities and the boundary between research and serving."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

SEED = 20261026
FEATURE_NAMES = tuple(f"sensor_{i:03d}" for i in range(590))
SCHEMA_VERSION = "secom-sensors-v1"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
ARCHIVE_SHA256 = "eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e"
DEVELOPMENT_END = 1096
FOLDS = ((438, 767), (767, 1096))
LIMITATION = (
    "Retrospective research demonstration using historical anonymous measurements. "
    "Risk scores are uncalibrated; results do not establish early warning, "
    "physical causes or factory deployment readiness."
)


@dataclass(frozen=True)
class Dataset:
    X: npt.NDArray[np.float64]
    y: npt.NDArray[np.int64]
    timestamps: npt.NDArray[Any]
    ids: list[str]


@dataclass(frozen=True)
class LoadedBundle:
    path: Path
    manifest: dict[str, Any]
    predictor: Any
    quality: dict[str, Any]
    schema: dict[str, Any]
    policy: dict[str, Any]
    report: dict[str, Any]


def workspace_path(value: str | Path | None = None) -> Path:
    """Keep mutable data outside package resources and source control."""
    return Path(value or "workspace").resolve()
