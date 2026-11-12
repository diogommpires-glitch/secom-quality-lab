import numpy as np
import pandas as pd
import pytest

from secom_quality.artifacts import load_development_fixture
from secom_quality.common import sha256_file
from secom_quality.contracts import FEATURE_NAMES, LoadedBundle


def test_offline_fixture_accepts_only_ordered_development_records(tmp_path):
    frame = pd.DataFrame(np.zeros((1096, 590)), columns=FEATURE_NAMES)
    frame["record_id"] = [f"secom-r{i + 1:06d}" for i in range(1096)]
    frame["timestamp"] = pd.date_range("2008-07-19", periods=1096, freq="h")
    frame["label"] = np.asarray([1] * 78 + [0] * 1018)
    path = tmp_path / "development.parquet"
    frame.to_parquet(path, index=False)
    manifest = {"purpose": "monitoring_demo", "files": {"development.parquet": sha256_file(path)}}
    bundle = LoadedBundle(tmp_path, manifest, None, {}, {}, {}, {})
    result = load_development_fixture(bundle)
    assert result.X.shape == (1096, 590)
    assert result.y.sum() == 78
    frame.loc[1095, "record_id"] = "secom-r001097"
    frame.to_parquet(path, index=False)
    manifest["files"]["development.parquet"] = sha256_file(path)
    with pytest.raises(ValueError, match="development rows"):
        load_development_fixture(bundle)


def test_offline_fixture_rejects_changed_bytes(tmp_path):
    path = tmp_path / "development.parquet"
    path.write_bytes(b"corrupt")
    bundle = LoadedBundle(
        tmp_path,
        {"purpose": "monitoring_demo", "files": {"development.parquet": "expected"}},
        None,
        {},
        {},
        {},
        {},
    )
    with pytest.raises(ValueError, match="fixture changed"):
        load_development_fixture(bundle)
