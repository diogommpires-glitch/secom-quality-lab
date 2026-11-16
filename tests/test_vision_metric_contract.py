"""Checked local PRO area, with historical interpolation reported separately."""

import numpy as np
import pytest
from PIL import Image

from secom_quality.vision.ad2 import au_pro_30
from secom_quality.vision.metrics import localization_metrics


def test_vertical_pro_segment_at_cutoff_has_zero_area(tmp_path):
    mask = np.array([[1, 1] + [0] * 10], dtype=np.uint8)
    score = np.array([[1.0, 0.5, 0.75, 0.75, 0.75] + [0.0] * 7], dtype=np.float32)
    Image.fromarray(mask * 255).save(tmp_path / "mask.png")
    np.save(tmp_path / "map.npy", score, allow_pickle=False)
    records = [{"mask_path": "mask.png", "map_path": str(tmp_path / "map.npy")}]
    for result in (localization_metrics([mask], [score]), au_pro_30(records, tmp_path)):
        assert result["au_pro_30"] == pytest.approx(0.5)
        assert result["au_pro_30_legacy"] == pytest.approx(0.75)
        assert result["metric_revision"] == "FPR-boundary-left-limit-v1"


def test_metric_correction_does_not_modify_frozen_legacy_bytes(tmp_path):
    history = tmp_path / "historical-report.json"
    before = b'{"au_pro_30":0.75,"method":"legacy"}\n'
    history.write_bytes(before)
    mask = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    result = localization_metrics([mask], [mask.astype(float)])
    assert result["au_pro_30"] == pytest.approx(1)
    assert result["au_pro_30_legacy"] == pytest.approx(1)
    assert history.read_bytes() == before


def test_large_perfect_region_only_clamps_float64_accumulation_roundoff(tmp_path):
    mask = np.array([[1] * 1_000_000 + [0] * 10], dtype=np.uint8)
    Image.fromarray(mask * 255).save(tmp_path / "mask.png")
    np.save(tmp_path / "map.npy", mask.astype(np.float32), allow_pickle=False)
    result = au_pro_30([{"mask_path": "mask.png", "map_path": str(tmp_path / "map.npy")}], tmp_path)
    assert result["au_pro_30"] == 1.0
    assert result["au_pro_30_legacy"] == pytest.approx(1.0, abs=1e-8)
