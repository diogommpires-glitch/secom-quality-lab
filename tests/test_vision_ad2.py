"""Synthetic-only AD2 fixtures test boundaries, native geometry and lifecycle."""

import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from secom_quality.common import fingerprint, read_json, sha256_file, write_json
from secom_quality.vision import ad2


def image_bytes(defect=False, mask=False):
    values = (
        np.zeros((32, 24), dtype=np.uint8) if mask else np.full((32, 24, 3), 80, dtype=np.uint8)
    )
    if defect:
        values[8:12, 10:14] = 255 if mask else 220
    stream = io.BytesIO()
    Image.fromarray(values).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(ad2, "EXPECTED_COUNTS", dict.fromkeys(ad2.CATEGORIES, (2, 1, 1, 1, 1)))
    archive = tmp_path / "synthetic-ad2.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for category in ad2.CATEGORIES:
            files = {
                "train/good/000.png": image_bytes(),
                "train/good/001.png": image_bytes(),
                "validation/good/000.png": image_bytes(),
                "test_public/good/000.png": image_bytes(),
                "test_public/bad/000.png": image_bytes(True),
                "test_public/ground_truth/bad/000_mask.png": image_bytes(True, True),
                "test_private/000_regular.png": image_bytes(),
                "test_private_mixed/000_mixed.png": image_bytes(True),
            }
            for name, body in files.items():
                zipped.writestr(f"mvtec_ad_2/{category}/{name}", body)
    return archive


def selection(root):
    config = {
        "family": "patchcore",
        "config_id": "synthetic-reference",
        "resolution": [704, 256],
        "coreset_fraction": 0.01,
        "max_memory_patches": 20000,
    }
    value = {
        "config": {"family": "supersimplenet", "config_id": "synthetic-selected"},
        "reference_configs": [config],
        "seeds": [20261026],
        "protocol_hash": "synthetic-ksdd2",
    }
    value["selection_hash"] = fingerprint(value)
    write_json(root / "vision" / "research" / "selection.json", value)


def test_audit_quarantines_test_pixels_and_rejects_modified_source(source, tmp_path, monkeypatch):
    original = Image.open

    def only_normal(path, *args, **kwargs):
        assert "protected" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", only_normal)
    root = tmp_path / "workspace"
    manifest = ad2.audit_ad2(root, source)
    assert manifest["test_pixels_decoded"] is False
    assert set(manifest["counts"]) == set(ad2.CATEGORIES)
    normal = ad2._development(ad2._base(root))
    assert len(normal) == 24 and all(row["label"] == 0 for row in normal)
    assert all(row["image_path"].startswith("development/") for row in normal)
    assert ad2.audit_ad2(root, source) == manifest
    (ad2._base(root) / normal[0]["image_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        ad2.audit_ad2(root, source)


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_archive_traversal_and_links_are_rejected(tmp_path, kind):
    archive = tmp_path / ("malicious." + kind)
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("../escape.png", image_bytes())
    else:
        with tarfile.open(archive, "w") as tar:
            member = tarfile.TarInfo("can/train/good/000.png")
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
            tar.addfile(member)
    with pytest.raises(ValueError, match=r"Unsafe|links"):
        ad2.audit_ad2(tmp_path / "workspace", archive)
    assert not (tmp_path / "escape.png").exists()
    assert not (tmp_path / "workspace" / "vision" / "ad2").exists()


def test_missing_data_and_selection_are_actionable(tmp_path, source):
    with pytest.raises(FileNotFoundError, match="pending external data"):
        ad2.run_ad2(tmp_path)
    ad2.audit_ad2(tmp_path, source)
    with pytest.raises(ValueError, match="Freeze the KSDD2"):
        ad2.freeze_ad2(tmp_path)
    assert not (ad2._base(tmp_path) / "research" / "public_test_access.json").exists()


class SyntheticPredictor:
    """Returns input red-channel pixels to test stitching, without learned weights."""

    def predict(self, paths):
        maps = []
        for path in paths:
            with Image.open(path) as image:
                maps.append(np.asarray(image, dtype=np.float32)[:, :, 0] / 255)
        return np.asarray([values.max() for values in maps]), maps


def test_native_tiles_cover_every_pixel_and_stitch_without_resize(tmp_path):
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (700, 550, 3), dtype=np.uint8)
    path = tmp_path / "native.png"
    Image.fromarray(image).save(path)
    coverage = np.zeros((700, 550), dtype=int)
    for x, y, width, height in ad2.tile_coordinates(550, 700):
        coverage[y : y + height, x : x + width] += 1
    assert coverage.min() >= 1 and coverage.max() > 1
    score, values = ad2.predict_tiled(SyntheticPredictor(), path, tmp_path / "tiles")
    np.testing.assert_allclose(values, image[:, :, 0] / 255, atol=1e-7)
    assert values.shape == (700, 550) and score == float(values.max())
    assert not list((tmp_path / "tiles").glob("*.png"))


def test_region_metric_handles_perfect_and_undefined_maps(tmp_path):
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    Image.fromarray(mask).save(tmp_path / "mask.png")
    np.save(tmp_path / "map.npy", mask.astype(np.float32))
    records = [{"mask_path": "mask.png", "map_path": str(tmp_path / "map.npy")}]
    result = ad2.au_pro_30(records, tmp_path)
    assert result["au_pro_30"] == pytest.approx(1)
    assert result["regions"] == 1
    np.save(tmp_path / "map.npy", np.full(mask.shape, 0.5, dtype=np.float32))
    assert ad2.au_pro_30(records, tmp_path)["au_pro_30"] == pytest.approx(0.15)
    np.save(tmp_path / "map.npy", (mask == 0).astype(np.float32))
    assert ad2.au_pro_30(records, tmp_path)["au_pro_30"] == pytest.approx(0)
    diagonal = np.zeros((8, 8), dtype=np.uint8)
    diagonal[2, 2] = diagonal[3, 3] = 255
    Image.fromarray(diagonal).save(tmp_path / "mask.png")
    np.save(tmp_path / "map.npy", diagonal.astype(np.float32))
    assert ad2.au_pro_30(records, tmp_path)["regions"] == 1
    result = ad2.au_pro_30([{**records[0], "mask_path": None}], tmp_path)
    assert result["au_pro_30"] is None


@pytest.fixture
def completed(tmp_path, source, monkeypatch):
    from secom_quality.vision import models

    root = tmp_path / "workspace"
    ad2.audit_ad2(root, source)
    selection(root)
    calls = []

    def fit(config, train, seed, output, device):
        assert config["family"] == "patchcore" and config["resolution"] == [512, 512]
        assert set(train.label) == {0} and set(train.partition) == {"train"}
        assert all("/train/good/" in identity for identity in train.record_id)
        assert seed == 20261026
        calls.append(list(train.record_id))
        output.mkdir(parents=True)
        write_json(output / "predictor.json", {"synthetic_fixture_only": True})
        return SyntheticPredictor()

    monkeypatch.setattr(models, "fit_model", fit)
    monkeypatch.setattr(models, "load_predictor", lambda path, device: SyntheticPredictor())
    original_read = ad2.read_json
    original_open = Image.open

    def require_all_models():
        assert len(calls) == 8, "Test access preceded completion of all eight models"
        access = original_read(ad2._base(root) / "research" / "public_test_access.json")
        assert set(access["models"]) == set(ad2.CATEGORIES)
        assert access["models_hash"] == fingerprint(access["models"])

    def guarded_read(path):
        if Path(path) == ad2._base(root) / "protected" / "manifest.json":
            require_all_models()
        return original_read(path)

    def guarded_open(path, *args, **kwargs):
        if "/protected/" in str(path).replace("\\", "/"):
            require_all_models()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(ad2, "read_json", guarded_read)
    monkeypatch.setattr(Image, "open", guarded_open)
    report = ad2.run_ad2(root, "cpu")
    return root, report, calls


def test_all_categories_normal_only_frozen_lifecycle_and_resume(completed):
    root, report, calls = completed
    assert len(calls) == 8 and len(report["categories"]) == 8
    assert report["external_submission_performed"] is False
    assert {row["category"] for row in report["categories"]} == set(ad2.CATEGORIES)
    protocol = read_json(ad2._base(root) / "research" / "protocol.json")
    assert protocol["ksdd2_config"]["resolution"] == [704, 256]
    assert protocol["config"]["resolution"] == [512, 512]
    assert ad2.run_ad2(root, "cpu") == report
    assert len(calls) == 8
    changed = read_json(root / "vision" / "research" / "selection.json")
    changed["reference_configs"][0]["coreset_fraction"] = 0.05
    changed["selection_hash"] = fingerprint(
        {key: value for key, value in changed.items() if key != "selection_hash"}
    )
    write_json(root / "vision" / "research" / "selection.json", changed)
    with pytest.raises(ValueError, match="Immutable"):
        ad2.run_ad2(root, "cpu")


def test_private_export_is_float16_original_size_without_external_submission(completed):
    tifffile = pytest.importorskip("tifffile")
    root, _, _ = completed
    result = ad2.export_ad2(root, "cpu")
    assert result["external_submission_performed"] is False
    assert len(result["files"]) == 16
    path = Path(result["directory"]) / "anomaly_images/can/test_private/000_regular.tiff"
    values = tifffile.imread(path)
    assert values.shape == (32, 24) and values.dtype == np.float16
    assert np.isfinite(values).all()
    assert ad2.export_ad2(root, "cpu") == result
    with tarfile.open(result["archive"]) as archive:
        assert len(archive.getnames()) == 16
        assert all(
            name.startswith("private_submission/anomaly_images/") for name in archive.getnames()
        )


def test_all_eight_models_are_bound_before_public_manifest_or_pixels(completed):
    # The completed fixture guards both the protected manifest and Image.open;
    # either access before all8models+theirledger causes an assertion failure.
    root, report, calls = completed
    access = read_json(ad2._base(root) / "research" / "public_test_access.json")
    assert len(calls) == 8 and set(access["models"]) == set(ad2.CATEGORIES)
    assert report["access_hash"] == access["access_hash"]
    for files in access["models"].values():
        assert "ad2_fit.json" in files and "predictor.json" in files


@pytest.mark.parametrize("cached_report", [True, False])
def test_missing_model_after_test_access_never_refits(completed, monkeypatch, cached_report):
    from secom_quality.vision import models

    root, _, calls = completed
    research = ad2._base(root) / "research"
    (research / "models" / "walnuts" / "predictor.json").unlink()
    if not cached_report:
        (research / "report.json").unlink()

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("A model was refitted after public-test exposure")

    monkeypatch.setattr(models, "fit_model", forbidden_fit)
    with pytest.raises(ValueError, match=r"missing|checksum"):
        ad2.run_ad2(root, "cpu")
    assert len(calls) == 8


def test_changed_model_cannot_be_rebound_by_repairing_fit_metadata(completed):
    root, _, _ = completed
    directory = ad2._base(root) / "research" / "models" / "can"
    write_json(directory / "predictor.json", {"synthetic_model_changed": True})
    fit = read_json(directory / "ad2_fit.json")
    fit["files"]["predictor.json"] = sha256_file(directory / "predictor.json")
    fit["fit_hash"] = fingerprint({key: value for key, value in fit.items() if key != "fit_hash"})
    write_json(directory / "ad2_fit.json", fit)
    with pytest.raises(ValueError, match="model bytes changed"):
        ad2.run_ad2(root, "cpu")


def test_cached_report_rejects_unexpected_public_files(completed):
    root, _, _ = completed
    (ad2._base(root) / "research" / "public" / "unexpected.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="missing or unexpected"):
        ad2.run_ad2(root, "cpu")
