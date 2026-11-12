"""Geometry, safe decoding and tensor-only predictor round trips."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from secom_quality.vision.transforms import image_array, letterbox, mask_array, read_rgb


def test_letterbox_restores_observed_coordinates_and_excludes_padding():
    image = Image.fromarray(np.zeros((30, 10, 3), dtype=np.uint8))
    canvas, geometry = letterbox(image, (80, 40))
    assert canvas.shape == (80, 40, 3)
    spatial = np.zeros((80, 40), dtype=np.float32)
    spatial[:, : geometry.left] = 999
    spatial[
        geometry.top : geometry.top + geometry.height,
        geometry.left : geometry.left + geometry.width,
    ] = 7
    restored = geometry.restore(spatial)
    assert restored.shape == (30, 10)
    assert np.allclose(restored, 7)
    mask, valid = mask_array(None, geometry)
    assert mask.sum() == 0
    assert valid.sum() == geometry.height * geometry.width


def test_image_decoder_rejects_corruption_and_mask_misalignment(tmp_path: Path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="invalid_image"):
        read_rgb(path)
    Image.new("RGB", (20, 30)).save(path)
    array, geometry = image_array(path, (70, 30))
    assert np.isfinite(array).all()
    mask = tmp_path / "mask.png"
    Image.new("L", (21, 30)).save(mask)
    with pytest.raises(ValueError, match="alignment"):
        mask_array(mask, geometry)


def test_coreset_is_deterministic_unique_and_fraction_bounded():
    pytest.importorskip("torch")
    from secom_quality.vision.models import approximate_coreset

    features = np.random.default_rng(5).normal(size=(80, 12)).astype(np.float32)
    first = approximate_coreset(features, 0.1, 20, 123)
    second = approximate_coreset(features, 0.1, 20, 123)
    assert np.array_equal(first, second)
    assert len(first) == len(set(first)) == 8
    assert len(approximate_coreset(features, 1, 3, 123)) == 3


def test_linear_predictor_tensor_roundtrip_is_offline(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from secom_quality.vision.models import VisionPredictor, _backbone, load_predictor

    torch.set_num_threads(2)
    path = tmp_path / "image.png"
    Image.new("RGB", (24, 60), (80, 120, 170)).save(path)
    config = {"family": "resnet18_logistic", "resolution": [96, 48], "C": 0.1}
    model = VisionPredictor(config, _backbone("resnet18_logistic", pretrained=False))
    model.head = {"coefficients": np.linspace(-0.01, 0.01, 512).tolist(), "intercept": 0.25}
    before_scores, before_maps = model.predict([path])
    destination = tmp_path / "predictor"
    model.save(destination)
    # Loading must not invoke a network downloader or unpickle arbitrary objects.
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *_a, **_k: pytest.fail("network"))
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: pytest.fail("pickle"))
    after_scores, after_maps = load_predictor(destination).predict([path])
    assert np.allclose(before_scores, after_scores, atol=1e-6)
    assert np.allclose(before_maps, after_maps, atol=1e-6)
    assert after_maps[0].shape == (60, 24)


def test_checkpoint_restores_optimizer_scheduler_and_rng(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from secom_quality.vision.models import _restore_checkpoint, _save_checkpoint

    torch.manual_seed(7)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2])
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    expected_weight = model.weight.detach().clone()
    rng = np.random.default_rng(9)
    _save_checkpoint(tmp_path, model, optimizer, scheduler, 1, rng, np.array([1.0, 2.0]))
    expected_random = rng.random()
    with torch.no_grad():
        model.weight.zero_()
    epoch, frequencies = _restore_checkpoint(tmp_path, model, optimizer, scheduler, rng)
    assert epoch == 1
    assert np.array_equal(frequencies, [1, 2])
    assert torch.equal(model.weight, expected_weight)
    assert rng.random() == expected_random
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="incomplete or corrupt"):
        _restore_checkpoint(tmp_path, model, optimizer, scheduler, rng)


def test_fitting_rejects_held_out_partition_before_backbone_download(tmp_path: Path):
    pytest.importorskip("torch")
    import pandas as pd

    from secom_quality.vision.models import fit_model

    train = pd.DataFrame({"image_path": ["unused"], "label": [1], "partition": ["test"]})
    with pytest.raises(ValueError, match="Held-out"):
        fit_model({"family": "resnet18_logistic"}, train, 1, tmp_path, device="cpu")
