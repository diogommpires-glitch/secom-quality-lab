"""Bundle trust is verified even when network reconstruction is cached."""

from types import SimpleNamespace

import pytest

from secom_quality.common import fingerprint, read_json, write_json
from secom_quality.vision import bundle, models


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "dependency_versions", lambda: {"test": "1"})
    monkeypatch.setattr(models, "load_predictor", lambda *a, **kw: SimpleNamespace(predict=None))
    research = tmp_path / "vision/research"
    selection = {"config": {"config_id": "fixture"}, "seeds": [1], "protocol_hash": "fixture"}
    selection["selection_hash"] = fingerprint(selection)
    write_json(research / "selection.json", selection)
    write_json(research / "final/report.json", {"selection_hash": selection["selection_hash"]})
    model = research / "final/models/1"
    write_json(model / "predictor.json", {"fixture": True})
    (model / "weights.safetensors").write_bytes(b"explicit synthetic tensor fixture")
    (model / "checkpoint.safetensors").write_bytes(b"not an inference artifact")
    write_json(model / "training_history.json", {"must_not_ship": True})
    bundle.build_bundle(tmp_path)
    return tmp_path / "vision/bundles/main"


def test_only_inference_files_ship_and_cache_rechecks_every_byte(built):
    assert not (built / "predictor/checkpoint.safetensors").exists()
    assert not (built / "predictor/training_history.json").exists()
    first = bundle.load_verified_bundle(built)
    assert first is bundle.load_verified_bundle(built)
    (built / "predictor/weights.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        bundle.load_verified_bundle(built)


def test_unexpected_files_and_manifest_changes_fail_readiness(built):
    manifest = read_json(built / "manifest.json")
    manifest["model_id"] = "changed"
    write_json(built / "manifest.json", manifest)
    with pytest.raises(ValueError, match="identity"):
        bundle.load_verified_bundle(built)
    manifest["model_id"] = "fixture"
    write_json(built / "manifest.json", manifest)
    (built / "surprise.pkl").write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="unexpected"):
        bundle.load_verified_bundle(built)
