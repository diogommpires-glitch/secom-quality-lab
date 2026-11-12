"""Vision boundary tests use an explicitly synthetic predictor, never benchmark scores."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from secom_quality.api import create_app
from secom_quality.inference import ApplicationError
from secom_quality.vision import api as vision_api
from secom_quality.vision import inference
from secom_quality.vision.storage import VisionStore


def png() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (24, 32), (130, 120, 100)).save(stream, format="PNG")
    return stream.getvalue()


def archive(count: int = 10) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        for index in range(count):
            zipped.writestr(f"image-{index:02d}.png", png())
    return stream.getvalue()


class SyntheticPredictor:
    def predict(self, paths: list[Path]):
        return np.ones(len(paths)), [np.zeros((32, 24), dtype=np.float32) for _ in paths]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        inference,
        "load_active",
        lambda root: {
            "manifest": {"bundle_id": "synthetic-test", "model_id": "synthetic-test"},
            "predictor": SyntheticPredictor(),
            "report": {},
        },
    )
    # API module holds its own imported reference for readiness.
    monkeypatch.setattr(vision_api, "load_active", inference.load_active)
    with TestClient(create_app(tmp_path)) as http:
        yield http, tmp_path


def test_queue_retry_assets_and_safe_export(client):
    http, root = client
    data = archive()
    response = http.post(
        "/api/v2/vision/batches",
        files={"file": ("batch.zip", data)},
        data={"client_request_id": "same", "capacity": "0.1"},
    )
    assert response.status_code == 200
    batch = response.json()
    assert batch["review_capacity"] == 1
    assert batch["eligible_count"] == 10
    assert sum(row["review_recommended"] for row in batch["predictions"]) == 1
    order = sorted(batch["predictions"], key=lambda row: row["rank"])
    import hashlib

    assert [row["record_id"] for row in order] == sorted(
        [row["record_id"] for row in order],
        key=lambda identity: hashlib.sha256(identity.encode()).hexdigest(),
    )
    retry = http.post(
        "/api/v2/vision/batches",
        files={"file": ("batch.zip", data)},
        data={"client_request_id": "same", "capacity": "0.1"},
    ).json()
    assert retry["batch_id"] == batch["batch_id"] and retry["replayed"]
    conflict = http.post(
        "/api/v2/vision/batches",
        files={"file": ("batch.zip", data)},
        data={"client_request_id": "same", "capacity": "0.2"},
    )
    assert conflict.status_code == 409
    assert len(VisionStore(root).list()) == 1
    for url in (batch["predictions"][0]["image_url"], batch["predictions"][0]["overlay_url"]):
        asset = http.get(url)
        assert asset.status_code == 200 and asset.headers["content-type"] == "image/png"
    exported = http.get(f"/api/v2/vision/batches/{batch['batch_id']}/export")
    assert exported.status_code == 200 and "record_id,risk_score" in exported.text
    paged = http.get(
        f"/api/v2/vision/batches/{batch['batch_id']}/predictions?offset=2&limit=3"
    ).json()
    assert len(paged["predictions"]) == 3 and paged["total"] == 10


def test_atomic_corrupt_batch_and_tiny_capacity(client):
    http, root = client
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        zipped.writestr("good.png", png())
        zipped.writestr("bad.png", b"not an image")
    response = http.post(
        "/api/v2/vision/batches",
        files={"file": ("batch.zip", stream.getvalue())},
        data={"client_request_id": "invalid"},
    )
    assert response.status_code == 422
    assert not VisionStore(root).list()
    assert not (root / "vision" / "assets").exists()
    tiny = http.post(
        "/api/v2/vision/batches",
        files={"file": ("tiny.png", png())},
        data={"client_request_id": "tiny"},
    ).json()
    assert tiny["review_capacity"] == 0 and tiny["eligible_count"] == 1
    assert tiny["predictions"][0]["calibration_status"] == "not_validated"


@pytest.mark.parametrize(
    "names", [["../escape.png"], ["a.png", "a.jpg"], ["a/b.png"], ["a\\b.png"]]
)
def test_archive_paths_and_duplicate_ids_fail(names):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        for name in names:
            zipped.writestr(name, png())
    with pytest.raises(ApplicationError):
        inference.parse_upload(stream.getvalue(), "batch.zip")


def test_magic_pixels_and_nonfinite_output(client, monkeypatch):
    with pytest.raises(ApplicationError, match="format"):
        inference.decode_image(png(), "wrong.jpg")
    stream = io.BytesIO()
    Image.new("RGB", (2, 2)).save(stream, format="PNG")
    with pytest.raises(ApplicationError, match="pixels"):
        inference.decode_image(stream.getvalue(), "tiny.png")
    http, _ = client
    monkeypatch.setattr(
        SyntheticPredictor,
        "predict",
        lambda self, paths: (np.array([float("inf")]), [np.zeros((32, 24))]),
    )
    result = http.post("/api/v2/vision/predictions", files={"file": ("one.png", png())})
    assert result.status_code == 200
    assert result.json()["risk_score"] is None
    assert result.json()["quality_status"] == "abstain"
    assert "nonfinite_score" in result.json()["quality_reason_codes"]


def test_csv_formulas_are_escaped():
    text = inference.export_csv({"predictions": [{"record_id": "=cmd()", "risk_score": 0.1}]})
    assert "'=cmd()" in text


def test_outcome_form_fields_and_oversized_upload_fail(client):
    http, root = client
    response = http.post(
        "/api/v2/vision/batches",
        files={"file": ("one.png", png())},
        data={"client_request_id": "no-outcome", "label": "failure"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "request_fields"
    assert not VisionStore(root).list()
    response = http.post(
        "/api/v2/vision/batches",
        content=b"x",
        headers={"Content-Length": str(inference.MAX_BYTES + 2 * 1024 * 1024)},
    )
    assert response.status_code == 413


def test_missing_vision_readiness_does_not_hide_sensor_routes(tmp_path):
    with TestClient(create_app(tmp_path)) as http:
        response = http.get("/api/v2/vision/status")
        assert response.status_code == 503 and response.json()["ready"] is False
        assert http.get("/health/live").status_code == 200
        assert http.get("/api/v2/vision/unknown").status_code == 404
