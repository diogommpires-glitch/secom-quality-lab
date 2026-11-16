"""HTTP workflows exercise the same core used by CLI and monitoring."""

import asyncio
import threading
from dataclasses import replace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_application_inference import json_record, make_bundle

from secom_quality import artifacts
from secom_quality.api import create_app
from secom_quality.contracts import Dataset, LoadedBundle
from secom_quality.inference import parse_csv, record_from_values, records_csv, score_records


@pytest.fixture
def client(tmp_path, monkeypatch):
    main = make_bundle(tmp_path)
    demo = LoadedBundle(
        main.path,
        {**main.manifest, "purpose": "monitoring_demo", "bundle_id": "test-demo"},
        main.predictor,
        main.quality,
        main.schema,
        main.policy,
        main.report,
    )
    monkeypatch.setattr(
        artifacts, "load_bundle", lambda path: demo if path.name == "monitoring_demo" else main
    )
    X = np.random.default_rng(42).normal(size=(1096, 590))
    dataset = Dataset(
        X,
        (np.arange(1096) % 10 == 0).astype(int),
        np.arange(1096),
        [f"dev-{i}" for i in range(1096)],
    )
    monkeypatch.setattr(artifacts, "load_development_fixture", lambda bundle: dataset)
    with TestClient(create_app(tmp_path)) as current:
        yield current, main


def test_api_row_matches_shared_core_and_has_no_probability_claim(client):
    http, bundle = client
    request = json_record()
    actual = http.post("/api/v1/predictions", json=request)
    assert actual.status_code == 200
    expected = score_records([record_from_values("sample-1", np.zeros(590))], bundle)[0]
    assert actual.json()["risk_score"] == expected["risk_score"]
    assert actual.json()["calibrated_probability"] is None
    assert "raw_score" not in actual.json()
    assert http.get("/health/ready").json()["status"] == "ready"


def test_api_upload_retry_export_and_protected_rows(client):
    http, _ = client
    records = [record_from_values(f"row-{i}", np.zeros(590)) for i in range(20)]
    content = records_csv(records)
    response = http.post(
        "/api/v1/batches",
        data={"client_request_id": "api-request"},
        files={"file": ("records.csv", content, "text/csv")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["review_capacity"] == 2
    retry = http.post(
        "/api/v1/batches",
        data={"client_request_id": "api-request"},
        files={"file": ("records.csv", content, "text/csv")},
    )
    assert retry.json()["batch_id"] == result["batch_id"]
    assert retry.json()["replayed"] is True
    exported = http.get(f"/api/v1/batches/{result['batch_id']}/export")
    assert exported.status_code == 200 and "risk_score" in exported.text
    protected = http.post("/api/v1/predictions", json=json_record("final-1"))
    assert protected.status_code == 403
    assert protected.json()["detail"]["code"] == "protected_test_record"


def test_batch_keeps_liveness_responsive_until_scoring_finishes(tmp_path):
    started, release = threading.Event(), threading.Event()
    main = make_bundle(tmp_path)

    class WaitingPredictor:
        classes_ = np.array([0, 1])

        def predict_proba(self, values):
            started.set()
            release.wait(timeout=2)
            return np.tile([0.5, 0.5], (len(values), 1))

    main = replace(main, predictor=WaitingPredictor())
    app = create_app(tmp_path)
    app.state.bundle = main
    content = records_csv([record_from_values("synthetic", np.zeros(590))])

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            scoring = asyncio.create_task(
                http.post(
                    "/api/v1/batches",
                    data={"client_request_id": "liveness"},
                    files={"file": ("records.csv", content, "text/csv")},
                )
            )
            try:
                assert await asyncio.to_thread(started.wait, 3)
                assert (await http.get("/health/live")).status_code == 200
                assert not scoring.done(), "Liveness was blocked until scoring finished"
            finally:
                release.set()
                response = await scoring
            assert response.status_code == 200

    asyncio.run(exercise())


def test_api_errors_and_demo_download_are_explicit(client):
    http, _ = client
    error = http.post("/api/v1/predictions", json={"record_id": "broken", "sensors": {}})
    assert error.status_code == 422
    assert error.json()["detail"]["code"] == "sensor_schema"
    assert error.json()["detail"]["request_id"] == error.headers["x-request-id"]
    unknown = http.get("/api/v1/training")
    assert unknown.status_code == 404
    assert unknown.headers["content-type"].startswith("application/json")
    records = parse_csv(http.get("/api/v1/demo/inputs").content)
    assert len(records) == 20
    assert all(record["record_id"].startswith("dev-") for record in records)


def test_api_replay_uses_isolated_bundle_and_cursor_contract(client):
    http, _ = client
    created = http.post("/api/v1/demo/replays", json={"scenario": "missingness"})
    assert created.status_code == 200
    stream = created.json()
    assert stream["bundle_id"] == "test-demo"
    assert "history" not in stream and "reference_features" not in stream
    advanced = http.post(
        f"/api/v1/demo/replays/{stream['stream_id']}/advance",
        json={"expected_cursor": 0, "steps": 20},
    )
    assert advanced.status_code == 200
    assert advanced.json()["cursor"] == 20
    assert advanced.json()["latest_window"]["matured_label_count"] == 0


def test_streamed_request_limit_is_enforced_without_content_length(client):
    http, _ = client
    payload = (chunk for chunk in [b"x" * (1024 * 1024)] * 22)
    response = http.post(
        "/api/v1/predictions", content=payload, headers={"content-type": "application/json"}
    )
    assert response.status_code == 413
