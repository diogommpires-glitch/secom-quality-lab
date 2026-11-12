import copy

import numpy as np
import pytest
from test_application_inference import make_bundle

from secom_quality.contracts import Dataset, LoadedBundle
from secom_quality.inference import ApplicationError
from secom_quality.monitoring import ReplayService


@pytest.fixture
def bundle(tmp_path):
    return make_bundle(tmp_path)


def make_service(tmp_path, bundle):
    manifest = {**bundle.manifest, "purpose": "monitoring_demo", "bundle_id": "test-demo"}
    demo = LoadedBundle(
        bundle.path,
        manifest,
        bundle.predictor,
        copy.deepcopy(bundle.quality),
        bundle.schema,
        bundle.policy,
        bundle.report,
    )
    X = np.random.default_rng(42).normal(size=(1096, 590))
    y = (np.arange(1096) % 10 == 0).astype(int)
    dataset = Dataset(X, y, np.arange(1096), [f"dev-{i}" for i in range(1096)])
    return ReplayService(tmp_path, demo, dataset)


def test_delayed_outcomes_and_retry_do_not_duplicate(tmp_path, bundle):
    service = make_service(tmp_path, bundle)
    stream = service.create("baseline")
    stream_id = stream["stream_id"]
    first = service.advance(stream_id, 0)
    assert first["latest_window"]["matured_label_count"] == 0
    assert first["cursor"] == 20
    assert service.advance(stream_id, 0) == first
    second = service.advance(stream_id, 20)
    assert second["latest_window"]["matured_label_count"] == 0
    third = service.advance(stream_id, 40)
    assert third["latest_window"]["matured_label_count"] == 20
    assert third["latest_window"]["pending_label_count"] == 40
    with pytest.raises(ApplicationError):
        service.advance(stream_id, 0)


def test_mature_performance_window_lags_input_window(tmp_path, bundle):
    service = make_service(tmp_path, bundle)
    stream = service.create("baseline")
    for _ in range(7):
        stream = service.advance(stream["stream_id"], stream["cursor"])
    window = stream["latest_window"]
    assert window["feature_window"]["count"] == 100
    assert window["performance_window"]["count"] == 100
    assert window["performance_window"]["last_sequence"] == 99
    assert window["cursor"] == 140
    assert window["performance_window"]["metrics"] is not None


def test_fixed_missingness_scenario_and_final_flush(tmp_path, bundle):
    service = make_service(tmp_path, bundle)
    stream = service.create("missingness")
    for _ in range(28):
        stream = service.advance(stream["stream_id"], stream["cursor"])
    assert stream["cursor"] == 558
    assert stream["status"] == "flushing"
    assert stream["latest_window"]["pending_label_count"] == 38
    stream = service.advance(stream["stream_id"], stream["cursor"])
    assert stream["status"] == "flushing"
    assert stream["latest_window"]["pending_label_count"] == 18
    stream = service.advance(stream["stream_id"], stream["cursor"])
    assert stream["status"] == "completed"
    assert stream["latest_window"]["pending_label_count"] == 0
    assert stream["latest_window"]["feature_window"]["warning"] is True
    assert stream["latest_window"]["feature_window"]["missingness_delta"] > 0.2
    assert len(service.windows(stream["stream_id"])) == 30
