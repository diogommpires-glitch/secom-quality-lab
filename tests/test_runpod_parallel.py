"""Cloud scheduling tests do not rent GPUs or read production test data."""

import importlib.util
from pathlib import Path

import pytest

from secom_quality.common import write_json

PATH = Path(__file__).parents[1] / "scripts/runpod/parallel_study.py"
SPEC = importlib.util.spec_from_file_location("runpod_parallel", PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_frozen_matrix_task_counts_and_device_isolation():
    protocol = {
        "candidate_configs": [{"config_id": str(i)} for i in range(13)],
        "seeds": [1, 2, 3, 4, 5],
    }
    tasks = module.initial_tasks(protocol)
    assert len(tasks) == len({module.task_key(t) for t in tasks}) == 39
    extra = module.finalist_tasks(
        protocol, [{"config": {"config_id": "a"}}, {"config": {"config_id": "b"}}]
    )
    assert len(extra) == len({module.task_key(t) for t in extra}) == 24
    assert {t["seed"] for t in extra} == {2, 3, 4, 5}
    one, two = module.worker_environment("0"), module.worker_environment("1")
    assert one["CUDA_VISIBLE_DEVICES"] == "0" and two["CUDA_VISIBLE_DEVICES"] == "1"
    assert one["OMP_NUM_THREADS"] == two["OPENBLAS_NUM_THREADS"] == "4"


def test_scheduler_assigns_each_task_once_and_rejects_duplicate_ownership(tmp_path):
    calls = []

    class Process:
        pid = 123

        def poll(self):
            return 0

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        return Process()

    tasks = [{"kind": "development", "config_id": "x", "seed": 1, "fold": i} for i in range(3)]
    module.dispatch(tmp_path, tasks, ["0", "1"], popen=fake_popen, pause=lambda _: None)
    assert len(calls) == 3 and {device for _, device in calls} == {"0", "1"}
    assert all(args[0] == module.sys.executable and "--worker" in args for args, _ in calls)
    with pytest.raises(ValueError, match="disjoint"):
        module.dispatch(tmp_path, tasks + tasks, ["0", "1"], popen=fake_popen)


def test_stop_marker_prevents_new_compute(tmp_path):
    (tmp_path / "vision/runtime").mkdir(parents=True)
    (tmp_path / "vision/runtime/STOP_REQUESTED").write_text("stop")
    with pytest.raises(KeyboardInterrupt):
        module.dispatch(
            tmp_path, [{"kind": "development", "config_id": "x", "seed": 1, "fold": 0}], ["0"]
        )


def test_final_worker_cannot_refit_after_access(tmp_path, monkeypatch):
    from secom_quality.vision import models, study

    monkeypatch.setattr(study, "checked_protocol", lambda _: {"protocol_hash": "synthetic"})
    monkeypatch.setattr(study, "freeze_selection", lambda _: {"config": {"config_id": "x"}})
    write_json(tmp_path / "vision/research/final-access.json", {"access_recorded": True})
    monkeypatch.setattr(models, "fit_model", lambda *a, **k: pytest.fail("Post-exposure refit"))
    with pytest.raises(ValueError, match="after test exposure"):
        module.worker(tmp_path, {"kind": "final", "config_id": "x", "seed": 1})


def test_primary_completion_is_not_a_terminal_watchdog_signal(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from secom_quality.common import read_json
    from secom_quality.vision import ad2, bundle, protocol, report, robustness, study

    for variable in (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        monkeypatch.setenv(variable, module.os.environ.get(variable, ""))
    config = {"config_id": "synthetic", "family": "supersimplenet"}
    frozen = {"protocol_hash": "synthetic", "candidate_configs": [config], "seeds": [1, 2, 3, 4, 5]}
    rows = [
        {
            "config": config,
            "completed_fits": 3,
            "valid": True,
            "mean_recall": 1,
            "mean_ap": 1,
            "mean_inference_seconds": 1,
            "order": 0,
        }
    ]
    selection = {
        "config": config,
        "seeds": frozen["seeds"],
        "reference_configs": [],
        "serving_seed": 1,
        "selection_hash": "synthetic",
    }
    monkeypatch.setattr(protocol, "checked_protocol", lambda _: frozen)
    monkeypatch.setattr(study, "comparison", lambda _: rows)
    monkeypatch.setattr(study, "run_experiments", lambda *a, **k: None)
    monkeypatch.setattr(study, "freeze_selection", lambda _: selection)
    monkeypatch.setattr(study, "evaluate_final", lambda *a, **k: None)
    monkeypatch.setattr(robustness, "run_stress", lambda *a, **k: None)
    monkeypatch.setattr(report, "build_report", lambda _: None)
    monkeypatch.setattr(bundle, "build_bundle", lambda _: {"bundle_id": "synthetic"})
    monkeypatch.setattr(module, "dispatch", lambda *a, **k: None)
    for name in ("audit_ad2", "freeze_ad2", "run_ad2", "export_ad2"):
        monkeypatch.setattr(ad2, name, lambda *a, **k: None)
    snapshots = []
    original = module.write_json

    def record(path, value):
        original(path, value)
        if path.name == "status.json":
            snapshots.append(dict(value))

    monkeypatch.setattr(module, "write_json", record)
    old_threads = torch.get_num_threads()
    try:
        module.run(tmp_path, ["0", "1"])
    finally:
        torch.set_num_threads(old_threads)
    primary = next(state for state in snapshots if state["phase"] == "primary_study")
    assert primary["state"] == "running" and primary["primary_status"] == "completed"
    terminal = [state for state in snapshots if state["state"] == "completed"]
    assert len(terminal) == 1 and terminal[0]["phase"] == "research"
    assert read_json(tmp_path / "vision/runtime/runpod/status.json")["state"] == "completed"
    assert (tmp_path / "vision/runtime/runpod/job.pid").read_text().strip().isdigit()
