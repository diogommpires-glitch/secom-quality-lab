"""Real process-counter and artifact boundaries for the read-only live sampler."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).parents[1] / "scripts/runpod/live_monitor.py"
spec = importlib.util.spec_from_file_location("live_monitor", MODULE)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_process(proc, pid, parent, command, *, start=10, cpu=0, reads=0, writes=0):
    path = proc / str(pid)
    path.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 30
    fields[0], fields[1] = "S", str(parent)
    fields[11], fields[12], fields[19] = str(cpu), "0", str(start)
    (path / "stat").write_text(f"{pid} (python with spaces) " + " ".join(fields))
    (path / "cmdline").write_bytes("\0".join(command).encode() + b"\0")
    (path / "io").write_text(f"rchar: {reads}\nwchar: {writes}\n")
    (path / "status").write_text("State:\tS (sleeping)\nVmRSS:\t1234 kB\n")
    (path / "wchan").write_text("request_wait_answer\n")


@pytest.fixture
def live(tmp_path):
    root, proc = tmp_path / "study", tmp_path / "proc"
    runtime = root / "vision/runtime/runpod"
    write_json(runtime / "status.json", {"state": "running", "phase": "initial_matrix"})
    write_json(
        runtime / "watchdog.json",
        {
            "state": "monitoring",
            "updated_unix": 100,
            "started_unix": 0,
            "hourly_usd": 1.19,
            "budget_usd": 40,
            "max_hours": 32,
        },
    )
    write_json(
        root / "vision/research/protocol.json",
        {"candidate_configs": [{"config_id": "a"}, {"config_id": "b"}], "seeds": [77, 78]},
    )
    (runtime / "job.pid").write_text("41")
    request = runtime / "task-fixture.json"
    write_json(request, {"kind": "development", "config_id": "a", "seed": 77, "fold": 0})
    write_process(proc, 41, 1, ["python", "/code/parallel_study.py", "--workspace", str(root)])
    write_process(
        proc,
        42,
        41,
        ["python", "/code/parallel_study.py", "--worker", str(request)],
        reads=500,
        cpu=100,
    )
    return root, proc, runtime


def sample(live, previous=None, now=100, gpu=None):
    root, proc, _ = live
    return monitor.snapshot(
        root, previous, now=now, proc=proc, gpu={"available": False} if gpu is None else gpu
    )


def test_cpu_and_io_deltas_prove_work_and_unchanged_counters_are_waiting(live):
    first = sample(live)
    root, proc, _ = live
    command = monitor.process(42, proc)["command"]
    write_process(proc, 42, 41, command, cpu=150, reads=800, writes=25)
    second = sample(live, first, now=105)
    worker = second["workers"][0]
    assert second["activity"] == "cpu_work"
    assert worker["cpu_seconds_delta"] == 0.5
    assert worker["cpu_percent_one_core"] == 10
    assert worker["rchar_delta"] == 300 and worker["wchar_delta"] == 25
    assert worker["rss_bytes"] == 1234 * 1024
    assert worker["wait_channel"] == "request_wait_answer"
    third = sample(live, second, now=110)
    assert third["activity"] == "waiting" and not third["terminal"]
    write_process(proc, 42, 41, command, cpu=150, reads=900, writes=25)
    assert sample(live, third, now=115)["activity"] == "io_progress"
    assert root.exists()


def test_pid_reuse_never_inherits_another_process_counter_delta(live):
    first = sample(live)
    _, proc, _ = live
    command = monitor.process(42, proc)["command"]
    write_process(proc, 42, 41, command, start=99, cpu=9000, reads=9000)
    second = sample(live, first, now=105)
    assert "cpu_seconds_delta" not in second["workers"][0]
    assert "rchar_delta" not in second["workers"][0]
    assert second["activity"] == "waiting"


def test_initial_denominator_and_all_development_counts_exclude_staging_and_other_seeds(live):
    root, _, _ = live
    experiments = root / "vision/research/experiments"
    for config, seed, fold, status in [
        ("a", 77, "fold-0", "completed"),
        ("b", 77, "fold-1", "failed"),
        ("a", 78, "fold-0", "completed"),
        ("a", 77, "fold-2.staging", "completed"),
    ]:
        write_json(
            experiments / config / str(seed) / fold / "result.json",
            {"status": status, "metrics": {"not_read_by_sampler": "private"}},
        )
    (experiments / "predictions.parquet").write_bytes(b"never decode these predictions")
    result = sample(live)
    assert result["fits"]["initial"] == {"completed": 1, "failed": 1, "expected": 6}
    assert result["fits"]["total_development"] == {"completed": 2, "failed": 1, "unknown": 0}
    assert "metrics" not in json.dumps(result) and "private" not in json.dumps(result)


def test_only_current_active_training_progress_is_reported(live):
    root, _, _ = live
    base = root / "vision/research/experiments"
    write_json(
        base / "a/77/fold-0.staging/model/progress.json",
        {"epoch": 3, "epochs": 100, "mean_loss": 0.4, "seconds": 33, "labels": [1, 0]},
    )
    write_json(base / "b/77/fold-1.staging/model/progress.json", {"epoch": 99, "epochs": 100})
    training = sample(live)["workers"][0]["training"]
    assert training == {"epoch": 3, "epochs": 100, "mean_loss": 0.4, "seconds": 33}
    assert (
        monitor.training_progress(
            root, {"kind": "development", "config_id": "../b", "seed": 77, "fold": 0}
        )
        is None
    )


def test_missing_or_failing_gpu_probe_is_unknown_and_not_a_study_failure(live):
    def failing(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 3)

    assert monitor.gpu_sample(failing) == {"available": False}
    record = sample(live)
    assert not record["gpu"]["available"] and not record["terminal"]
    result = monitor.gpu_sample(
        lambda *args, **kwargs: SimpleNamespace(stdout="91, 15000, 32607\n")
    )
    assert result["available"] and result["utilization_percent"] == 91
    assert sample(live, gpu=result)["activity"] == "gpu_work"


def test_reused_coordinator_pid_is_not_live_and_stop_requires_terminal_evidence(live):
    _, proc, runtime = live
    write_process(proc, 41, 1, ["python", "unrelated.py"])
    record = sample(live)
    assert not record["coordinator_alive"] and not record["terminal"]
    write_json(runtime / "status.json", {"state": "completed", "phase": "research"})
    assert sample(live)["terminal"]
    write_json(runtime / "status.json", {"state": "running", "phase": "initial_matrix"})
    (runtime.parent / "STOP_REQUESTED").write_text("user_stop")
    assert sample(live)["terminal"]


def test_original_guard_clock_is_only_observed_and_remaining_time_is_correct(live):
    record = sample(live, now=3600)
    assert record["remaining_guard_hours"] == 31
    assert record["guard_age_seconds"] == 3500
    assert record["guard_state"] == "monitoring"


def test_atomic_output_is_valid_json_without_temporary_files(tmp_path):
    path = tmp_path / "status/live-telemetry.json"
    monitor.atomic_json(path, {"state": "running", "epoch": 2})
    assert json.loads(path.read_text()) == {"state": "running", "epoch": 2}
    assert len(list(path.parent.iterdir())) == 1


@pytest.fixture
def benchmark_lease(live):
    root, proc, runtime = live
    output = root.parent / "future-performance"
    write_json(
        output / "progress.json",
        {
            "state": "running",
            "stage": "warm_ssn_fit",
            "profile": "future",
            "resolution": [704, 256],
            "epochs": 12,
            "fit_directory": str(output / "fit"),
            "labels": [0, 1],
        },
    )
    write_json(
        output / "fit/progress.json",
        {"epoch": 3, "epochs": 12, "mean_loss": 0.5, "seconds": 30, "scores": [0.2]},
    )
    lease = {
        "lease_id": "fixture-lease",
        "state": "pilot_running",
        "coordinator": {"pid": 41, "start_ticks": 10},
        "owner": {"pid": 91, "start_ticks": 11},
        "pilot": {"pid": 92, "start_ticks": 12},
        "output": str(output),
        "progress_path": str(output / "progress.json"),
    }
    write_json(runtime / "gpu-benchmark-lease.json", lease)
    write_process(proc, 91, 1, ["python", "/code/benchmark_lease.py"], start=11)
    write_process(
        proc, 92, 91, ["python", "/code/future_speed_pilot.py"], start=12, cpu=100, reads=1000
    )
    return lease


def test_verified_performance_lease_reports_separate_work_and_epochs(live, benchmark_lease):
    first = sample(live)
    _, proc, _ = live
    command = monitor.process(92, proc)["command"]
    write_process(proc, 92, 91, command, start=12, cpu=300, reads=2000)
    result = sample(live, first, now=105)
    performance = result["operational_benchmark"]
    assert performance["scope"] == "development_training_only_performance_not_selection"
    assert performance["pilot"]["cpu_percent_one_core"] == 40
    assert performance["training"] == {"epoch": 3, "epochs": 12, "mean_loss": 0.5, "seconds": 30}
    assert performance["progress"]["profile"] == "future"
    assert result["phase"] == "initial_matrix" and result["fits"]["initial"]["completed"] == 0
    assert result["activity"] == "cpu_work"
    assert "labels" not in json.dumps(result) and "scores" not in json.dumps(result)


def test_reused_lease_owner_or_pilot_identity_is_not_trusted(live, benchmark_lease):
    _, proc, _ = live
    command = monitor.process(91, proc)["command"]
    write_process(proc, 91, 1, command, start=999)
    result = sample(live)["operational_benchmark"]
    assert result["attention_required"] and result["reason"] == "lease_owner_not_verified"
    write_process(proc, 91, 1, command, start=11)
    write_process(proc, 92, 91, ["python", "/code/future_speed_pilot.py"], start=999)
    assert sample(live)["operational_benchmark"]["reason"] == "pilot_finished_or_identity_changed"


def test_performance_progress_must_stay_in_isolated_output(live, benchmark_lease):
    root, _, runtime = live
    benchmark_lease["output"] = str(root / "protected")
    benchmark_lease["progress_path"] = str(root / "protected/progress.json")
    write_json(runtime / "gpu-benchmark-lease.json", benchmark_lease)
    assert sample(live)["operational_benchmark"]["reason"] == "unexpected_progress_path"


def test_old_terminal_lease_does_not_appear_as_active_work(live, benchmark_lease):
    _, _, runtime = live
    benchmark_lease["state"] = "completed"
    write_json(runtime / "gpu-benchmark-lease.json", benchmark_lease)
    assert sample(live)["operational_benchmark"] is None


@pytest.mark.parametrize("payload", ["[]", "null", "42", "invalid JSON"])
def test_invalid_snapshot_shape_is_unknown_instead_of_crashing(tmp_path, payload):
    path = tmp_path / "metadata.json"
    path.write_text(payload)
    assert monitor.read_json(path) == {}


def test_derived_benchmark_resource_log_does_not_include_predictions(live, benchmark_lease):
    record = sample(live)
    monitor.record_benchmark_sample(record)
    path = Path(benchmark_lease["output"]) / "resource-samples.jsonl"
    line = json.loads(path.read_text())
    assert line["progress"]["stage"] == "warm_ssn_fit"
    assert line["training"]["epoch"] == 3
    assert "labels" not in path.read_text() and "scores" not in path.read_text()
    assert record["operational_benchmark"]["resource_log_status"] == "recorded"


def test_sampler_never_precreates_the_fresh_pilot_output(tmp_path):
    output = tmp_path / "not-created-yet"
    record = {"operational_benchmark": {"pilot": {"pid": 9}, "output": str(output)}}
    monitor.record_benchmark_sample(record)
    assert not output.exists()
