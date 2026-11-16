"""Synthetic /proc and signal spies exercise ownership without touching processes."""

import json
import signal

import pytest

from scripts.runpod import process_identity, watchdog


def fake_process(proc, pid, root, script, *, start=10):
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[0], fields[19] = "S", str(start)
    (directory / "stat").write_text(f"{pid} (python) " + " ".join(fields))
    command = ["python3", f"/code/{script}", "--workspace", str(root)]
    if script == "watchdog.py":
        command += [
            "--pod-id",
            "current-pod",
            "--started-unix",
            "0",
            "--hourly-usd",
            "1.19",
            "--budget-usd",
            "40",
            "--max-hours",
            "32",
        ]
    (directory / "cmdline").write_bytes(b"\0".join(part.encode() for part in command))
    return process_identity.process(pid, proc)


@pytest.fixture
def guard(tmp_path):
    root, proc = tmp_path / "study", tmp_path / "proc"
    runtime = root / "vision/runtime/runpod"
    runtime.mkdir(parents=True)
    identity = fake_process(proc, 40, root, "watchdog.py")
    record = {
        "identity": identity,
        "pod_id": "current-pod",
        "state": "monitoring",
        "updated_unix": 100,
        "started_unix": 0,
        "hourly_usd": 1.19,
        "budget_usd": 40,
        "max_hours": 32,
    }
    (runtime / "watchdog.pid").write_text("40")
    (runtime / "watchdog.json").write_text(json.dumps(record))
    return root, proc, runtime, record


@pytest.mark.parametrize(
    "change",
    [
        "pid_reuse",
        "workspace",
        "pod",
        "limits",
        "stale",
        "missing_identity",
        "command_pod",
        "command_limits",
    ],
)
def test_launch_guard_requires_exact_fresh_identity_and_original_limits(guard, change):
    root, proc, runtime, record = guard
    assert watchdog.verify_guard(root, "current-pod", 0, 1.19, 40, 32, proc=proc, now=100)
    if change == "pid_reuse":
        fake_process(proc, 40, root, "watchdog.py", start=99)
    elif change == "workspace":
        record["identity"] = fake_process(proc, 40, root.parent / "other", "watchdog.py")
    elif change == "pod":
        record["pod_id"] = "old-pod"
    elif change == "limits":
        record["started_unix"] = 90
    elif change == "stale":
        record["updated_unix"] = 0
    elif change in {"command_pod", "command_limits"}:
        command = record["identity"]["command"]
        flag = "--pod-id" if change == "command_pod" else "--started-unix"
        command[command.index(flag) + 1] = "old-pod" if change == "command_pod" else "90"
        (proc / "40/cmdline").write_bytes(b"\0".join(part.encode() for part in command))
    else:
        del record["identity"]
    (runtime / "watchdog.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="creation-bound guard"):
        watchdog.verify_guard(root, "current-pod", 0, 1.19, 40, 32, proc=proc, now=100)


def test_watchdog_signals_only_bound_coordinator_via_pidfd(guard, monkeypatch):
    root, proc, runtime, _ = guard
    calls = []
    monkeypatch.setattr(
        watchdog.os, "pidfd_open", lambda pid: calls.append(("open", pid)) or 800, raising=False
    )
    monkeypatch.setattr(
        signal,
        "pidfd_send_signal",
        lambda fd, number: calls.append(("signal", fd, number)),
        raising=False,
    )
    monkeypatch.setattr(watchdog.os, "close", lambda fd: calls.append(("close", fd)))
    (runtime / "job.pid").write_text("42")
    assert not watchdog.signal_coordinator(root, proc)
    assert calls == []
    identity = fake_process(proc, 42, root, "parallel_study.py")
    (runtime / "job-identity.json").write_text(json.dumps(identity))
    assert watchdog.signal_coordinator(root, proc)
    assert calls == [("open", 42), ("signal", 800, signal.SIGTERM), ("close", 800)]
    calls.clear()
    fake_process(proc, 42, root, "parallel_study.py", start=99)
    assert not watchdog.signal_coordinator(root, proc) and calls == []


def test_coordinator_identity_is_rechecked_after_pidfd_open(guard, monkeypatch):
    root, proc, runtime, _ = guard
    identity = fake_process(proc, 42, root, "parallel_study.py")
    (runtime / "job.pid").write_text("42")
    (runtime / "job-identity.json").write_text(json.dumps(identity))
    calls = []

    def open_pidfd(pid):
        fake_process(proc, pid, root, "parallel_study.py", start=99)
        return 800

    monkeypatch.setattr(watchdog.os, "pidfd_open", open_pidfd, raising=False)
    monkeypatch.setattr(
        signal,
        "pidfd_send_signal",
        lambda *args: pytest.fail("Reused PID signalled"),
        raising=False,
    )
    monkeypatch.setattr(watchdog.os, "close", lambda fd: calls.append(fd))
    assert not watchdog.signal_coordinator(root, proc)
    assert calls == [800]


def test_unknown_local_job_never_prevents_current_provider_stop(guard, monkeypatch):
    root, proc, runtime, record = guard
    monkeypatch.setenv("RUNPOD_POD_ID", "current-pod")
    monkeypatch.setattr(watchdog, "process", lambda pid: record["identity"])
    monkeypatch.setattr(watchdog.time, "sleep", lambda _: None)
    monkeypatch.setattr(watchdog.time, "time", lambda: 1000)
    original = watchdog.signal_coordinator
    monkeypatch.setattr(watchdog, "signal_coordinator", lambda workspace: original(workspace, proc))
    calls = []
    monkeypatch.setattr(watchdog, "stop_compute", lambda pod_id: calls.append(pod_id))
    (runtime / "job.pid").write_text("999999")
    watchdog.monitor(root, "current-pod", 0, 1.19, 40, 0.001)
    assert calls == ["current-pod"]
    assert json.loads((runtime / "watchdog.json").read_text())["state"] == "provider_stop_accepted"


@pytest.mark.parametrize("all_equals", [False, True])
def test_guard_accepts_bare_equals_and_mixed_argument_forms(guard, all_equals):
    root, proc, runtime, record = guard
    command = record["identity"]["command"]
    updated = command[:2]
    for index in range(2, len(command), 2):
        flag, value = command[index : index + 2]
        if all_equals or flag in {"--workspace", "--started-unix", "--max-hours"}:
            updated.append(flag + "=" + value)
        else:
            updated.extend([flag, value])
    (proc / "40/cmdline").write_bytes(b"\0".join(part.encode() for part in updated))
    record["identity"] = process_identity.process(40, proc)
    (runtime / "watchdog.json").write_text(json.dumps(record))
    assert watchdog.verify_guard(root, "current-pod", 0, 1.19, 40, 32, proc=proc, now=100)


@pytest.mark.parametrize("flag", ["--workspace", "--pod-id", "--started-unix"])
def test_guard_rejects_duplicate_named_arguments_even_with_matching_values(guard, flag):
    root, proc, runtime, record = guard
    command = record["identity"]["command"]
    command.append(flag + "=" + process_identity.option(command, flag))
    (proc / "40/cmdline").write_bytes(b"\0".join(part.encode() for part in command))
    (runtime / "watchdog.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="creation-bound guard"):
        watchdog.verify_guard(root, "current-pod", 0, 1.19, 40, 32, proc=proc, now=100)


@pytest.mark.parametrize(
    "command", [["--workspace"], ["--workspace="], ["--workspace", "--pod-id", "value"]]
)
def test_named_argument_missing_value_is_rejected(command):
    with pytest.raises(ValueError, match="Missing --workspace"):
        process_identity.option(command, "--workspace")


def test_creation_bound_coordinator_accepts_equals_workspace(guard):
    root, proc, runtime, _ = guard
    identity = fake_process(proc, 42, root, "parallel_study.py")
    command = [*identity["command"][:2], "--workspace=" + str(root)]
    (proc / "42/cmdline").write_bytes(b"\0".join(part.encode() for part in command))
    identity = process_identity.process(42, proc)
    (runtime / "job.pid").write_text("42")
    (runtime / "job-identity.json").write_text(json.dumps(identity))
    assert process_identity.coordinator(root, proc)["pid"] == 42
