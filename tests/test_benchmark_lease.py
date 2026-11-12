"""Safety gates for exclusive benchmark ownership and primary-study recovery."""

import errno
import hashlib
import importlib.util
import itertools
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/runpod/benchmark_lease.py"
SPEC = importlib.util.spec_from_file_location("benchmark_lease", SCRIPT)
lease = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lease)


def fake_process(
    proc, pid, *, state="S", parent=1, start=None, command=None, group=None, session=None, tag=None
):
    folder = proc / str(pid)
    folder.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[:4] = [state, str(parent), str(group or pid), str(session or pid)]
    fields[19] = str(start or pid * 100)
    (folder / "stat").write_text(f"{pid} (python3) " + " ".join(fields))
    (folder / "cmdline").write_bytes(
        b"\0".join(x.encode() for x in (command or ["python3", "other.py"]))
    )
    (folder / "environ").write_bytes((lease.TAG + "=" + tag).encode() if tag else b"")
    return lease.process(pid)


def set_state(proc, pid, state):
    text = (proc / str(pid) / "stat").read_text()
    head, tail = text.rsplit(") ", 1)
    fields = tail.split()
    fields[0] = state
    (proc / str(pid) / "stat").write_text(head + ") " + " ".join(fields))


@pytest.fixture
def world(tmp_path, monkeypatch):
    for name, number in [("SIGSTOP", 19), ("SIGCONT", 18), ("SIGKILL", 9)]:
        monkeypatch.setattr(signal, name, getattr(signal, name, number), raising=False)
    root, proc = tmp_path / "study", tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(lease, "PROC", proc)
    monkeypatch.setattr(lease.time, "time", lambda: 1000.0)
    r = lease.runtime(root)
    r.mkdir(parents=True)
    guard = {
        "state": "monitoring",
        "updated_unix": 1000.0,
        "started_unix": 0.0,
        "hourly_usd": 1.19,
        "budget_usd": 40.0,
        "max_hours": 32.0,
        "pod_id": "ourpod",
    }
    lease.atomic_json(r / "watchdog.json", guard)
    (r / "watchdog.pid").write_text("60")
    (r / "job.pid").write_text("20")
    lease.atomic_json(r / "status.json", {"state": "running", "phase": "initial_matrix"})
    fake_process(
        proc,
        60,
        command=["python3", "/project/scripts/runpod/watchdog.py", "--workspace", str(root)],
    )
    coordinator = fake_process(
        proc,
        20,
        command=["python3", "/project/scripts/runpod/parallel_study.py", "--workspace", str(root)],
    )
    owner = fake_process(
        proc, 10, command=["python3", "/project/scripts/runpod/benchmark_lease.py"]
    )
    fixed = lease.guard_check(root)
    record = {
        "lease_id": "unique-lease",
        "state": "waiting_for_primary",
        "owner": owner,
        "coordinator": coordinator,
        "guard": fixed,
        "pause_intended": True,
    }
    return SimpleNamespace(root=root, proc=proc, runtime=r, record=record, guard=guard)


def fake_signals(world, monkeypatch, *, stop_error=False, stop_request=False):
    calls = []

    def send(identity, number):
        calls.append((identity["pid"], number))
        if number == signal.SIGSTOP:
            set_state(world.proc, identity["pid"], "T")
            if stop_request:
                (world.runtime.parent / "STOP_REQUESTED").write_text("user_stop")
            if stop_error:
                raise OSError("Injected failure after pause delivery")
        elif number == signal.SIGCONT:
            set_state(world.proc, identity["pid"], "S")
        elif number == signal.SIGKILL:
            set_state(world.proc, identity["pid"], "Z")
        return True

    monkeypatch.setattr(lease, "signal_identity", send)
    return calls


def setup_run(world, monkeypatch):
    monkeypatch.setattr(lease.os, "getpid", lambda: 10)
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        assert "--recover-watch" in command, "No GPU pilot is allowed in these interruption tests"
        record = lease.read_json(world.runtime / "gpu-benchmark-lease.json")
        fake_process(world.proc, 50, command=command)
        lease.atomic_json(
            world.runtime / f"lease-recovery-{record['lease_id']}.ready",
            {"lease_id": record["lease_id"], "owner": record["owner"]},
        )
        return SimpleNamespace(pid=50, poll=lambda: None)

    monkeypatch.setattr(lease.subprocess, "Popen", popen)
    models = world.root.parent / "future.py"
    models.write_text("# isolated candidate")
    output = world.root / "vision/pilots/fresh"
    return models, output, calls


def add_worker(world, *, state="S"):
    task = {"kind": "development", "config_id": "frozen", "seed": 123, "fold": 0}
    request = world.runtime / "task-test.json"
    lease.atomic_json(request, task)
    worker = fake_process(
        world.proc,
        30,
        parent=20,
        state=state,
        command=["python3", "/project/scripts/runpod/parallel_study.py", "--worker", str(request)],
    )
    return worker, task


def test_reused_pid_is_never_signalled_or_resumed(world, monkeypatch):
    expected = dict(world.record["coordinator"])
    fake_process(world.proc, 20, state="T", start=999999, command=expected["command"])
    monkeypatch.setattr(lease.os, "kill", lambda *args: pytest.fail("Reused PID was signalled"))
    assert not lease.signal_identity(expected, signal.SIGCONT)
    assert not lease.restore_primary(world.root, world.record)


@pytest.mark.parametrize(
    "change",
    [
        {"updated_unix": 900},
        {"max_hours": 33},
        {"budget_usd": 41},
        {"started_unix": -200000},
        {"hourly_usd": float("nan")},
    ],
)
def test_invalid_stale_or_expired_guard_blocks_work(world, change):
    guard = {**world.guard, **change}
    (world.runtime / "watchdog.json").write_text(json.dumps(guard))
    with pytest.raises(lease.FinancialStop):
        lease.guard_check(world.root)


def test_missing_guard_process_is_not_treated_as_healthy(world):
    (world.runtime / "watchdog.pid").write_text("61")
    with pytest.raises(lease.FinancialStop):
        lease.guard_check(world.root)


def test_changed_original_guard_limits_are_rejected(world):
    lease.atomic_json(world.runtime / "watchdog.json", {**world.guard, "max_hours": 31})
    with pytest.raises(lease.FinancialStop, match="configuration changed"):
        lease.guard_check(world.root, world.record["guard"])


def test_worker_observation_timeout_preserves_worker_and_resumes_coordinator(world, monkeypatch):
    add_worker(world)
    models, output, popens = setup_run(world, monkeypatch)
    calls = fake_signals(world, monkeypatch)
    record = lease.run_lease(world.root, models, output, wait_seconds=0, max_seconds=1)
    assert record["state"] == "deferred" and record["primary_resumed"]
    assert calls == [(20, signal.SIGSTOP), (20, signal.SIGCONT)]
    assert lease.process(30)["state"] == "S" and len(popens) == 1
    assert popens[0][1]["close_fds"] is True and popens[0][1]["start_new_session"] is True
    assert (world.runtime / "job.pid").read_text() == "20"


def test_error_after_pause_delivery_still_resumes_primary(world, monkeypatch):
    models, output, _ = setup_run(world, monkeypatch)
    calls = fake_signals(world, monkeypatch, stop_error=True)
    record = lease.run_lease(world.root, models, output, wait_seconds=0, max_seconds=1)
    assert record["state"] == "failed" and record["primary_resumed"]
    assert calls == [(20, signal.SIGSTOP), (20, signal.SIGCONT)]


def test_stop_request_during_wait_leaves_primary_paused(world, monkeypatch):
    add_worker(world)
    models, output, _ = setup_run(world, monkeypatch)
    calls = fake_signals(world, monkeypatch, stop_request=True)
    record = lease.run_lease(world.root, models, output, wait_seconds=0, max_seconds=1)
    assert record["state"] == "attention" and not record["primary_resumed"]
    assert calls == [(20, signal.SIGSTOP)]
    assert lease.process(20)["state"] == "T" and lease.process(30)["state"] == "S"


def test_existing_user_pause_is_not_adopted_by_lease(world, monkeypatch):
    set_state(world.proc, 20, "T")
    models, output, popens = setup_run(world, monkeypatch)
    calls = fake_signals(world, monkeypatch)
    with pytest.raises(lease.LeaseError, match="independently running"):
        lease.run_lease(world.root, models, output, max_seconds=1)
    assert calls == [] and popens == []


def test_external_job_owner_is_never_overwritten(world, monkeypatch):
    set_state(world.proc, 20, "T")
    (world.runtime / "job.pid").write_text("21")
    calls = fake_signals(world, monkeypatch)
    assert not lease.restore_primary(world.root, world.record)
    assert calls == [] and (world.runtime / "job.pid").read_text() == "21"


def test_recovery_after_owner_death_resumes_durable_pause_intent(world, monkeypatch):
    set_state(world.proc, 10, "Z")
    set_state(world.proc, 20, "T")
    lease.save(world.root, world.record)
    calls = fake_signals(world, monkeypatch)
    lease.recover_watch(world.root, world.record["lease_id"])
    recovered = lease.read_json(world.runtime / "gpu-benchmark-lease.json")
    assert recovered["state"] == "recovered" and recovered["primary_resumed"]
    assert calls == [(20, signal.SIGCONT)]


def test_recovery_never_acts_on_replacement_lease(world, monkeypatch):
    lease.save(world.root, world.record, lease_id="replacement")
    monkeypatch.setattr(
        lease, "signal_identity", lambda *args: pytest.fail("Replacement lease touched")
    )
    lease.recover_watch(world.root, "old-lease")


def test_orphan_workers_are_owned_by_uuid_not_reused_group_id(world, monkeypatch):
    pilot = fake_process(world.proc, 40, state="Z", command=["python3", "future_speed_pilot.py"])
    world.record["pilot"] = pilot
    fake_process(world.proc, 41, group=40, session=40, tag=world.record["lease_id"])
    fake_process(world.proc, 42, group=40, session=40, tag="some-other-owner")
    calls = fake_signals(world, monkeypatch)
    lease.terminate_pilot(world.record, grace=0)
    assert calls == [(41, signal.SIGINT), (41, signal.SIGKILL)]
    assert lease.process(42)["state"] == "S"


def test_spawn_to_journal_crash_still_discovers_tagged_pilot(world, monkeypatch):
    fake_process(
        world.proc, 40, command=["python3", "future_speed_pilot.py"], tag=world.record["lease_id"]
    )
    calls = fake_signals(world, monkeypatch)
    lease.terminate_pilot(world.record, grace=0)
    assert calls == [(40, signal.SIGINT), (40, signal.SIGKILL)]


def test_completed_result_requires_matching_protocol_and_intact_files(world):
    worker, _task = add_worker(world)
    folder = world.root / "vision/research/experiments/frozen/123/fold-0"
    folder.mkdir(parents=True)
    (folder / "weights.safetensors").write_bytes(b"fixed-weights")
    config = {"config_id": "frozen", "family": "synthetic"}
    lease.atomic_json(
        world.root / "vision/research/protocol.json",
        {"protocol_hash": "fixed", "candidate_configs": [config]},
    )
    result = {
        "config": config,
        "protocol_hash": "fixed",
        "status": "completed",
        "seed": 123,
        "fold": 0,
        "files": {"weights.safetensors": hashlib.sha256(b"fixed-weights").hexdigest()},
    }
    lease.atomic_json(folder / "result.json", result)
    lease.completed_worker(world.root, worker)
    (folder / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(lease.Deferred, match="checksum"):
        lease.completed_worker(world.root, worker)


def test_exclusive_lock_rejects_a_second_owner(world):
    with (
        lease.lease_lock(world.root),
        pytest.raises(lease.LeaseError),
        lease.lease_lock(world.root),
    ):
        pytest.fail("Duplicate lock acquired")


@pytest.mark.parametrize("returncode,state", [(0, "completed"), (7, "failed")])
def test_owned_pilot_completion_or_failure_always_restores_primary(
    world, monkeypatch, returncode, state
):
    models, output, _ = setup_run(world, monkeypatch)
    watcher_popen = lease.subprocess.Popen
    calls = fake_signals(world, monkeypatch)

    def popen(command, **kwargs):
        if "--recover-watch" in command:
            return watcher_popen(command, **kwargs)
        current = lease.read_json(world.runtime / "gpu-benchmark-lease.json")
        fake_process(world.proc, 40, state="Z", command=command, tag=current["lease_id"])
        if returncode == 0:
            output.mkdir(parents=True)
            lease.atomic_json(output / "progress.json", {"state": "completed"})
        assert kwargs["env"][lease.TAG] == current["lease_id"]
        assert kwargs["start_new_session"] and kwargs["close_fds"]
        return SimpleNamespace(
            pid=40, poll=lambda: returncode, returncode=returncode, wait=lambda timeout: returncode
        )

    monkeypatch.setattr(lease.subprocess, "Popen", popen)
    record = lease.run_lease(world.root, models, output, wait_seconds=0, max_seconds=1)
    assert record["state"] == state and record["primary_resumed"]
    assert calls == [(20, signal.SIGSTOP), (20, signal.SIGCONT)]
    assert (world.runtime / "job.pid").read_text() == "20"


def test_unverified_pilot_termination_stops_work_instead_of_resuming(world, monkeypatch):
    add_worker(world)
    models, output, _ = setup_run(world, monkeypatch)
    calls = fake_signals(world, monkeypatch)

    def fail_cleanup(*args):
        raise lease.LeaseError("Owned pilot termination could not be verified")

    monkeypatch.setattr(lease, "terminate_pilot", fail_cleanup)
    record = lease.run_lease(world.root, models, output, wait_seconds=0, max_seconds=1)
    assert record["state"] == "attention" and not record["primary_resumed"]
    assert calls == [(20, signal.SIGSTOP)]
    assert (world.runtime.parent / "STOP_REQUESTED").exists()


def test_single_transient_process_observation_is_not_authoritative_death(world, monkeypatch):
    def unavailable(pid):
        raise PermissionError("temporary process observation failure")

    monkeypatch.setattr(lease, "process", unavailable)
    with pytest.raises(PermissionError):
        lease.alive(world.record["owner"])


def test_unconfirmed_continue_is_not_reported_as_success(world, monkeypatch):
    add_worker(world)
    models, output, _ = setup_run(world, monkeypatch)
    calls = []

    def send(identity, number):
        calls.append((identity["pid"], number))
        if number == signal.SIGSTOP:
            set_state(world.proc, identity["pid"], "T")
        return True

    monkeypatch.setattr(lease, "signal_identity", send)
    monkeypatch.setattr(lease.time, "sleep", lambda seconds: None)
    record = lease.run_lease(world.root, models, output, wait_seconds=0, max_seconds=1)
    assert record["state"] == "attention" and not record["primary_resumed"]
    assert (world.runtime.parent / "STOP_REQUESTED").exists()


def test_dead_guard_uses_pod_scoped_stop_fallback(world, monkeypatch):
    set_state(world.proc, 60, "Z")
    stops = []
    module = SimpleNamespace(stop_compute=lambda pod_id: stops.append(pod_id))
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda target: None))
    monkeypatch.setattr(lease.importlib.util, "spec_from_file_location", lambda *args: spec)
    monkeypatch.setattr(lease.importlib.util, "module_from_spec", lambda selected: module)
    lease.financial_stop(world.root, world.record, "Guard process is dead")
    assert stops == ["ourpod"]
    assert lease.read_json(world.runtime / "status.json")["state"] == "failed"


def test_primary_terminal_state_blocks_benchmark_start(world):
    lease.atomic_json(world.runtime / "status.json", {"state": "completed"})
    with pytest.raises(lease.FinancialStop):
        lease.guard_check(world.root)


def test_other_workspace_coordinator_is_not_paused(world, monkeypatch):
    fake_process(
        world.proc,
        20,
        command=[
            "python3",
            "/project/scripts/runpod/parallel_study.py",
            "--workspace",
            "/some-other-study",
        ],
    )
    models, output, popens = setup_run(world, monkeypatch)
    calls = fake_signals(world, monkeypatch)
    with pytest.raises(lease.LeaseError):
        lease.run_lease(world.root, models, output, max_seconds=1)
    assert calls == [] and popens == []


def test_near_financial_deadline_never_resumes_primary(world, monkeypatch):
    set_state(world.proc, 20, "T")
    calls = fake_signals(world, monkeypatch)
    deadline = world.guard["started_unix"] + world.guard["max_hours"] * 3600
    monkeypatch.setattr(lease.time, "time", lambda: deadline - 10)
    lease.atomic_json(
        world.runtime / "watchdog.json", {**world.guard, "updated_unix": deadline - 10}
    )
    with pytest.raises(lease.FinancialStop):
        lease.restore_primary(world.root, world.record)
    assert calls == []


def test_unsupported_pidfd_falls_back_after_identity_recheck(world, monkeypatch):
    calls = []

    def unavailable(pid):
        raise OSError(errno.ENOSYS, "pidfd unavailable in this container")

    monkeypatch.setattr(lease.os, "pidfd_open", unavailable, raising=False)
    monkeypatch.setattr(lease.signal, "pidfd_send_signal", lambda *args: None, raising=False)
    monkeypatch.setattr(lease.os, "kill", lambda pid, number: calls.append((pid, number)))
    assert lease.signal_identity(world.record["coordinator"], signal.SIGCONT)
    assert calls == [(20, signal.SIGCONT)]


def test_process_exit_during_signal_is_terminal_not_cleanup_failure(world, monkeypatch):
    def exited(pid):
        raise ProcessLookupError("Owned process just exited")

    monkeypatch.setattr(lease.os, "pidfd_open", exited, raising=False)
    monkeypatch.setattr(lease.signal, "pidfd_send_signal", lambda *args: None, raising=False)
    monkeypatch.setattr(lease.os, "kill", lambda *args: pytest.fail("Gone PID was signalled"))
    assert not lease.signal_identity(world.record["coordinator"], signal.SIGCONT)


def test_child_fork_during_pending_stop_is_observed_before_gpu_handoff(world, monkeypatch):
    observations = 0
    real_process = lease.process

    def pending_stop(pid):
        nonlocal observations
        if pid == 20:
            observations += 1
            if observations == 2:
                add_worker(world)
                set_state(world.proc, 20, "T")
        return real_process(pid)

    monkeypatch.setattr(lease, "process", pending_stop)
    monkeypatch.setattr(lease.time, "sleep", lambda seconds: None)
    tick = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(lease.time, "monotonic", lambda: next(tick))
    with pytest.raises(lease.Deferred, match="still active"):
        lease.wait_primary(world.root, world.record, seconds=1)
    assert real_process(30)["state"] == "S"
    assert observations >= 2


def test_restore_cancels_owned_pending_stop_even_before_state_becomes_T(world, monkeypatch):
    calls = fake_signals(world, monkeypatch)
    assert lease.process(20)["state"] == "S"
    assert lease.restore_primary(world.root, world.record)
    assert calls == [(20, signal.SIGCONT)]


def test_unrelated_environment_permission_denial_does_not_abort_cleanup(world, monkeypatch):
    fake_process(world.proc, 80, command=["unrelated-service"])
    original = Path.read_bytes

    def guarded(path):
        if path == world.proc / "80/environ":
            raise PermissionError("Unrelated service is not dumpable")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    assert lease.pilot_members(world.record) == []


def test_known_benchmark_environment_denial_stays_unknown(world, monkeypatch):
    fake_process(world.proc, 40, command=["python3", "future_speed_pilot.py"])
    original = Path.read_bytes

    def guarded(path):
        if path == world.proc / "40/environ":
            raise PermissionError("Possible owned pilot cannot be observed")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(lease.LeaseError, match="Cannot verify"):
        lease.pilot_members(world.record)


def test_symbolic_fit_member_is_not_accepted_after_resolution(world):
    worker, _task = add_worker(world)
    folder = world.root / "vision/research/experiments/frozen/123/fold-0"
    folder.mkdir(parents=True)
    target = folder / "target.safetensors"
    target.write_bytes(b"fixed")
    link = folder / "weights.safetensors"
    link.symlink_to(target)
    config = {"config_id": "frozen"}
    lease.atomic_json(
        world.root / "vision/research/protocol.json",
        {"protocol_hash": "fixed", "candidate_configs": [config]},
    )
    digest = hashlib.sha256(b"fixed").hexdigest()
    result = {
        "config": config,
        "protocol_hash": "fixed",
        "status": "completed",
        "seed": 123,
        "fold": 0,
        "files": {"target.safetensors": digest, "weights.safetensors": digest},
    }
    lease.atomic_json(folder / "result.json", result)
    with pytest.raises(lease.Deferred, match="Unsafe"):
        lease.completed_worker(world.root, worker)


def stub_pod_stop(monkeypatch):
    stops = []
    module = SimpleNamespace(stop_compute=lambda pod_id: stops.append(pod_id))
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda target: None))
    monkeypatch.setattr(lease.importlib.util, "spec_from_file_location", lambda *args: spec)
    monkeypatch.setattr(lease.importlib.util, "module_from_spec", lambda selected: module)
    return stops


@pytest.mark.parametrize("foreign", ["workspace", "pod"])
def test_foreign_watchdog_never_suppresses_current_pod_stop(world, monkeypatch, foreign):
    stops = stub_pod_stop(monkeypatch)
    if foreign == "workspace":
        fake_process(
            world.proc,
            60,
            command=[
                "python3",
                "/project/scripts/runpod/watchdog.py",
                "--workspace",
                "/other-study",
            ],
        )
    else:
        lease.atomic_json(
            world.runtime / "watchdog.json", {**world.guard, "pod_id": "different-pod"}
        )
    lease.financial_stop(world.root, world.record, "Guard belongs to a different workload")
    assert stops == ["ourpod"]


@pytest.mark.parametrize("content", [None, "{bad-json", "[]"])
def test_invalid_primary_status_publishes_failed_control_record(world, monkeypatch, content):
    path = world.runtime / "status.json"
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    with pytest.raises(lease.FinancialStop):
        lease.guard_check(world.root)
    stops = stub_pod_stop(monkeypatch)
    lease.financial_stop(world.root, world.record, "Primary status cannot be verified")
    assert lease.read_json(path)["state"] == "failed"
    assert (world.runtime.parent / "STOP_REQUESTED").exists()
    assert stops == []  # The correct live watchdog can now observe a valid failure.
    if content is not None:
        assert any(p.read_text() == content for p in world.runtime.glob("status.invalid-*.json"))


def test_unreadable_job_marker_in_recovery_requests_stop_and_saves_attention(world, monkeypatch):
    set_state(world.proc, 10, "Z")
    set_state(world.proc, 20, "T")
    (world.runtime / "job.pid").unlink()
    calls = fake_signals(world, monkeypatch)
    record = lease.recover_owner(world.root, world.record)
    assert record["state"] == "attention" and not record["primary_resumed"]
    assert calls == []
    assert (world.runtime.parent / "STOP_REQUESTED").exists()
    assert lease.read_json(world.runtime / "status.json")["state"] == "failed"


def test_insufficient_benchmark_reserve_defers_without_stopping_healthy_primary(world, monkeypatch):
    models, output, popens = setup_run(world, monkeypatch)
    calls = fake_signals(world, monkeypatch)
    deadline = world.guard["started_unix"] + world.guard["max_hours"] * 3600
    monkeypatch.setattr(lease.time, "time", lambda: deadline - 100)
    lease.atomic_json(
        world.runtime / "watchdog.json", {**world.guard, "updated_unix": deadline - 100}
    )
    with pytest.raises(lease.Deferred, match="insufficient time"):
        lease.run_lease(world.root, models, output, wait_seconds=900, max_seconds=3600)
    assert calls == [] and popens == []
    assert not (world.runtime.parent / "STOP_REQUESTED").exists()
    assert lease.read_json(world.runtime / "status.json")["state"] == "running"


def test_independent_deadline_recovers_an_alive_paused_owner(world, monkeypatch):
    set_state(world.proc, 10, "T")
    set_state(world.proc, 20, "T")
    add_worker(world)
    lease.save(world.root, world.record, state="waiting_for_primary", wait_deadline_unix=999.0)
    calls = fake_signals(world, monkeypatch)
    times = itertools.count(0.0, 21.0)
    monkeypatch.setattr(lease.time, "monotonic", lambda: next(times))
    lease.recover_watch(world.root, world.record["lease_id"])
    recovered = lease.read_json(world.runtime / "gpu-benchmark-lease.json")
    assert recovered["state"] == "recovered" and recovered["primary_resumed"]
    assert calls == [(10, signal.SIGTERM), (10, signal.SIGKILL), (20, signal.SIGCONT)]
    assert lease.process(30)["state"] == "S"  # Existing research worker is never interrupted.


def test_expired_wait_deadline_is_inactive_during_a_live_pilot(world, monkeypatch):
    record = {
        **world.record,
        "state": "pilot_running",
        "wait_deadline_unix": 900.0,
        "pilot_deadline_unix": 1100.0,
    }
    lease.save(world.root, record)
    monkeypatch.setattr(
        lease,
        "signal_identity",
        lambda *args: pytest.fail("Pilot interrupted by old wait deadline"),
    )
    assert lease.explicit_deadline(record) == 1100.0
    assert not lease.expire_owner(world.root, record["lease_id"], record["owner"])


@pytest.mark.parametrize("content", [None, "{bad-json", "[]"])
def test_bad_initial_watchdog_uses_only_provider_current_pod_identity(world, monkeypatch, content):
    models, output, popens = setup_run(world, monkeypatch)
    calls = fake_signals(world, monkeypatch)
    stops = stub_pod_stop(monkeypatch)
    monkeypatch.setenv("RUNPOD_POD_ID", "trusted-current-pod")
    path = world.runtime / "watchdog.json"
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    with pytest.raises(lease.FinancialStop):
        lease.run_lease(world.root, models, output, max_seconds=1)
    assert stops == ["trusted-current-pod"]
    assert calls == [] and popens == []
    assert lease.read_json(world.runtime / "status.json")["state"] == "failed"
    assert (world.runtime / "job.pid").read_text() == "20"
    emergency = lease.initial_stop_guard(world.root)
    assert set(emergency) == {"pod_id", "emergency_identity_only"}


@pytest.mark.parametrize("provider_identity", [None, "untrusted pod; stop-other"])
def test_bad_guard_without_trusted_provider_identity_never_guesses(
    world, monkeypatch, provider_identity
):
    models, output, popens = setup_run(world, monkeypatch)
    stops = stub_pod_stop(monkeypatch)
    if provider_identity is None:
        monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    else:
        monkeypatch.setenv("RUNPOD_POD_ID", provider_identity)
    (world.runtime / "watchdog.json").write_text("[]")
    with pytest.raises(lease.LeaseError, match="No trusted current Pod identity"):
        lease.run_lease(world.root, models, output, max_seconds=1)
    assert stops == [] and popens == []
    assert lease.read_json(world.runtime / "status.json")["state"] == "failed"
    assert (world.runtime.parent / "STOP_REQUESTED").exists()


def test_nonobject_stop_guard_cannot_be_trusted_as_a_financial_record(world, monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "trusted-current-pod")
    stops = stub_pod_stop(monkeypatch)
    lease.financial_stop(world.root, {"guard": []}, "Captured guard object is invalid")
    assert stops == ["trusted-current-pod"]
    assert lease.read_json(world.runtime / "status.json")["state"] == "failed"
