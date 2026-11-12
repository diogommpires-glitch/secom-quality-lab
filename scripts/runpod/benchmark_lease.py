"""Exclusive, recoverable GPU benchmark lease; the frozen study stays unchanged."""

import argparse
import contextlib
import errno
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

PROC = Path("/proc")
TERMINAL = {"completed", "deferred", "failed", "cancelled", "recovered"}
TAG = "SECOM_BENCHMARK_LEASE_ID"


class LeaseError(Exception):
    pass


class Deferred(LeaseError):
    pass


class FinancialStop(LeaseError):
    pass


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def runtime(root):
    return root / "vision/runtime/runpod"


def save(root, record, **changes):
    record.update(changes, updated_unix=time.time())
    atomic_json(runtime(root) / "gpu-benchmark-lease.json", record)


@contextlib.contextmanager
def lease_lock(root):
    path = runtime(root) / "gpu-benchmark-lease.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        os.set_inheritable(stream.fileno(), False)
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            def lock():
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

            def unlock():
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def lock():
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def unlock():
                fcntl.flock(stream, fcntl.LOCK_UN)

        try:
            lock()
        except OSError as exc:
            raise LeaseError("Another lease/recovery owns the benchmark lock") from exc
        try:
            yield
        finally:
            stream.seek(0)
            unlock()


def process(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise LeaseError("Unsafe process identity")
    path = PROC / str(pid)
    try:
        fields = (path / "stat").read_text().rpartition(") ")[2].split()
        command = [
            part.decode(errors="replace")
            for part in (path / "cmdline").read_bytes().split(b"\0")
            if part
        ]
    except FileNotFoundError:
        return None
    return {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "state": fields[0],
        "parent": int(fields[1]),
        "group": int(fields[2]),
        "session": int(fields[3]),
        "command": command,
    }


def matches(identity, current):
    return bool(
        current
        and current["pid"] == identity["pid"]
        and current["start_ticks"] == identity["start_ticks"]
        and (current["state"] == "Z" or current["command"] == identity["command"])
    )


def alive(identity):
    current = process(identity["pid"])
    return matches(identity, current) and current["state"] != "Z"


def command_is(current, name):
    return bool(current and any(Path(part).name == name for part in current["command"]))


def signal_identity(identity, number):
    current = process(identity["pid"])
    if not matches(identity, current) or current["state"] == "Z":
        return False
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            try:
                fd = os.pidfd_open(identity["pid"])
                try:
                    if not matches(identity, process(identity["pid"])):
                        return False
                    signal.pidfd_send_signal(fd, number)
                    return True
                finally:
                    os.close(fd)
            except OSError as exc:
                if exc.errno not in {errno.ENOSYS, errno.EINVAL, errno.EPERM}:
                    raise
        if not matches(identity, process(identity["pid"])):
            return False
        os.kill(identity["pid"], number)
        return True
    except ProcessLookupError:
        return False


def children(identity):
    result = []
    for entry in PROC.iterdir():
        if entry.name.isdigit() and int(entry.name) > 1:
            current = process(int(entry.name))
            if current and current["parent"] == identity["pid"]:
                result.append(current)
    return result


def workspace_matches(current, root):
    try:
        args = current["command"]
        return Path(args[args.index("--workspace") + 1]).resolve() == root.resolve()
    except (ValueError, IndexError, TypeError):
        return False


def guard_check(root, expected=None, reserve=0, *, for_resume=False):
    r = runtime(root)
    if (r.parent / "STOP_REQUESTED").exists():
        raise FinancialStop("An existing stop request prevents work")
    try:
        guard = read_json(r / "watchdog.json")
        status = read_json(r / "status.json")
        if not isinstance(guard, dict) or not isinstance(status, dict):
            raise ValueError("Guard and primary status must be JSON objects")
        if status.get("state") in {
            "failed",
            "paused",
            "interrupted",
            "completed",
        }:
            raise ValueError("Primary study has a stop or terminal state")
        fixed = {
            key: guard[key]
            for key in ("started_unix", "hourly_usd", "budget_usd", "max_hours", "pod_id")
        }
        values = [
            float(fixed[key]) for key in ("started_unix", "hourly_usd", "budget_usd", "max_hours")
        ]
        started, rate, budget, hours = values
        age = time.time() - float(guard["updated_unix"])
        gp = process(int((r / "watchdog.pid").read_text()))
        if not all(math.isfinite(value) for value in [*values, age]):
            raise ValueError("Nonfinite guard values")
        if not (
            started <= time.time() + 60 and 0 < rate <= 2.5 and 0 < budget <= 40 and 0 < hours <= 32
        ):
            raise ValueError("Guard limits exceed the existing authorization")
        if (
            guard["state"] != "monitoring"
            or not 0 <= age <= 45
            or not command_is(gp, "watchdog.py")
            or not workspace_matches(gp, root)
            or gp["state"] in {"Z", "T", "t"}
        ):
            raise ValueError("Guard is not freshly monitoring")
        if expected is not None and fixed != expected:
            raise ValueError("Original guard configuration changed")
        deadline = started + min(hours, budget / rate) * 3600
        if time.time() >= deadline:
            raise ValueError("Original financial deadline reached")
    except (OSError, ValueError, KeyError, TypeError, LeaseError, IndexError) as exc:
        raise FinancialStop(str(exc)) from exc
    if time.time() + reserve >= deadline:
        error = FinancialStop if for_resume else Deferred
        raise error("Healthy guard has insufficient time for this benchmark reservation")
    return fixed


def completed_worker(root, identity):
    args = identity["command"]
    try:
        request = Path(args[args.index("--worker") + 1]).resolve()
        if not request.is_relative_to(runtime(root).resolve()):
            raise ValueError("Task request escapes the approved runtime")
        task = read_json(request)
        config, seed, fold = task["config_id"], task["seed"], task["fold"]
        if (
            task["kind"] != "development"
            or Path(config).name != config
            or "\\" in config
            or fold not in range(3)
            or not isinstance(seed, int)
        ):
            raise ValueError("Only the current development task may finish under this lease")
        folder = root / "vision/research/experiments" / config / str(seed) / f"fold-{fold}"
        result = read_json(folder / "result.json")
        protocol = read_json(root / "vision/research/protocol.json")
        frozen = next(item for item in protocol["candidate_configs"] if item["config_id"] == config)
        if (
            result["status"] != "completed"
            or result["config"] != frozen
            or result["seed"] != seed
            or result["fold"] != fold
            or result["protocol_hash"] != protocol["protocol_hash"]
        ):
            raise ValueError("Current worker did not record a completed matching fit")
        files = result["files"]
        if not files or {
            p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
        } != set(files) | {"result.json"}:
            raise ValueError("Completed fit files are missing or unexpected")
        for name, digest in files.items():
            raw_file = folder / name
            file = raw_file.resolve()
            if raw_file.is_symlink() or not file.is_relative_to(folder.resolve()):
                raise ValueError("Unsafe completed fit member")
            with file.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("Completed fit checksum differs")
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as exc:
        raise Deferred(str(exc)) from exc


def wait_primary(root, record, seconds):
    deadline = time.monotonic() + seconds
    while True:
        current = process(record["coordinator"]["pid"])
        if not matches(record["coordinator"], current) or current["state"] == "Z":
            raise Deferred("Primary coordinator changed before its pause")
        if current["state"] in {"T", "t"}:
            break
        guard_check(root, record["guard"])
        if time.monotonic() >= deadline:
            raise Deferred("Coordinator pause not yet observed; benchmark deferred")
        time.sleep(0.05)
    workers = children(record["coordinator"])
    if len(workers) > 1 or any(
        not command_is(item, "parallel_study.py") or "--worker" not in item["command"]
        for item in workers
    ):
        raise Deferred("Current worker identity cannot be established safely")
    if workers:
        save(root, record, waiting_worker=workers[0])
    while workers:
        guard_check(root, record["guard"])
        if not matches(record["coordinator"], process(record["coordinator"]["pid"])):
            raise Deferred("Primary coordinator changed")
        remaining = []
        for worker in workers:
            if alive(worker):
                remaining.append(worker)
            else:
                completed_worker(root, worker)
        workers = remaining
        if workers and time.monotonic() >= deadline:
            raise Deferred(
                "Current worker is still active; benchmark deferred without interruption"
            )
        if workers:
            time.sleep(2)
    current = process(record["coordinator"]["pid"])
    if not matches(record["coordinator"], current) or current["state"] not in {"T", "t"}:
        raise Deferred("Exclusive coordinator pause no longer holds")


def pilot_members(record):
    pilot = record.get("pilot")
    owned = []
    tag = (TAG + "=" + record["lease_id"]).encode()
    for entry in PROC.iterdir():
        if not entry.name.isdigit() or int(entry.name) <= 1:
            continue
        current = process(int(entry.name))
        if not current or current["state"] == "Z":
            continue
        if pilot and (current["group"] != pilot["pid"] or current["session"] != pilot["pid"]):
            continue
        try:
            tagged = tag in (entry / "environ").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        except PermissionError:
            if pilot is not None or command_is(current, "future_speed_pilot.py"):
                raise LeaseError("Cannot verify a possible owned benchmark process") from None
            continue
        if tagged:
            owned.append(current)
    return owned


def terminate_pilot(record, child=None, grace=20):
    members = pilot_members(record)
    for item in members:
        signal_identity(item, signal.SIGINT)
    deadline = time.monotonic() + grace
    while pilot_members(record) and time.monotonic() < deadline:
        time.sleep(0.2)
    for item in pilot_members(record):
        signal_identity(item, signal.SIGKILL)
    until = time.monotonic() + 1
    while pilot_members(record) and time.monotonic() < until:
        time.sleep(0.05)
    if pilot_members(record):
        raise LeaseError("Owned benchmark processes remain after termination")
    if child is not None:
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            raise LeaseError("Owned pilot did not terminate") from None


def restore_primary(root, record):
    if not record.get("pause_intended"):
        return True
    guard_check(root, record["guard"], reserve=30, for_resume=True)
    try:
        status = read_json(runtime(root) / "status.json")
        if status.get("state") in {"failed", "paused", "interrupted", "completed"}:
            return False
        if int((runtime(root) / "job.pid").read_text()) != record["coordinator"]["pid"]:
            return False
        current = process(record["coordinator"]["pid"])
        if not matches(record["coordinator"], current) or current["state"] == "Z":
            return False
        # A SIGSTOP may still be pending while /proc reports R/S. We own the
        # pause intent, so SIGCONT also cancels that pending stop before return.
        guard_check(root, record["guard"], reserve=30, for_resume=True)
        if not signal_identity(record["coordinator"], signal.SIGCONT):
            return False
        for _ in range(20):
            current = process(record["coordinator"]["pid"])
            if not matches(record["coordinator"], current):
                return False
            if current["state"] not in {"T", "t", "Z"}:
                return True
            time.sleep(0.05)
        return False
    except (OSError, ValueError, KeyError, LeaseError):
        return False


def valid_pod_id(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) is not None


def emergency_guard_identity():
    pod_id = os.environ.get("RUNPOD_POD_ID")
    return (
        {"pod_id": pod_id, "emergency_identity_only": True}
        if valid_pod_id(pod_id)
        else {"emergency_identity_only": True}
    )


def initial_stop_guard(root):
    # A missing/corrupt control record never creates a fresh clock or budget.
    # Only the provider-supplied current-Pod identity may fill that gap.
    try:
        guard = read_json(runtime(root) / "watchdog.json")
        required = ("started_unix", "hourly_usd", "budget_usd", "max_hours")
        if not isinstance(guard, dict) or not valid_pod_id(guard.get("pod_id")):
            raise ValueError("No usable guard object")
        values = [float(guard[key]) for key in required]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite guard values")
        started, rate, budget, hours = values
        if not (
            started <= time.time() + 60 and 0 < rate <= 2.5 and 0 < budget <= 40 and 0 < hours <= 32
        ):
            raise ValueError("Unusable original guard limits")
        return guard
    except (OSError, ValueError, KeyError, TypeError):
        return emergency_guard_identity()


def financial_stop(root, record, reason):
    r = runtime(root)
    captured = record.get("guard")
    if not isinstance(captured, dict):
        captured = emergency_guard_identity()
    stop = r.parent / "STOP_REQUESTED"
    if not stop.exists():
        stop.write_text("benchmark_lease_financial_stop: " + reason + "\n")
    status_path = r / "status.json"
    try:
        status = read_json(status_path)
        if not isinstance(status, dict):
            raise ValueError("Primary status is not a JSON object")
    except (OSError, ValueError, TypeError):
        with contextlib.suppress(OSError):
            (r / ("status.invalid-" + uuid.uuid4().hex + ".json")).write_bytes(
                status_path.read_bytes()
            )
        status = {}
    if status.get("state") not in {"completed", "failed", "paused", "interrupted"}:
        status.update(state="failed", phase="benchmark_lease_guard", error=reason)
    else:
        status["benchmark_lease_stop_reason"] = reason
    published = True
    try:
        atomic_json(status_path, status)
    except OSError:
        published = False
    try:
        guard_pid = int((r / "watchdog.pid").read_text())
        current = process(guard_pid)
        latest = read_json(r / "watchdog.json")
        healthy = (
            published
            and not captured.get("emergency_identity_only", False)
            and all(
                key in captured
                for key in ("started_unix", "hourly_usd", "budget_usd", "max_hours", "pod_id")
            )
            and isinstance(latest, dict)
            and command_is(current, "watchdog.py")
            and workspace_matches(current, root)
            and current["state"] not in {"Z", "T", "t"}
            and latest.get("state") == "monitoring"
            and latest.get("pod_id") == captured.get("pod_id")
            and all(
                latest.get(key) == captured.get(key)
                for key in ("started_unix", "hourly_usd", "budget_usd", "max_hours")
            )
            and 0 <= time.time() - latest["updated_unix"] <= 45
        )
    except (OSError, ValueError, KeyError, LeaseError, TypeError):
        healthy = False
    if not healthy:
        path = Path(__file__).with_name("watchdog.py")
        spec = importlib.util.spec_from_file_location("lease_watchdog", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pod_id = captured.get("pod_id")
        if not valid_pod_id(pod_id):
            pod_id = emergency_guard_identity().get("pod_id")
        if not valid_pod_id(pod_id):
            raise LeaseError("No trusted current Pod identity is available for the stop request")
        module.stop_compute(pod_id)


def job_owner_changed(root, record):
    try:
        return int((runtime(root) / "job.pid").read_text()) != record["coordinator"]["pid"]
    except (OSError, ValueError):
        return False


def recover_owner(root, record):
    try:
        terminate_pilot(record)
        resumed = restore_primary(root, record)
    except (FinancialStop, OSError, LeaseError) as exc:
        financial_stop(root, record, str(exc))
        resumed = False
    if not resumed and not job_owner_changed(root, record):
        financial_stop(root, record, "Primary recovery could not be verified")
    save(root, record, state="recovered" if resumed else "attention", primary_resumed=resumed)
    return record


def explicit_deadline(record):
    key = {"waiting_for_primary": "wait_deadline_unix", "pilot_running": "pilot_deadline_unix"}.get(
        record.get("state")
    )
    value = record.get(key) if key else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def expire_owner(root, lease_id, owner):
    path = runtime(root) / "gpu-benchmark-lease.json"
    latest = read_json(path)
    deadline = explicit_deadline(latest)
    if latest.get("lease_id") != lease_id or deadline is None or time.time() < deadline:
        return False
    if not alive(owner):
        return True
    signal_identity(owner, signal.SIGTERM)
    until = time.monotonic() + 20
    while alive(owner) and time.monotonic() < until:
        current = read_json(path)
        if current.get("lease_id") != lease_id or (
            current.get("state") in TERMINAL and current.get("primary_resumed")
        ):
            return False
        # A legitimate phase transition replaces its deadline; do not enforce
        # an expired wait deadline after the pilot phase has already started.
        active_deadline = explicit_deadline(current)
        if active_deadline is not None and time.time() < active_deadline:
            return False
        time.sleep(0.2)
    if alive(owner):
        latest = read_json(path)
        if latest.get("lease_id") != lease_id:
            return False
        signal_identity(owner, signal.SIGKILL)
    until = time.monotonic() + 2
    while alive(owner) and time.monotonic() < until:
        time.sleep(0.05)
    if alive(owner):
        raise LeaseError("Deadline owner termination could not be verified")
    return True


def recover_watch(root, lease_id):
    path = runtime(root) / "gpu-benchmark-lease.json"
    record = read_json(path)
    if record["lease_id"] != lease_id:
        return
    atomic_json(
        runtime(root) / f"lease-recovery-{lease_id}.ready",
        {"lease_id": lease_id, "owner": record["owner"]},
    )
    while True:
        latest = read_json(path)
        if latest["lease_id"] != lease_id or (
            latest["state"] in TERMINAL and latest.get("primary_resumed")
        ):
            return
        if not alive(latest["owner"]):
            break
        deadline = explicit_deadline(latest)
        if (
            deadline is not None
            and time.time() >= deadline
            and expire_owner(root, lease_id, latest["owner"])
        ):
            break
        time.sleep(1)
    with lease_lock(root):
        latest = read_json(path)
        if latest["lease_id"] == lease_id and not alive(latest["owner"]):
            recover_owner(root, latest)


def run_lease(root, future_models, output, wait_seconds=900, max_seconds=3600, epochs=12):
    root, future_models, output = root.resolve(), future_models.resolve(), output.resolve()
    if output.exists() or not future_models.is_file():
        raise LeaseError("Pilot output must be fresh and future model source must exist")
    if not 0 <= wait_seconds <= 900 or not 0 < max_seconds <= 3600 or not 1 <= epochs <= 12:
        raise LeaseError("Benchmark wait/runtime/epoch bounds exceeded")
    if any(
        output.is_relative_to(root / "vision" / name)
        for name in ("research", "protected", "images", "data", "source", "ad2")
    ):
        raise LeaseError("Pilot output cannot overwrite scientific source or results")
    with lease_lock(root):
        r = runtime(root)
        prior = (
            read_json(r / "gpu-benchmark-lease.json")
            if (r / "gpu-benchmark-lease.json").exists()
            else {}
        )
        if prior and (prior.get("state") not in TERMINAL or alive(prior["owner"])):
            raise LeaseError("Previous lease must finish or recover before replacement")
        try:
            fixed = guard_check(root, reserve=wait_seconds + max_seconds + 60)
        except FinancialStop as exc:
            financial_stop(root, {"guard": initial_stop_guard(root)}, str(exc))
            raise
        coordinator = process(int((r / "job.pid").read_text()))
        if (
            not command_is(coordinator, "parallel_study.py")
            or not workspace_matches(coordinator, root)
            or "--worker" in coordinator["command"]
            or coordinator["state"] in {"Z", "T", "t"}
        ):
            raise LeaseError("Primary coordinator is not an active, independently running process")
        owner = process(os.getpid())
        if owner["session"] != owner["pid"] or owner["group"] != owner["pid"]:
            raise LeaseError("Launch the lease through setsid for isolated signal ownership")
        record = {
            "lease_id": str(uuid.uuid4()),
            "state": "waiting_for_primary",
            "owner": owner,
            "coordinator": coordinator,
            "guard": fixed,
            "pause_intended": False,
            "created_unix": time.time(),
            "updated_unix": time.time(),
            "output": str(output),
            "progress_path": str(output / "progress.json"),
            "pilot_max_seconds": max_seconds,
            "future_models_sha256": hashlib.sha256(future_models.read_bytes()).hexdigest(),
        }
        save(root, record)
        child = None
        outcome = "failed"
        financial_reason = None
        try:
            with (r / ("lease-recovery-" + record["lease_id"] + ".log")).open("ab") as log:
                watcher = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--workspace",
                        str(root),
                        "--recover-watch",
                        record["lease_id"],
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            ready = r / ("lease-recovery-" + record["lease_id"] + ".ready")
            until = time.monotonic() + 5
            while not ready.exists():
                if watcher.poll() is not None or time.monotonic() >= until:
                    raise LeaseError("Independent recovery watcher did not become ready")
                time.sleep(0.1)
            save(
                root,
                record,
                recovery=process(watcher.pid),
                pause_intended=True,
                wait_started_unix=time.time(),
                wait_deadline_unix=time.time() + wait_seconds,
            )
            if not signal_identity(coordinator, signal.SIGSTOP):
                raise Deferred("Coordinator changed before the pause")
            wait_primary(root, record, wait_seconds)
            guard_check(root, fixed, reserve=max_seconds + 60)
            project = Path(__file__).resolve().parents[2]
            command = [
                str(project / ".venv-runpod/bin/python"),
                "-u",
                str(project / "scripts/runpod/future_speed_pilot.py"),
                "--workspace",
                str(root),
                "--future-models",
                str(future_models),
                "--output",
                str(output),
                "--epochs",
                str(epochs),
            ]
            env = {**os.environ, TAG: record["lease_id"]}
            save(
                root,
                record,
                state="pilot_running",
                pilot_started_unix=time.time(),
                pilot_deadline_unix=time.time() + max_seconds,
            )
            with (r / ("lease-pilot-" + record["lease_id"] + ".log")).open("ab") as log:
                child = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                    close_fds=True,
                )
            identity = process(child.pid)
            if identity is None:
                raise LeaseError("Pilot exited before its ownership could be recorded")
            save(
                root,
                record,
                state="pilot_running",
                pilot=identity,
            )
            deadline = time.monotonic() + max_seconds
            while child.poll() is None:
                guard_check(root, fixed)
                if time.monotonic() >= deadline:
                    raise LeaseError(
                        "Pilot deadline reached; preserving evidence and resuming the study"
                    )
                time.sleep(2)
            if child.returncode:
                raise LeaseError(f"Pilot exited with code{child.returncode}")
            if read_json(output / "progress.json").get("state") != "completed":
                raise LeaseError("Pilot did not publish its completion marker")
            outcome = "completed"
        except Deferred as exc:
            outcome = "deferred"
            save(root, record, reason=str(exc))
        except FinancialStop as exc:
            save(root, record, reason=str(exc), financial_stop=True)
            financial_reason = str(exc)
        except BaseException as exc:
            save(root, record, reason=type(exc).__name__ + ": " + str(exc))
        finally:
            try:
                terminate_pilot(record, child)
                if financial_reason is not None:
                    financial_stop(root, record, financial_reason)
                resumed = restore_primary(root, record)
            except (FinancialStop, OSError, LeaseError) as exc:
                save(root, record, cleanup_error=str(exc))
                financial_stop(root, record, str(exc))
                resumed = False
            if not resumed and not job_owner_changed(root, record):
                financial_stop(root, record, "Primary resumption could not be verified")
            save(root, record, state=outcome if resumed else "attention", primary_resumed=resumed)

        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--future-models", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wait-seconds", type=float, default=900)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--recover-watch")
    args = parser.parse_args()
    if args.recover_watch:
        recover_watch(args.workspace.resolve(), args.recover_watch)
        return
    if args.future_models is None or args.output is None:
        parser.error("Future source and fresh output are required")

    def interrupt(number, frame):
        raise KeyboardInterrupt("Benchmark lease interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    print(
        json.dumps(
            run_lease(
                args.workspace,
                args.future_models,
                args.output,
                args.wait_seconds,
                args.max_seconds,
                args.epochs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
