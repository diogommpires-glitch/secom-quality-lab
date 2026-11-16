"""Pod-local budget/runtime guard; account-management credentials stay local.

This is a software guard, not a provider-enforced spending cap. Test the Pod's
own runpodctl shutdown capability before launching the long study.
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.runpod.process_identity import coordinator, matches, option, process, workspace_matches


def decision(started, now, rate, budget, max_hours, job_state):
    if (
        not all(math.isfinite(v) for v in (started, now, rate, budget, max_hours))
        or min(rate, budget, max_hours) <= 0
    ):
        raise ValueError("Finite positive billing limits are required")
    elapsed = max(0.0, now - started)
    if elapsed >= max_hours * 3600:
        return "wall_time_limit"
    if elapsed / 3600 * rate >= budget:
        return "compute_budget_limit"
    if job_state.get("state") in {"failed", "paused", "interrupted"}:
        return "job_" + job_state["state"]
    if job_state.get("phase") == "research" and job_state.get("state") == "completed":
        return "job_completed"
    return None


def stop_compute(pod_id, run=subprocess.run):
    if not isinstance(pod_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
        raise ValueError("Invalid Pod ID")
    # Uses the preconfigured Pod-scoped credential, never the account key.
    # Current CLI is noun/verb; older Pod images may still bundle verb/noun.
    for command in (["runpodctl", "pod", "stop", pod_id], ["runpodctl", "stop", "pod", pod_id]):
        result = run(command, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return
    raise RuntimeError(
        "Pod self-stop failed; provider output is withheld to avoid credential leakage"
    )


def write_state(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_guard(root, pod_id, started, rate, budget, max_hours, *, proc=Path("/proc"), now=None):
    runtime = root / "vision/runtime/runpod"
    now = time.time() if now is None else now
    try:
        record = json.loads((runtime / "watchdog.json").read_text())
        current = process(int((runtime / "watchdog.pid").read_text()), proc)
        expected = {
            "pod_id": pod_id,
            "started_unix": started,
            "hourly_usd": rate,
            "budget_usd": budget,
            "max_hours": max_hours,
        }
        command = current["command"] if current else []
        arguments = {
            "--pod-id": pod_id,
            "--started-unix": started,
            "--hourly-usd": rate,
            "--budget-usd": budget,
            "--max-hours": max_hours,
        }
        for flag, expected_value in arguments.items():
            actual = option(command, flag)
            if (actual if flag == "--pod-id" else float(actual)) != expected_value:
                raise ValueError("Guard command differs from the current Pod or original limits")
        if (
            not isinstance(record, dict)
            or not matches(record.get("identity"), current)
            or not workspace_matches(current, root)
            or not any(Path(part).name == "watchdog.py" for part in current["command"])
            or current["state"] in {"T", "t"}
            or "--verify" in command
            or record.get("state") != "monitoring"
            or not 0 <= now - float(record["updated_unix"]) <= 45
            or any(record.get(key) != value for key, value in expected.items())
            or decision(started, now, rate, budget, max_hours, {}) is not None
        ):
            raise ValueError("Guard identity, current Pod, freshness or original limits mismatch")
    except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "A live creation-bound guard for this workspace and Pod is required"
        ) from exc
    return record


def signal_coordinator(root, proc=Path("/proc")):
    identity = coordinator(root, proc)
    if (
        identity is None
        or not hasattr(os, "pidfd_open")
        or not hasattr(signal, "pidfd_send_signal")
    ):
        return False
    try:
        descriptor = os.pidfd_open(identity["pid"])
        try:
            if not matches(identity, coordinator(root, proc)):
                return False
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            return True
        finally:
            os.close(descriptor)
    except OSError:
        return False  # Provider stop must still run when local ownership/signalling fails.


def monitor(root, pod_id, started, rate, budget, max_hours):
    if pod_id != os.environ.get("RUNPOD_POD_ID"):
        raise ValueError("Guard must target the provider-supplied current Pod")
    runtime = root / "vision/runtime"
    status = runtime / "runpod/status.json"
    guard = runtime / "runpod/watchdog.json"
    guard.parent.mkdir(parents=True, exist_ok=True)
    identity = process(os.getpid())
    if identity is None or not workspace_matches(identity, root):
        raise ValueError("Cannot establish guard process creation and workspace identity")
    (guard.parent / "watchdog.pid").write_text(str(identity["pid"]) + "\n")
    while True:
        try:
            state = json.loads(status.read_text()) if status.exists() else {}
            if not isinstance(state, dict):
                state = {}
        except (ValueError, OSError):
            state = {}
        reason = decision(started, time.time(), rate, budget, max_hours, state)
        record = {
            "identity": identity,
            "state": "monitoring",
            "pod_id": pod_id,
            "started_unix": started,
            "updated_unix": time.time(),
            "hourly_usd": rate,
            "budget_usd": budget,
            "max_hours": max_hours,
            "estimated_compute_usd": max(0, time.time() - started) / 3600 * rate,
        }
        write_state(guard, record)
        if reason:
            runtime.mkdir(parents=True, exist_ok=True)
            (runtime / "STOP_REQUESTED").write_text(reason + "\n")
            write_state(guard, {**record, "state": "stopping_compute", "reason": reason})
            signal_coordinator(root)
            time.sleep(25)  # Give workers time to preserve the latest valid checkpoint.
            while True:
                try:
                    stop_compute(pod_id)
                    # A successful stop may end this process before another write.
                    write_state(
                        guard, {**record, "state": "provider_stop_accepted", "reason": reason}
                    )
                    return
                except (OSError, RuntimeError, subprocess.TimeoutExpired):
                    write_state(
                        guard,
                        {
                            **record,
                            "state": "self_stop_retrying",
                            "reason": reason,
                            "note": "Work halted; compute billing is not confirmed stopped. Inspect the provider.",
                        },
                    )
                    time.sleep(30)
        time.sleep(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--pod-id",
        default=os.environ.get("RUNPOD_POD_ID"),
        required=not os.environ.get("RUNPOD_POD_ID"),
    )
    parser.add_argument("--started-unix", type=float, required=True)
    parser.add_argument("--hourly-usd", type=float, required=True)
    parser.add_argument("--budget-usd", type=float, default=100)
    parser.add_argument("--max-hours", type=float, default=40)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.pod_id != os.environ.get("RUNPOD_POD_ID"):
        parser.error("Guard must target the provider-supplied current Pod")
    action = verify_guard if args.verify else monitor
    action(
        args.workspace.resolve(),
        args.pod_id,
        args.started_unix,
        args.hourly_usd,
        args.budget_usd,
        args.max_hours,
    )
