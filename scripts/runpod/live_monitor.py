"""Read-only progress and resource sampling for the frozen RunPod study."""

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process(pid, proc=Path("/proc"), ticks=100):
    directory = proc / str(pid)
    try:
        fields = (directory / "stat").read_text().rpartition(") ")[2].split()
        command = (directory / "cmdline").read_bytes().decode(errors="replace").split("\0")
        counters = dict(line.split(":", 1) for line in (directory / "io").read_text().splitlines())
        memory = next(
            (
                line.split()[1]
                for line in (directory / "status").read_text().splitlines()
                if line.startswith("VmRSS:")
            ),
            "0",
        )
        return {
            "pid": int(pid),
            "parent": int(fields[1]),
            "state": fields[0],
            "start_ticks": int(fields[19]),
            "cpu_seconds": (int(fields[11]) + int(fields[12])) / ticks,
            "rchar": int(counters.get("rchar", 0)),
            "wchar": int(counters.get("wchar", 0)),
            "rss_bytes": int(memory) * 1024,
            "wait_channel": (directory / "wchan").read_text().strip(),
            "command": command,
        }
    except (OSError, ValueError, IndexError):
        return None


def is_study_process(record, worker):
    return bool(
        record
        and any(Path(part).name == "parallel_study.py" for part in record["command"])
        and ("--worker" in record["command"]) == worker
    )


def fit_counts(root, protocol):
    initial = {
        "completed": 0,
        "failed": 0,
        "expected": len(protocol.get("candidate_configs", [])) * 3,
    }
    total = {"completed": 0, "failed": 0, "unknown": 0}
    configs = {item["config_id"] for item in protocol.get("candidate_configs", [])}
    seeds = protocol.get("seeds", [])
    directory = root / "vision/research/experiments"
    for path in directory.glob("*/*/fold-*/result.json"):
        config, seed, fold, _ = path.relative_to(directory).parts
        if fold not in {"fold-0", "fold-1", "fold-2"}:
            continue
        status = read_json(path).get("status")
        total[status if status in {"completed", "failed"} else "unknown"] += 1
        if (
            seeds
            and config in configs
            and seed == str(seeds[0])
            and status in {"completed", "failed"}
        ):
            initial[status] += 1
    return {"initial": initial, "total_development": total}


def task_identity(record, runtime):
    try:
        command = record["command"]
        request = Path(command[command.index("--worker") + 1]).resolve()
        if not request.is_relative_to(runtime.resolve()):
            return None
        task = read_json(request)
        return {key: task.get(key) for key in ("kind", "config_id", "seed", "fold")}
    except (ValueError, IndexError):
        return None


def training_progress(root, task):
    if not task or task.get("kind") != "development":
        return None
    config, seed, fold = task.get("config_id"), task.get("seed"), task.get("fold")
    if (
        not isinstance(config, str)
        or Path(config).name != config
        or "\\" in config
        or not isinstance(seed, int)
        or fold not in range(3)
    ):
        return None
    path = (
        root
        / "vision/research/experiments"
        / config
        / str(seed)
        / f"fold-{fold}.staging/model/progress.json"
    )
    result = read_json(path)
    return (
        {key: result.get(key) for key in ("epoch", "epochs", "mean_loss", "seconds")}
        if result
        else None
    )


def observed_deltas(item, old, elapsed):
    if old and elapsed > 0 and old["start_ticks"] == item["start_ticks"]:
        for counter in ("cpu_seconds", "rchar", "wchar"):
            item[counter + "_delta"] = max(0, item[counter] - old[counter])
        item["cpu_percent_one_core"] = item["cpu_seconds_delta"] / elapsed * 100


def operational_benchmark(root, coordinator, proc, ticks):
    """Observe an explicit live lease without treating it as study-selection work."""
    lease = read_json(root / "vision/runtime/runpod/gpu-benchmark-lease.json")
    if not isinstance(lease, dict) or lease.get("state") != "pilot_running":
        return None

    def matches(record, identity, script):
        try:
            return bool(
                record
                and record["state"] != "Z"
                and record["pid"] > 1
                and record["pid"] == int(identity["pid"])
                and record["start_ticks"] == int(identity["start_ticks"])
                and any(Path(part).name == script for part in record["command"])
            )
        except (KeyError, TypeError, ValueError):
            return False

    result = {
        "lease_id": lease.get("lease_id"),
        "state": lease["state"],
        "scope": "development_training_only_performance_not_selection",
    }
    try:
        owner = process(int(lease["owner"]["pid"]), proc, ticks)
        if (
            not matches(owner, lease["owner"], "benchmark_lease.py")
            or "--recover-watch" in owner["command"]
        ):
            return {**result, "attention_required": True, "reason": "lease_owner_not_verified"}
        if not matches(coordinator, lease["coordinator"], "parallel_study.py"):
            return {**result, "attention_required": True, "reason": "primary_identity_changed"}
        pilot = process(int(lease["pilot"]["pid"]), proc, ticks)
        if not matches(pilot, lease["pilot"], "future_speed_pilot.py"):
            return {
                **result,
                "attention_required": True,
                "reason": "pilot_finished_or_identity_changed",
            }
        output = Path(lease["output"]).resolve()
        progress_path = Path(lease["progress_path"]).resolve()
        if (
            output.is_relative_to(root.resolve())
            or not output.is_relative_to(root.resolve().parent)
            or progress_path != output / "progress.json"
        ):
            return {**result, "attention_required": True, "reason": "unexpected_progress_path"}
        result["output"] = str(output)
        progress = read_json(progress_path)
        result["progress"] = {
            key: progress.get(key)
            for key in ("state", "stage", "profile", "resolution", "epoch", "epochs")
        }
        directory = progress.get("fit_directory")
        if directory and Path(directory).resolve().is_relative_to(output):
            training = read_json(Path(directory) / "progress.json")
            result["training"] = (
                {key: training.get(key) for key in ("epoch", "epochs", "mean_loss", "seconds")}
                if training
                else None
            )
        pilot.pop("command")
        result["pilot"] = pilot
        return result
    except (KeyError, OSError, TypeError, ValueError):
        return {**result, "attention_required": True, "reason": "incomplete_lease_metadata"}


def record_benchmark_sample(current):
    """Append derived resource observations only after the pilot creates its output."""
    benchmark = current.get("operational_benchmark") or {}
    if not benchmark.get("pilot") or not benchmark.get("output"):
        return
    output = Path(benchmark["output"])
    if not output.is_dir() or not (output / "progress.json").is_file():
        return
    path = output / "resource-samples.jsonl"
    if path.is_symlink():
        benchmark["resource_log_status"] = "refused_symlink"
        return
    pilot = benchmark["pilot"]
    sample = {
        "sample_unix": current["sample_unix"],
        "lease_id": benchmark["lease_id"],
        "progress": benchmark.get("progress"),
        "training": benchmark.get("training"),
        "cpu_percent_one_core": pilot.get("cpu_percent_one_core"),
        "rchar_delta": pilot.get("rchar_delta"),
        "wchar_delta": pilot.get("wchar_delta"),
        "process_rss_bytes": pilot.get("rss_bytes"),
        "gpu": current["gpu"],
        "guard_state": current["guard_state"],
        "guard_age_seconds": current["guard_age_seconds"],
    }
    try:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample, allow_nan=False) + "\n")
        benchmark["resource_log_status"] = "recorded"
    except OSError:
        benchmark["resource_log_status"] = "unavailable"


def gpu_sample(run=subprocess.run):
    try:
        result = run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        values = [float(item.strip()) for item in result.stdout.strip().splitlines()[0].split(",")]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError("Invalid GPU counters")
        return {
            "available": True,
            "utilization_percent": values[0],
            "memory_used_mib": values[1],
            "memory_total_mib": values[2],
        }
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {"available": False}


def snapshot(root, previous=None, *, now=None, proc=Path("/proc"), gpu=None, ticks=100):
    now = time.time() if now is None else now
    runtime = root / "vision/runtime/runpod"
    status, guard = read_json(runtime / "status.json"), read_json(runtime / "watchdog.json")
    try:
        coordinator = process(int((runtime / "job.pid").read_text()), proc, ticks)
    except (OSError, ValueError):
        coordinator = None
    if not is_study_process(coordinator, False) or coordinator["state"] == "Z":
        coordinator = None
    workers = []
    elapsed = now - previous["sample_unix"] if previous else 0
    prior = {item["pid"]: item for item in previous.get("workers", [])} if previous else {}
    if coordinator:
        for directory in proc.iterdir():
            if not directory.name.isdigit():
                continue
            item = process(int(directory.name), proc, ticks)
            if not item or item["parent"] != coordinator["pid"] or not is_study_process(item, True):
                continue
            item["task"] = task_identity(item, runtime)
            item["training"] = training_progress(root, item["task"])
            old = prior.get(item["pid"])
            observed_deltas(item, old, elapsed)
            item.pop("command")
            workers.append(item)
    benchmark = operational_benchmark(root, coordinator, proc, ticks)
    actors = list(workers)
    if benchmark and benchmark.get("pilot"):
        old = (previous.get("operational_benchmark") or {}).get("pilot") if previous else None
        observed_deltas(benchmark["pilot"], old, elapsed)
        actors.append(benchmark["pilot"])
    gpu = gpu_sample() if gpu is None else gpu
    activity = "sampling" if not previous else "waiting"
    if gpu.get("available") and gpu["utilization_percent"] > 1:
        activity = "gpu_work"
    elif any(item.get("cpu_seconds_delta", 0) >= 0.02 for item in actors):
        activity = "cpu_work"
    elif any(item.get("rchar_delta", 0) + item.get("wchar_delta", 0) > 0 for item in actors):
        activity = "io_progress"
    remaining = None
    try:
        age = max(0, now - float(guard["started_unix"])) / 3600
        remaining = max(
            0,
            min(
                float(guard["max_hours"]) - age,
                float(guard["budget_usd"]) / float(guard["hourly_usd"]) - age,
            ),
        )
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        pass
    stopped = (runtime.parent / "STOP_REQUESTED").exists()
    return {
        "sample_unix": now,
        "state": status.get("state", "unknown"),
        "phase": status.get("phase"),
        "coordinator_alive": bool(coordinator),
        "coordinator_state": coordinator["state"] if coordinator else None,
        "activity": activity,
        "activity_note": "Zero GPU utilization alone does not establish a stall.",
        "workers": workers,
        "operational_benchmark": benchmark,
        "cpu_percent_basis": "100% means one logical CPU core",
        "gpu": gpu,
        "fits": fit_counts(root, read_json(root / "vision/research/protocol.json")),
        "guard_state": guard.get("state", "unknown"),
        "guard_age_seconds": max(0, now - guard["updated_unix"])
        if isinstance(guard.get("updated_unix"), (int, float))
        else None,
        "remaining_guard_hours": remaining,
        "stop_requested": stopped,
        "terminal": stopped
        or (
            not coordinator
            and status.get("state") in {"completed", "failed", "interrupted", "paused"}
        ),
    }


def monitor(root, interval=5):
    previous, signature = None, None
    ticks = os.sysconf("SC_CLK_TCK")
    while True:
        current = snapshot(root, previous, ticks=ticks)
        record_benchmark_sample(current)
        atomic_json(root / "vision/runtime/runpod/live-telemetry.json", current)
        compact = {
            key: current[key]
            for key in ("state", "phase", "activity", "fits", "guard_state", "terminal")
        }
        compact["active"] = [
            {"task": item["task"], "training": item["training"]} for item in current["workers"]
        ]
        benchmark = current.get("operational_benchmark")
        if benchmark:
            compact["operational_benchmark"] = {
                key: benchmark.get(key)
                for key in (
                    "lease_id",
                    "state",
                    "scope",
                    "progress",
                    "training",
                    "attention_required",
                    "reason",
                )
            }
        changed = json.dumps(compact, sort_keys=True)
        if changed != signature:
            print(json.dumps({"sample_unix": current["sample_unix"], **compact}), flush=True)
            signature = changed
        if current["terminal"]:
            return
        previous = current
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    arguments = parser.parse_args()
    if not math.isfinite(arguments.interval) or arguments.interval < 1:
        parser.error("Sampling interval must be finite and at least one second")
    monitor(arguments.workspace.resolve(), arguments.interval)
