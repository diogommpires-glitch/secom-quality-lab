"""Run independent frozen fits on dedicated GPUs; model recipes stay unchanged."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from secom_quality.common import fingerprint, read_json, utc_now, write_json
from secom_quality.vision.study import exclusive


def initial_tasks(protocol: dict) -> list[dict]:
    return [
        {
            "kind": "development",
            "config_id": config["config_id"],
            "seed": protocol["seeds"][0],
            "fold": fold,
        }
        for config in protocol["candidate_configs"]
        for fold in range(3)
    ]


def finalist_tasks(protocol: dict, leaders: list[dict]) -> list[dict]:
    return [
        {"kind": "development", "config_id": row["config"]["config_id"], "seed": seed, "fold": fold}
        for row in leaders
        for seed in protocol["seeds"][1:]
        for fold in range(3)
    ]


def task_key(task: dict) -> tuple:
    return task["kind"], task["config_id"], task["seed"], task.get("fold")


def worker_environment(gpu: str) -> dict[str, str]:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
    }


def dispatch(
    root: Path, tasks: list[dict], gpus: list[str], popen=subprocess.Popen, pause=time.sleep
):
    """Only one fit per GPU and one owner per task; interruption stops child jobs."""
    if not gpus or len(set(gpus)) != len(gpus) or len({task_key(t) for t in tasks}) != len(tasks):
        raise ValueError("Dedicated GPU IDs and disjoint tasks are required")
    pending = deque(tasks)
    active: dict[str, tuple[Any, dict]] = {}
    runtime = root / "vision/runtime/runpod"
    runtime.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    try:
        while pending or active:
            if (root / "vision/runtime/STOP_REQUESTED").exists():
                raise KeyboardInterrupt("Remote stop requested; completed fits are preserved")
            for gpu in gpus:
                if gpu not in active and pending:
                    task = pending.popleft()
                    token = fingerprint(task)[:16]
                    request = runtime / f"task-{token}.json"
                    write_json(request, task)
                    with (runtime / f"task-{token}.log").open("ab") as log:
                        process = popen(
                            [
                                sys.executable,
                                "-u",
                                str(script),
                                "--workspace",
                                str(root),
                                "--worker",
                                str(request),
                            ],
                            env=worker_environment(gpu),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=os.name != "nt",
                        )
                    active[gpu] = (process, task)
                    print(
                        json.dumps(
                            {"event": "fit_started", "gpu": gpu, "pid": process.pid, **task}
                        ),
                        flush=True,
                    )
            for gpu, (process, task) in list(active.items()):
                code = process.poll()
                if code is not None:
                    if code != 0:
                        raise RuntimeError(f"Worker failed ({code}); inspect {runtime}: {task}")
                    print(json.dumps({"event": "fit_recorded", "gpu": gpu, **task}), flush=True)
                    del active[gpu]
            if active:
                pause(0.5)
    finally:
        for process, _ in active.values():
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGINT if os.name != "nt" else signal.SIGTERM)
                    process.wait(timeout=20)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    process.kill()


def worker(root: Path, task: dict):
    from secom_quality.vision.data import load_development
    from secom_quality.vision.models import fit_model, load_predictor
    from secom_quality.vision.study import checked_protocol, freeze_selection, run_fit

    protocol = checked_protocol(root)
    if task["kind"] == "development":
        config = next(
            c for c in protocol["candidate_configs"] if c["config_id"] == task["config_id"]
        )
        outcome = run_fit(root, config, task["seed"], task["fold"], device="cuda")
        print(json.dumps({"status": outcome["status"], "task": task}), flush=True)
        return
    if task["kind"] != "final":
        raise ValueError("Unknown worker task")
    research = root / "vision/research"
    selection = freeze_selection(root)
    if (research / "final-access.json").exists():
        raise ValueError("Final models cannot be refitted after test exposure")
    config = next(
        c
        for c in [selection["config"], *selection["reference_configs"]]
        if c["config_id"] == task["config_id"]
    )
    reference = config != selection["config"]
    allowed_seeds = [selection["serving_seed"]] if reference else selection["seeds"]
    if task["seed"] not in allowed_seeds:
        raise ValueError("Final task is outside the frozen selection")
    directory = research / "final.staging" / ("references" if reference else "models")
    directory /= config["config_id"] if reference else str(task["seed"])
    if (directory / "predictor.json").exists() and (directory / "weights.safetensors").exists():
        load_predictor(directory, device="cpu")
        return
    fit_model(config, load_development(root), task["seed"], directory, device="cuda")


def run(root: Path, gpus: list[str], include_ad2: bool = True):
    from secom_quality.vision.bundle import build_bundle
    from secom_quality.vision.protocol import checked_protocol
    from secom_quality.vision.report import build_report
    from secom_quality.vision.robustness import run_stress
    from secom_quality.vision.study import (
        _ordering,
        comparison,
        evaluate_final,
        freeze_selection,
        run_experiments,
    )

    root = root.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "4"
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(4)
    numerical_threads = threadpool_limits(limits=4)
    runtime = root / "vision/runtime"
    (runtime / "runpod").mkdir(parents=True, exist_ok=True)
    (runtime / "runpod/job.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    state: dict[str, Any] = {
        "pid": os.getpid(),
        "started_at": utc_now(),
        "gpus": gpus,
        "platform": "RunPod Linux",
    }

    def status(phase, outcome="running", **extra):
        if outcome == "running" and (runtime / "STOP_REQUESTED").exists():
            raise KeyboardInterrupt("Remote stop requested before starting another phase")
        state.update(phase=phase, state=outcome, updated_at=utc_now(), **extra)
        write_json(runtime / "run.json", state)
        write_json(runtime / "runpod/status.json", state)
        print(json.dumps(state), flush=True)

    with exclusive(runtime / "runpod-coordinator.lock"):
        try:
            protocol = checked_protocol(root)
            status("initial_matrix", protocol_hash=protocol["protocol_hash"])
            with exclusive(root / "vision/research/experiment.lock"):
                dispatch(root, initial_tasks(protocol), gpus)
            rows = comparison(root)
            if any(row["completed_fits"] != 3 for row in rows):
                raise ValueError("Initial matrix did not finish")
            leaders = sorted([row for row in rows if row["valid"]], key=_ordering)[:2]
            if not leaders:
                raise ValueError("No valid initial candidates")
            status("five_seed_finalists")
            with exclusive(root / "vision/research/experiment.lock"):
                dispatch(root, finalist_tasks(protocol, leaders), gpus)
            # All requested fits exist; this verifies and writes the exact shared comparison.
            run_experiments(root, device="cuda")
            status("development_corruption_analysis")
            run_stress(root, device="cuda")
            selection = freeze_selection(root)
            status("frozen_final_training", selection_hash=selection["selection_hash"])
            if not (root / "vision/research/final-access.json").exists():
                tasks = [
                    {"kind": "final", "config_id": selection["config"]["config_id"], "seed": seed}
                    for seed in selection["seeds"]
                ]
                tasks += [
                    {
                        "kind": "final",
                        "config_id": c["config_id"],
                        "seed": selection["serving_seed"],
                    }
                    for c in selection["reference_configs"]
                ]
                dispatch(root, tasks, gpus)
            status("final_evaluation")
            evaluate_final(root, device="cuda")
            build_report(root)
            bundle = build_bundle(root)
            status("primary_study", primary_status="completed", bundle_id=bundle["bundle_id"])
            if include_ad2:
                from secom_quality.vision.ad2 import audit_ad2, export_ad2, freeze_ad2, run_ad2

                source = root / "vision/ad2/source/manifest.json"
                if not source.exists():
                    audit_ad2(root, root / "vision/ad2_downloads/mvtec_ad_2.tar.gz")
                status("ad2_independent_challenge")
                freeze_ad2(root)
                run_ad2(root, device="cuda")
                export_ad2(root, device="cuda")
            status("research", "completed", external_submission=False)
        except BaseException as exc:
            status(
                state.get("phase", "startup"),
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            numerical_threads.restore_original_limits()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--skip-ad2", action="store_true")
    args = parser.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt("Remote termination requested; preserve checkpoints")

    signal.signal(signal.SIGTERM, interrupt)
    if args.worker:
        worker(args.workspace.resolve(), read_json(args.worker))
    else:
        run(args.workspace, args.gpus.split(","), not args.skip_ad2)


if __name__ == "__main__":
    main()
