"""Measure cloud throughput using development images only, without test metrics."""

import argparse
import json
import os
import re
from contextlib import suppress
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch

from secom_quality.common import write_json
from secom_quality.vision.data import load_development
from secom_quality.vision.models import fit_model
from secom_quality.vision.protocol import checked_protocol
from secom_quality.vision.study import run_fit


def parse_gpu_ids(value):
    """Choose one or two distinct local CUDA indices, defaulting to GPU 0."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(,[0-9]+)?", value):
        raise ValueError("SECOM_GPU_IDS must contain one or two comma-separated CUDA indices")
    gpus = [int(index) for index in value.split(",")]
    if len(set(gpus)) != len(gpus):
        raise ValueError("SECOM_GPU_IDS must contain distinct CUDA indices")
    return gpus


def parse_memory_limit(value):
    """Parse a cgroup byte limit; max, negative or malformed values are unbounded."""
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    return limit if limit >= 0 else None


def parse_cpu_limit(value):
    """Convert cgroup quota/period to a conservative whole-CPU capacity."""
    try:
        quota, period = (int(part) for part in value.split())
    except (AttributeError, TypeError, ValueError):
        return None
    return quota // period if quota >= 0 and period > 0 else None


def read_cgroup_limits(root=Path("/sys/fs/cgroup")):
    def read(relative):
        try:
            return (root / relative).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None

    memory = read("memory.max")
    if memory is None:
        memory = read("memory/memory.limit_in_bytes")
    cpu = read("cpu.max")
    if cpu is None:
        quota, period = read("cpu/cpu.cfs_quota_us"), read("cpu/cpu.cfs_period_us")
        cpu = f"{quota} {period}" if quota is not None and period is not None else None
    return {
        "cgroup_memory_limit_bytes": parse_memory_limit(memory),
        "cgroup_cpu_limit_cpus": parse_cpu_limit(cpu),
    }


def resource_capacity(cgroup_root=Path("/sys/fs/cgroup")):
    """Distinguish physical host totals from the process's effective allocation."""
    host_ram = psutil.virtual_memory().total
    host_cpus = psutil.cpu_count() or 0
    affinity_cpus = None
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        with suppress(OSError, TypeError):
            affinity_cpus = len(affinity(0))
    limits = read_cgroup_limits(cgroup_root)
    ram_limit = limits["cgroup_memory_limit_bytes"]
    cpu_limits = [host_cpus, affinity_cpus, limits["cgroup_cpu_limit_cpus"]]
    return {
        "host_system_ram_bytes": host_ram,
        "host_logical_cpus": host_cpus,
        "system_ram_bytes": min(host_ram, ram_limit) if ram_limit is not None else host_ram,
        "logical_cpus": min(limit for limit in cpu_limits if limit is not None),
        "affinity_logical_cpus": affinity_cpus,
        **limits,
    }


def check_hardware(gpu_ids):
    """Check execution resources without changing the frozen scientific protocol."""
    gpus = parse_gpu_ids(gpu_ids)
    if not torch.cuda.is_available() or max(gpus) >= torch.cuda.device_count():
        raise ValueError("Every requested CUDA GPU must be available")
    names = [torch.cuda.get_device_name(gpu) for gpu in gpus]
    if any(name != "NVIDIA GeForce RTX 5090" for name in names):
        raise ValueError("Only the reviewed NVIDIA GeForce RTX 5090 is allowed")
    # GPU firmware reserves part of advertised VRAM; 24 GB cards expose about 23 GiB.
    if any(torch.cuda.get_device_properties(gpu).total_memory < 23 * 1024**3 for gpu in gpus):
        raise ValueError("Each selected GPU needs at least 24 GB advertised VRAM")
    min_ram_gib = 55 if len(gpus) == 1 else 59
    resources = resource_capacity()
    ram_bytes = resources["system_ram_bytes"]
    # 60 GB advertised hosts expose about 55.9 GiB. The unchanged CPU/reservoir
    # workflow needs at least 55 GiB actual RAM even with only one GPU worker.
    if ram_bytes < min_ram_gib * 1024**3:
        raise ValueError(f"This GPU allocation needs at least {min_ram_gib} GiB actual system RAM")
    cpus = resources["logical_cpus"]
    if cpus < 8 * len(gpus):
        raise ValueError("Each selected GPU needs at least 8 available logical CPUs")
    return {
        "gpu_ids": gpus,
        "gpu_names": names,
        **resources,
        "min_system_ram_gib": min_ram_gib,
        "min_advertised_vram_gb": 24,
    }


def main(root, gpu_ids="0"):
    hardware = check_hardware(gpu_ids)
    gpus = hardware["gpu_ids"]
    directories = [root / f"vision/pilots/runpod-{probe}" for probe in range(2)]
    report_path = root / "vision/runtime/runpod/pilot.json"
    if report_path.exists() or any(directory.exists() for directory in directories):
        raise ValueError("Pilot evidence exists; inspect it or choose a new pilot workspace")
    protocol = checked_protocol(root)
    train = load_development(root).sort_values("record_id").groupby("label", sort=True).head(16)
    results = []
    for probe, resolution in enumerate(([704, 256], [1056, 384])):
        gpu = gpus[probe % len(gpus)]
        device = f"cuda:{gpu}"
        with torch.cuda.device(gpu):
            assert torch.cuda.is_bf16_supported(), "BF16 must work on each selected GPU"
            x = torch.ones((32, 32), device=device, dtype=torch.bfloat16)
            assert torch.isfinite(x @ x).all()
        torch.cuda.reset_peak_memory_stats(gpu)
        config = {
            "config_id": f"cloud-pilot-{probe}",
            "family": "supersimplenet",
            "resolution": resolution,
            "lr_multiplier": 1.0,
            "epochs": 1,
        }
        directory = directories[probe]
        started = perf_counter()
        predictor = fit_model(config, train, protocol["seeds"][0], directory, device=device)
        scores, maps = predictor.predict([Path(train.iloc[0].image_path)])
        assert np.isfinite(scores).all() and np.isfinite(maps[0]).all()
        history = json.loads((directory / "training_history.json").read_text())
        results.append(
            {
                "gpu": gpu,
                "name": torch.cuda.get_device_name(gpu),
                "resolution": resolution,
                "epoch_seconds": history[-1]["seconds"],
                "augmented_examples": history[-1]["examples"],
                "elapsed_seconds": perf_counter() - started,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(gpu),
                "estimated_100_epoch_fold_hours": history[-1]["seconds"]
                / history[-1]["examples"]
                * 1312
                * 100
                / 3600,
            }
        )
        del predictor, maps
        torch.cuda.empty_cache()
    # One real frozen normal-only fold measures the CPU/reservoir bottleneck too.
    first = next(c for c in protocol["candidate_configs"] if c["family"] == "patchcore")
    baseline = run_fit(root, first, protocol["seeds"][0], 0, device=f"cuda:{gpus[0]}")
    report = {
        "development_only": True,
        "hardware": hardware,
        "neural_pilots": results,
        "patchcore_first_fold_seconds": baseline["elapsed_seconds"],
        "initial_ssn_gpu_hours_estimate": sum(
            r["estimated_100_epoch_fold_hours"] * 9 for r in results
        ),
        "interpretation": "Throughput/memory probes, not predictive evidence. Reforecast cost before launching the full matrix.",
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--gpus", default=os.environ.get("SECOM_GPU_IDS", "0"))
    args = parser.parse_args()
    main(args.workspace.resolve(), args.gpus)
