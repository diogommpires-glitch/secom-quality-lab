"""Offline resource and pilot scheduling checks; no CUDA allocation or data access."""

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

PATH = Path(__file__).parents[1] / "scripts/runpod/pilot.py"
SPEC = importlib.util.spec_from_file_location("runpod_pilot", PATH)
assert SPEC and SPEC.loader
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def hardware(
    monkeypatch,
    *,
    count=1,
    ram=60_000_000_000,
    cpus=8,
    vram=32 * 1024**3,
    name="NVIDIA GeForce RTX 5090",
    available=True,
):
    monkeypatch.setattr(pilot.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(pilot.torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(pilot.torch.cuda, "get_device_name", lambda _: name)
    monkeypatch.setattr(
        pilot.torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=vram)
    )
    monkeypatch.setattr(pilot.psutil, "virtual_memory", lambda: SimpleNamespace(total=ram))
    monkeypatch.setattr(pilot.psutil, "cpu_count", lambda: cpus)
    monkeypatch.setattr(
        pilot,
        "resource_capacity",
        lambda: {
            "host_system_ram_bytes": ram,
            "host_logical_cpus": cpus or 0,
            "system_ram_bytes": ram,
            "logical_cpus": cpus or 0,
        },
    )


@pytest.mark.parametrize("value", ["", "0,", "0,0", "01,1", "-1", "0.0", "a", "0,1,2", None])
def test_selected_gpu_ids_reject_invalid_or_duplicate_devices(value):
    with pytest.raises(ValueError, match="SECOM_GPU_IDS"):
        pilot.parse_gpu_ids(value)


def test_one_gpu_accepts_actual_sixty_gb_host_and_records_requirements(monkeypatch):
    hardware(monkeypatch)
    result = pilot.check_hardware("0")
    assert result["gpu_ids"] == [0]
    assert result["system_ram_bytes"] == 60_000_000_000
    assert result["logical_cpus"] == 8
    assert result["min_system_ram_gib"] == 55
    assert result["min_advertised_vram_gb"] == 24


@pytest.mark.parametrize("ids,count,minimum,cpus", [("0", 1, 55, 8), ("0,1", 2, 59, 16)])
def test_actual_ram_boundary_and_one_or_two_worker_cpu_requirements(
    monkeypatch, ids, count, minimum, cpus
):
    hardware(monkeypatch, count=count, ram=minimum * 1024**3, cpus=cpus)
    assert pilot.check_hardware(ids)["min_system_ram_gib"] == minimum
    hardware(monkeypatch, count=count, ram=minimum * 1024**3 - 1, cpus=cpus)
    with pytest.raises(ValueError, match="actual system RAM"):
        pilot.check_hardware(ids)
    hardware(monkeypatch, count=count, ram=minimum * 1024**3, cpus=cpus - 1)
    with pytest.raises(ValueError, match="logical CPUs"):
        pilot.check_hardware(ids)


@pytest.mark.parametrize(
    "resources,ids,error",
    [
        ({"available": False}, "0", "available"),
        ({"count": 1}, "1", "available"),
        ({"count": 1}, "0,1", "available"),
        ({"name": "NVIDIA GeForce RTX 5090 D"}, "0", "Only the reviewed"),
        ({"name": "NVIDIA GeForce RTX 4090"}, "0", "Only the reviewed"),
        ({"vram": 23 * 1024**3 - 1}, "0", "24 GB advertised VRAM"),
        ({"cpus": None}, "0", "logical CPUs"),
    ],
)
def test_hardware_rejects_missing_or_unreviewed_resources(monkeypatch, resources, ids, error):
    hardware(monkeypatch, **resources)
    with pytest.raises(ValueError, match=error):
        pilot.check_hardware(ids)


@pytest.mark.parametrize(
    "ids,devices",
    [("0", ["cuda:0", "cuda:0"]), ("1", ["cuda:1", "cuda:1"]), ("0,1", ["cuda:0", "cuda:1"])],
)
def test_both_pilot_resolutions_use_selected_gpus_and_distinct_evidence(
    tmp_path, monkeypatch, ids, devices
):
    monkeypatch.setattr(pilot, "check_hardware", lambda _: {"gpu_ids": pilot.parse_gpu_ids(ids)})
    config = {"config_id": "frozen-patchcore", "family": "patchcore"}
    monkeypatch.setattr(
        pilot, "checked_protocol", lambda _: {"seeds": [17], "candidate_configs": [config]}
    )
    train = pd.DataFrame(
        {"record_id": ["a", "b"], "label": [0, 1], "image_path": ["a.png", "b.png"]}
    )
    monkeypatch.setattr(pilot, "load_development", lambda _: train)
    monkeypatch.setattr(pilot.torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(pilot.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(pilot.torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(pilot.torch.cuda, "max_memory_allocated", lambda _: 1024)
    monkeypatch.setattr(pilot.torch.cuda, "get_device_name", lambda _: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(pilot.torch.cuda, "empty_cache", lambda: None)
    actual_ones = pilot.torch.ones
    monkeypatch.setattr(
        pilot.torch, "ones", lambda shape, *, device, dtype: actual_ones(shape, dtype=dtype)
    )
    events, fits = [], []

    def fit(probe, selected_train, seed, directory, *, device):
        index = len(fits)
        fits.append((dict(probe), directory, device))
        events.append(f"fit-{index}")
        assert seed == 17 and list(selected_train.record_id) == ["a", "b"]
        directory.mkdir(parents=True)
        (directory / "training_history.json").write_text(
            json.dumps([{"seconds": 2, "examples": 16}]), encoding="utf-8"
        )

        def predict(paths):
            events.append(f"predict-{index}")
            assert paths == [Path("a.png")]
            return np.array([0.5]), [np.ones((2, 2))]

        return SimpleNamespace(predict=predict)

    def baseline(root, candidate, seed, fold, *, device):
        events.append("patchcore")
        assert candidate == config and seed == 17 and fold == 0 and device == devices[0]
        return {"elapsed_seconds": 3}

    monkeypatch.setattr(pilot, "fit_model", fit)
    monkeypatch.setattr(pilot, "run_fit", baseline)
    pilot.main(tmp_path, ids)
    assert events == ["fit-0", "predict-0", "fit-1", "predict-1", "patchcore"]
    assert [fit[2] for fit in fits] == devices
    assert [fit[0]["resolution"] for fit in fits] == [[704, 256], [1056, 384]]
    assert [fit[0]["config_id"] for fit in fits] == ["cloud-pilot-0", "cloud-pilot-1"]
    assert [fit[1].name for fit in fits] == ["runpod-0", "runpod-1"]
    report = json.loads((tmp_path / "vision/runtime/runpod/pilot.json").read_text())
    assert report["development_only"] is True
    assert [row["gpu"] for row in report["neural_pilots"]] == pilot.parse_gpu_ids(ids) * (
        2 // len(pilot.parse_gpu_ids(ids))
    )


@pytest.mark.parametrize(
    "existing",
    ["vision/pilots/runpod-0", "vision/pilots/runpod-1", "vision/runtime/runpod/pilot.json"],
)
def test_existing_pilot_evidence_is_rejected_before_loading_data_or_fitting(
    tmp_path, monkeypatch, existing
):
    monkeypatch.setattr(pilot, "check_hardware", lambda _: {"gpu_ids": [0]})
    monkeypatch.setattr(pilot, "checked_protocol", lambda _: pytest.fail("Pilot work started"))
    path = tmp_path / existing
    if path.suffix:
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
    else:
        path.mkdir(parents=True)
    with pytest.raises(ValueError, match="Pilot evidence exists"):
        pilot.main(tmp_path)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("123456", 123456),
        ("0", 0),
        ("max", None),
        ("-1", None),
        ("garbled", None),
        ("1.5", None),
        (None, None),
    ],
)
def test_cgroup_memory_limit_parser(raw, expected):
    assert pilot.parse_memory_limit(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1500000 100000", 15),
        ("1599999 100000", 15),
        ("50000 100000", 0),
        ("max 100000", None),
        ("-1 100000", None),
        ("1 0", None),
        ("1 -2", None),
        ("invalid", None),
        ("1 2 3", None),
        (None, None),
    ],
)
def test_cgroup_cpu_limit_parser_floors_fractional_cpu_quota(raw, expected):
    assert pilot.parse_cpu_limit(raw) == expected


def test_effective_resource_caps_preserve_host_totals_and_use_affinity_and_quota(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        pilot.psutil, "virtual_memory", lambda: SimpleNamespace(total=1024 * 1024**3)
    )
    monkeypatch.setattr(pilot.psutil, "cpu_count", lambda: 120)
    monkeypatch.setattr(pilot.os, "sched_getaffinity", lambda _: set(range(20)), raising=False)
    (tmp_path / "memory.max").write_text(str(117 * 1024**3), encoding="utf-8")
    (tmp_path / "cpu.max").write_text("1599999 100000", encoding="utf-8")
    resources = pilot.resource_capacity(tmp_path)
    assert resources["host_system_ram_bytes"] == 1024 * 1024**3
    assert resources["host_logical_cpus"] == 120
    assert resources["system_ram_bytes"] == 117 * 1024**3
    assert resources["logical_cpus"] == 15
    assert resources["affinity_logical_cpus"] == 20
    monkeypatch.setattr(pilot.os, "sched_getaffinity", lambda _: set(range(12)))
    assert pilot.resource_capacity(tmp_path)["logical_cpus"] == 12
    monkeypatch.setattr(pilot.psutil, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        pilot.psutil, "virtual_memory", lambda: SimpleNamespace(total=60_000_000_000)
    )
    resources = pilot.resource_capacity(tmp_path)
    assert resources["logical_cpus"] == 8 and resources["system_ram_bytes"] == 60_000_000_000


@pytest.mark.parametrize("memory,cpu", [("max", "max 100000"), ("bad", "bad"), (None, None)])
def test_missing_unlimited_or_invalid_cgroup_files_fall_back_to_host_and_affinity(
    tmp_path, monkeypatch, memory, cpu
):
    monkeypatch.setattr(
        pilot.psutil, "virtual_memory", lambda: SimpleNamespace(total=60_000_000_000)
    )
    monkeypatch.setattr(pilot.psutil, "cpu_count", lambda: 16)
    monkeypatch.setattr(pilot.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    if memory is not None:
        (tmp_path / "memory.max").write_text(memory, encoding="utf-8")
        (tmp_path / "cpu.max").write_text(cpu, encoding="utf-8")
    resources = pilot.resource_capacity(tmp_path)
    assert resources["system_ram_bytes"] == 60_000_000_000
    assert resources["logical_cpus"] == 8
    assert resources["cgroup_memory_limit_bytes"] is None
    assert resources["cgroup_cpu_limit_cpus"] is None


def test_v1_cgroup_limits_are_used_when_v2_files_are_missing(tmp_path):
    for name, value in {
        "memory/memory.limit_in_bytes": "60000000000",
        "cpu/cpu.cfs_quota_us": "850000",
        "cpu/cpu.cfs_period_us": "100000",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(value, encoding="utf-8")
    assert pilot.read_cgroup_limits(tmp_path) == {
        "cgroup_memory_limit_bytes": 60_000_000_000,
        "cgroup_cpu_limit_cpus": 8,
    }


def test_inaccessible_cgroup_files_and_affinity_preserve_windows_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pilot.psutil, "virtual_memory", lambda: SimpleNamespace(total=60_000_000_000)
    )
    monkeypatch.setattr(pilot.psutil, "cpu_count", lambda: 8)

    def unavailable(*args, **kwargs):
        raise OSError("Unavailable")

    monkeypatch.setattr(pilot.os, "sched_getaffinity", unavailable, raising=False)
    monkeypatch.setattr(Path, "read_text", unavailable)
    resources = pilot.resource_capacity(tmp_path)
    assert resources["system_ram_bytes"] == 60_000_000_000
    assert resources["logical_cpus"] == 8 and resources["affinity_logical_cpus"] is None


@pytest.mark.parametrize(
    "ram,cpus,error", [(54 * 1024**3, 15, "actual system RAM"), (117 * 1024**3, 7, "logical CPUs")]
)
def test_hardware_guard_uses_effective_allocation_instead_of_large_host_totals(
    monkeypatch, ram, cpus, error
):
    hardware(monkeypatch, ram=1024 * 1024**3, cpus=120)
    monkeypatch.setattr(
        pilot,
        "resource_capacity",
        lambda: {
            "host_system_ram_bytes": 1024 * 1024**3,
            "host_logical_cpus": 120,
            "system_ram_bytes": ram,
            "logical_cpus": cpus,
        },
    )
    with pytest.raises(ValueError, match=error):
        pilot.check_hardware("0")
