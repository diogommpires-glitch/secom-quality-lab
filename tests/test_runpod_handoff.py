"""Offline cloud-control checks: no credentials, paid resources or network needed."""

import importlib.util
import io
import json
import sys
import urllib.error
import zipfile
from pathlib import Path

import pytest


def load_script(name):
    path = Path(__file__).parents[1] / "scripts/runpod" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(f"reviewed_runpod_{name}", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


control = load_script("control")
packaging = load_script("package_handoff")
watchdog = load_script("watchdog")


class Response:
    def __init__(self, value):
        self.body = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def plan(gpu_count=1):
    return {
        "cloudType": "SECURE",
        "gpuCount": gpu_count,
        "gpuTypeIds": ["NVIDIA GeForce RTX 5090"],
        "interruptible": False,
        "dataCenterId": "EU-EXAMPLE-1",
        "templateId": "reviewed-template",
        "max_hourly_usd": 2.50,
        "budget_usd": 100,
        "max_hours": 40,
    }


class FakeClient:
    def __init__(self, hourly=2.38):
        self.hourly, self.calls = hourly, []

    def request(self, method, route, payload=None):
        self.calls.append((method, route, payload))
        if route == "/networkvolumes":
            return {"id": "volume123"}
        if method == "POST" and route == "/pods":
            return {
                "id": "pod123",
                "costPerHr": self.hourly,
                "env": {"RUNPOD_API_KEY": "must-not-print"},
            }
        if method == "GET":
            return {"id": "pod123", "desiredStatus": "EXITED", "networkVolumeId": "volume123"}
        return {}


def test_api_key_is_authorization_header_never_url_or_error_body():
    captured = []

    def opener(request, timeout):
        captured.append(request)
        assert timeout == 45
        return Response({"id": "pod123"})

    client = control.Client("secret-token", opener=opener)
    client.request("GET", "/pods/pod123")
    request = captured[0]
    assert request.get_header("Authorization") == "Bearer secret-token"
    assert request.get_header("User-agent") == "secom-quality-lab/1.0"
    assert "secret-token" not in request.full_url
    with pytest.raises(ValueError, match="Unsupported"):
        client.request("GET", "/pods?api_key=secret-token")

    def failure(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 401, "bad", {}, io.BytesIO(b'{"RUNPOD_API_KEY":"secret-token"}')
        )

    with pytest.raises(RuntimeError) as error:
        control.Client("secret-token", opener=failure).request("GET", "/pods")
    assert "secret-token" not in str(error.value)
    assert "HTTP 401" in str(error.value)


def test_dry_run_does_not_load_key_or_construct_client(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("A dry run contacted cloud or loaded credentials")

    monkeypatch.setattr(control, "Client", forbidden)
    monkeypatch.setattr(control, "load_key", forbidden)
    monkeypatch.setattr(sys, "argv", ["control.py", "provision"])
    control.main()
    assert "DRY RUN" in capsys.readouterr().out


@pytest.mark.parametrize("gpu_count,min_ram", [(1, 56), (2, 32)])
def test_provision_payload_uses_reviewed_gpus_and_persistent_storage(tmp_path, gpu_count, min_ram):
    client = FakeClient()
    state = control.create_resources(
        client, plan(gpu_count), "ssh-ed25519 public-key fixture", tmp_path / "state.json"
    )
    payload = next(payload for method, route, payload in client.calls if route == "/pods")
    assert payload["gpuCount"] == gpu_count
    assert payload["gpuTypeIds"] == ["NVIDIA GeForce RTX 5090"]
    assert payload["allowedCudaVersions"] == ["13.0", "13.1", "13.2", "13.3", "13.4"]
    assert payload["interruptible"] is False
    assert payload["cloudType"] == "SECURE"
    assert payload["networkVolumeId"] == "volume123"
    assert payload["volumeMountPath"] == "/workspace"
    assert payload["minRAMPerGPU"] == min_ram and payload["minVCPUPerGPU"] >= 8
    assert payload["containerDiskInGb"] == 50
    volume_payload = next(
        payload for _, route, payload in client.calls if route == "/networkvolumes"
    )
    assert volume_payload["size"] == 500
    assert payload["ports"] == ["22/tcp"]
    assert "RUNPOD_API_KEY" not in payload["env"]
    assert "must-not-print" not in json.dumps(state)
    assert state["actual_hourly_usd"] == 2.38
    assert all(method != "DELETE" for method, _, _ in client.calls)


@pytest.mark.parametrize(
    "changes",
    [{"gpuCount": count} for count in (0, 3, True, False, 1.0, "1", None)]
    + [
        {"gpuTypeIds": ["NVIDIA GeForce RTX 4090"]},
        {"gpuTypeIds": ["NVIDIA GeForce RTX 5090 D"]},
        {"cloudType": "COMMUNITY"},
        {"interruptible": True},
        {"max_hourly_usd": 2.51},
        {"budget_usd": 100.01},
        {"max_hours": 40.01},
    ],
)
def test_unreviewed_hardware_or_cost_boundaries_reject_before_cloud_requests(tmp_path, changes):
    client = FakeClient()
    with pytest.raises(ValueError):
        control.create_resources(
            client, plan() | changes, "ssh-ed25519 public-key fixture", tmp_path / "state.json"
        )
    assert client.calls == []
    assert not (tmp_path / "state.json").exists()


@pytest.mark.parametrize("hourly", [None, float("nan"), -1.0, 3.50])
def test_missing_or_excessive_price_requests_compute_stop_and_preserves_volume(tmp_path, hourly):
    client = FakeClient(hourly)
    with pytest.raises(ValueError, match=r"price|hourly"):
        control.create_resources(
            client, plan(), "ssh-ed25519 public-key fixture", tmp_path / "state.json"
        )
    assert any(
        route.startswith("/pods/pod123") and method in {"POST", "DELETE"}
        for method, route, _ in client.calls
    )
    assert not any(
        method == "DELETE" and route.startswith("/networkvolumes")
        for method, route, _ in client.calls
    )


def test_resource_creation_is_not_repeated_when_state_already_exists(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("{}")
    client = FakeClient()
    with pytest.raises(ValueError, match="state exists"):
        control.create_resources(client, plan(), "ssh-ed25519 public-key fixture", state)
    assert client.calls == []


def test_network_volume_deletion_is_not_an_allowed_controller_action():
    def forbidden(*args, **kwargs):
        raise AssertionError("Cloud request should have been rejected locally")

    with pytest.raises(ValueError):
        control.Client("token", opener=forbidden).request("DELETE", "/networkvolumes/volume123")


def test_ssh_polling_has_a_bounded_timeout_and_only_reads(monkeypatch):
    client = FakeClient()
    times = iter([0, 0, 2])
    monkeypatch.setattr(control.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(control.time, "sleep", lambda seconds: None)
    with pytest.raises(TimeoutError):
        control.wait_ssh(client, "pod123", timeout=1)
    assert all(method == "GET" for method, _, _ in client.calls)


def test_ssh_arguments_preserve_key_path_and_reject_address_injection(tmp_path):
    key = tmp_path / "path with spaces" / "key"
    arguments = control.ssh_arguments("127.0.0.1", 2222, key, "nvidia-smi")
    assert arguments[arguments.index("-i") + 1] == str(key)
    assert arguments[-2:] == ["root@127.0.0.1", "nvidia-smi"]
    with pytest.raises(ValueError):
        control.ssh_arguments("127.0.0.1; bad-command", 22, key, "nvidia-smi")


def test_handoff_whitelist_excludes_credentials_raw_data_and_environments(tmp_path):
    approved = {
        "src/secom_quality/vision/models.py": b"# reviewed public source\n",
        "src/secom_quality/web/assets/app.js": b"export const title='Lab';\n",
        "configs/runpod-plan.json": b'{"gpuCount":2}',
        "scripts/runpod/bootstrap.sh": b"#!/bin/bash\nset -euo pipefail\n",
        "scripts/runpod/credentials.ps1": b"# prompts for a key, contains no key\n",
        "docs/RUNPOD_PLAN.md": b"# Public handoff\n",
        "pyproject.toml": b"[project]\nname='synthetic'\n",
        "LICENSE": b"MIT\n",
        "workspace/vision/research/protocol.json": b'{"protocol_hash":"synthetic"}',
        "workspace/vision/runtime/local-stop.json": b'{"state":"stopped"}',
    }
    forbidden = [
        ".env",
        ".venv/private.txt",
        "workspace/vision/protected/test.png",
        "workspace/vision/research/experiments/weights.safetensors",
        "configs/.env",
        "configs/runpod-api-key.json",
        "configs/id_ed25519",
        "configs/private.pem",
        "docs/raw-test.png",
        "src/secom_quality/__pycache__/x.pyc",
    ]
    for name, body in approved.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    for name in forbidden:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SENTINEL_PRIVATE_VALUE_NEVER_PACKAGE")
    output = tmp_path / "release/source.zip"
    result = packaging.build_handoff(tmp_path, output)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "src/secom_quality/vision/models.py" in names
        assert "scripts/runpod/credentials.ps1" in names
        assert "migration/windows_protocol.json" in names
        assert "migration/windows_stop.json" in names
        assert not names.intersection(forbidden)
        assert all(
            b"SENTINEL_PRIVATE_VALUE_NEVER_PACKAGE" not in archive.read(name) for name in names
        )
        manifest = json.loads(archive.read("handoff-manifest.json"))
        assert names == set(manifest["files"]) | {"handoff-manifest.json"}
        import hashlib

        assert all(
            hashlib.sha256(archive.read(name)).hexdigest() == digest
            for name, digest in manifest["files"].items()
        )
    other = packaging.build_handoff(tmp_path, tmp_path / "release/source-second.zip")
    assert result["sha256"] == other["sha256"]


def test_watchdog_counts_installation_and_idle_time_toward_budget():
    assert watchdog.decision(0, 3600, 2.0, 2.0, 40, {}) == "compute_budget_limit"
    assert watchdog.decision(0, 3600, 2.0, 100, 1, {}) == "wall_time_limit"
    assert watchdog.decision(0, 60, 2.0, 100, 40, {"state": "failed"}) == "job_failed"
    assert (
        watchdog.decision(0, 60, 2.0, 100, 40, {"phase": "primary_study", "state": "completed"})
        is None
    )
    assert (
        watchdog.decision(0, 60, 2.0, 100, 40, {"phase": "research", "state": "completed"})
        == "job_completed"
    )
    with pytest.raises(ValueError, match="Finite"):
        watchdog.decision(0, 60, float("nan"), 100, 40, {})


def test_watchdog_stops_only_its_pod_and_redacts_provider_errors():
    calls = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return type("Result", (), {"returncode": 0})()

    watchdog.stop_compute("pod123", run=run)
    assert calls[0][0] == ["runpodctl", "pod", "stop", "pod123"]
    assert calls[0][1]["timeout"] == 60
    assert not any("volume" in str(argument) for argument in calls[0][0])

    def failed(*args, **kwargs):
        return type("Result", (), {"returncode": 1, "stderr": "secret-token"})()

    with pytest.raises(RuntimeError) as error:
        watchdog.stop_compute("pod123", run=failed)
    assert "secret-token" not in str(error.value)
    with pytest.raises(ValueError, match="Invalid"):
        watchdog.stop_compute("pod123;invalid", run=run)
