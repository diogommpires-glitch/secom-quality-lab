"""Small RunPod REST controller. Read-only by default; no credentials in output."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://rest.runpod.io/v1"
PROJECT = Path(__file__).resolve().parents[2]
KEY_FILE = Path.home() / ".config/secom_quality_lab/runpod.json"


def load_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key and KEY_FILE.exists():
        key = json.loads(KEY_FILE.read_text(encoding="utf-8-sig"))["api_key"]
    if not key:
        raise RuntimeError(
            "Set RUNPOD_API_KEY or run scripts/runpod/credentials.ps1. Never paste a key into chat."
        )
    return key.strip()


class Client:
    def __init__(self, key, opener=urllib.request.urlopen):
        self.key, self.opener = key, opener

    def request(self, method, path, payload=None):
        readonly = method == "GET" and re.fullmatch(
            r"/(pods|networkvolumes|templates)(/[A-Za-z0-9_-]+)?(\?includeRunpodTemplates=true)?",
            path,
        )
        mutation = method == "POST" and (
            path in {"/pods", "/networkvolumes"}
            or re.fullmatch(r"/pods/[A-Za-z0-9_-]+/(stop|start)", path)
        )
        if not (readonly or mutation):
            raise ValueError("Unsupported method/route; volume deletion is never automated")
        if not re.fullmatch(
            r"/(pods|networkvolumes|templates)(/[A-Za-z0-9_-]+)?(/stop|/start)?(\?includeRunpodTemplates=true)?",
            path,
        ):
            raise ValueError("Unsupported RunPod API route")
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            BASE + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
                "User-Agent": "secom-quality-lab/1.0",
            },
        )
        try:
            with self.opener(request, timeout=45) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            # Provider responses can echo env/payloads; never print their body.
            raise RuntimeError(
                f"RunPod {method} {path.split('?')[0]} failed with HTTP {exc.code}"
            ) from None
        except urllib.error.URLError:
            raise RuntimeError(
                "RunPod API connection failed; inspect resource state before retrying a mutation"
            ) from None


def safe_pod(pod):
    return {
        name: pod.get(name)
        for name in (
            "id",
            "name",
            "desiredStatus",
            "gpuCount",
            "memoryInGb",
            "vcpuCount",
            "costPerHr",
            "adjustedCostPerHr",
            "publicIp",
            "portMappings",
            "networkVolumeId",
            "lastStartedAt",
        )
    }


def validate_plan(plan):
    if (
        plan.get("cloudType") != "SECURE"
        or type(plan.get("gpuCount")) is not int
        or plan["gpuCount"] not in (1, 2)
        or plan.get("interruptible") is not False
    ):
        raise ValueError("Reviewed plan requires one or two on-demand Secure Cloud GPUs")
    if plan.get("gpuTypeIds") != ["NVIDIA GeForce RTX 5090"]:
        raise ValueError("GPU substitution requires a reviewed plan")
    for key in ("max_hourly_usd", "budget_usd", "max_hours"):
        if not math.isfinite(float(plan.get(key, 0))) or not 0 < float(plan[key]) <= 1000:
            raise ValueError("Positive bounded cost and runtime limits are required")
    if float(plan["max_hourly_usd"]) > 2.50 or float(plan["budget_usd"]) > 100:
        raise ValueError(
            "Initial authorization boundary is $2.50/hour and $100; explicitly review a new plan to increase it"
        )
    if float(plan["max_hours"]) > 40:
        raise ValueError("Initial runtime authorization boundary is 40 hours")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", plan.get("dataCenterId", "")):
        raise ValueError("Choose an available Secure Cloud data center tomorrow")
    if not plan.get("templateId") or "SELECT" in plan["templateId"]:
        raise ValueError("Choose and verify an official PyTorch SSH-enabled template tomorrow")


def create_resources(client, plan, public_key, state_path):
    validate_plan(plan)
    if not public_key.startswith("ssh-ed25519 "):
        raise ValueError("Expected the dedicated SSH public key, never a private key")
    state_path = Path(state_path)
    if state_path.exists():
        raise ValueError(
            "Provisioning state exists; inspect it rather than creating duplicate paid resources"
        )
    state_path.parent.mkdir(parents=True, exist_ok=True)

    def save(state):
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        os.replace(temporary, state_path)

    save(
        {
            "created_unix": time.time(),
            "plan": plan,
            "state": "volume_creation_requested_inspect_before_retry",
        }
    )
    volume = client.request(
        "POST",
        "/networkvolumes",
        {"name": "secom-quality-lab", "size": 500, "dataCenterId": plan["dataCenterId"]},
    )
    state = {
        "created_unix": time.time(),
        "volume_id": volume["id"],
        "plan": plan,
        "state": "volume_created",
    }
    save(state)
    payload = {
        key: plan[key]
        for key in ("cloudType", "gpuCount", "gpuTypeIds", "interruptible", "templateId")
    }
    payload.update(
        name="secom-quality-lab",
        computeType="GPU",
        containerDiskInGb=50,
        dataCenterIds=[plan["dataCenterId"]],
        networkVolumeId=volume["id"],
        volumeMountPath="/workspace",
        # A single worker still needs host RAM for the frozen CPU/reservoir work.
        # Bootstrap independently requires 55 GiB actual RAM (about 59.1 GB).
        minRAMPerGPU=56 if plan["gpuCount"] == 1 else 32,
        minVCPUPerGPU=8,
        # CUDA 13.0 runtime also works with newer supported CUDA 13.x drivers.
        allowedCudaVersions=["13.0", "13.1", "13.2", "13.3", "13.4"],
        supportPublicIp=True,
        ports=["22/tcp"],
        env={"PUBLIC_KEY": public_key.strip()},
    )
    # Mutations are never retried automatically: a lost response may mean a Pod exists.
    pod = client.request("POST", "/pods", payload)
    state.update(pod_id=pod["id"], pod=safe_pod(pod), state="pod_created")
    save(state)
    try:
        hourly = float(pod["costPerHr"])
        if not math.isfinite(hourly) or hourly <= 0 or hourly > plan["max_hourly_usd"]:
            raise ValueError("Actual Pod price exceeds or does not establish the reviewed ceiling")
    except (KeyError, TypeError, ValueError):
        client.request("POST", f"/pods/{pod['id']}/stop")
        state["state"] = "stopped_price_check_failed"
        save(state)
        raise ValueError(
            "Pod stopped because its actual hourly price was missing or exceeded the reviewed ceiling"
        ) from None
    state["actual_hourly_usd"] = hourly
    state["state"] = "awaiting_verified_watchdog_and_pilot"
    save(state)
    return state


def wait_ssh(client, pod_id, timeout=600):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        pod = client.request("GET", f"/pods/{pod_id}")
        address, port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
        if address and port:
            ipaddress.ip_address(address)
            port = int(port)
            if not 1 <= port <= 65535:
                raise ValueError("Invalid returned SSH port")
            return address, port
        time.sleep(10)
    raise TimeoutError(
        "Pod SSH was not ready within the bounded wait; stop compute and inspect the Pod"
    )


def ssh_arguments(address, port, key_path, command):
    ipaddress.ip_address(address)
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-i",
        str(key_path),
        "-p",
        str(int(port)),
        "root@" + address,
        command,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["list", "templates", "status", "provision", "stop", "start", "ssh"]
    )
    parser.add_argument("--pod-id")
    parser.add_argument("--plan", type=Path, default=PROJECT / "configs/runpod-plan.json")
    parser.add_argument("--state", type=Path, default=PROJECT / "workspace/runpod/provision.json")
    parser.add_argument("--public-key", type=Path, default=KEY_FILE.parent / "id_ed25519.pub")
    parser.add_argument("--private-key", type=Path, default=KEY_FILE.parent / "id_ed25519")
    parser.add_argument("--command", default="nvidia-smi")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.action == "provision" and not args.apply:
        print(args.plan.read_text())
        print(
            "DRY RUN: no resources created. Resolve availability, template, exact quote and approval before --apply."
        )
        return
    client = Client(load_key())
    if args.action == "list":
        print(
            json.dumps(
                {
                    "pods": [safe_pod(p) for p in client.request("GET", "/pods")],
                    "volumes": [
                        {k: v.get(k) for k in ("id", "name", "size", "dataCenterId")}
                        for v in client.request("GET", "/networkvolumes")
                    ],
                },
                indent=2,
            )
        )
    elif args.action == "templates":
        templates = client.request("GET", "/templates?includeRunpodTemplates=true")
        print(
            json.dumps(
                [
                    {k: t.get(k) for k in ("id", "name", "imageName", "isRunpod")}
                    for t in templates
                    if str(t.get("imageName", "")).startswith("runpod/pytorch")
                ],
                indent=2,
            )
        )
    elif args.action == "provision":
        print(
            json.dumps(
                create_resources(
                    client,
                    json.loads(args.plan.read_text()),
                    args.public_key.read_text(),
                    args.state,
                ),
                indent=2,
            )
        )
    else:
        if not args.pod_id or not re.fullmatch(r"[A-Za-z0-9_-]+", args.pod_id):
            parser.error("A valid --pod-id is required")
        if args.action in {"stop", "start"}:
            if not args.apply:
                print(f"DRY RUN: would {args.action} this Pod and retain its network volume")
            else:
                client.request("POST", f"/pods/{args.pod_id}/{args.action}")
                print(
                    f"{args.action.title()} requested. Check provider status; storage remains billable."
                )
        elif args.action == "status":
            print(json.dumps(safe_pod(client.request("GET", f"/pods/{args.pod_id}")), indent=2))
        else:
            address, port = wait_ssh(client, args.pod_id)
            subprocess.run(
                ssh_arguments(address, port, args.private_key, args.command),
                check=True,
                timeout=900,
            )


if __name__ == "__main__":
    main()
