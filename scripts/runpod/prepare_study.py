"""Wait for verified source preparation or bounded pilots; never launch the study matrix."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
AD2_SHA256 = "c0ded99ef32bfc8e352d52beb44515e5b292b8598cb963aadfa91ca0763505e4"
AD2_BYTES = 32739596982


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def check_stop(root):
    if (root / "vision/runtime/STOP_REQUESTED").exists():
        raise InterruptedError("STOP_REQUESTED prevents further preparation")


def process_info(pid, proc=Path("/proc")):
    try:
        fields = (proc / str(pid) / "stat").read_text().rpartition(") ")[2].split()
        return {"state": fields[0], "parent": int(fields[1])}
    except (OSError, ValueError, IndexError):
        return None


def stop_owned_fetch(root, proc=Path("/proc")):
    marker = root / "vision/runtime/runpod/ad2-download.pid"
    if not marker.exists():
        return []
    wrapper = int(marker.read_text().strip())
    if wrapper <= 1:
        raise ValueError("Invalid recorded AD2 wrapper PID")
    stopped = []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        pid = int(directory.name)
        info = process_info(pid, proc)
        if pid == os.getpid() or not info or info["state"] == "Z" or info["parent"] != wrapper:
            continue
        try:
            command = (directory / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if not any(Path(part.decode(errors="replace")).name == "fetch_ad2.py" for part in command):
            continue
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while (info := process_info(pid, proc)) and info["state"] != "Z":
            check_stop(root)
            if time.monotonic() >= deadline:
                raise RuntimeError("Owned AD2 fetch child did not stop; refusing archive promotion")
            time.sleep(0.2)
        stopped.append(pid)
    return stopped


def provenance(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or (
        value.get("state") != "complete"
        or value.get("sha256") != AD2_SHA256
        or type(value.get("received_bytes")) is not int
        or value["received_bytes"] != AD2_BYTES
        or value.get("expected_bytes", AD2_BYTES) != AD2_BYTES
    ):
        raise ValueError("AD2 acquisition provenance does not cover the complete pinned archive")
    return value


def verified_hash(path, root):
    if path.stat().st_size != AD2_BYTES:
        raise ValueError("AD2 archive length differs from the pinned source")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            check_stop(root)
            digest.update(chunk)
    if digest.hexdigest() != AD2_SHA256:
        raise ValueError("AD2 archive SHA256 differs from the pinned source")


def promote_upload(root, staged, original_provenance):
    destination = root / "vision/ad2_downloads"
    destination.mkdir(parents=True, exist_ok=True)
    canonical = destination / "mvtec_ad_2.tar.gz"
    if canonical.exists():
        raise FileExistsError("Canonical AD2 archive already exists; refusing to overwrite it")
    verified_hash(staged, root)
    provenance(original_provenance)
    stopped = stop_owned_fetch(root)
    check_stop(root)
    # link/unlink gives the same-filesystem move an atomic no-overwrite guarantee.
    os.link(staged, canonical)
    staged.unlink()
    original = original_provenance.read_bytes()
    temporary = destination / f"download.{os.getpid()}.tmp"
    temporary.write_bytes(original)
    os.replace(temporary, destination / "download.json")
    atomic_json(
        destination / "cloud-transfer.json",
        {
            "state": "completed",
            "archive_sha256": AD2_SHA256,
            "bytes": AD2_BYTES,
            "original_provenance": str(original_provenance),
            "stopped_owned_fetch_pids": stopped,
            "method": "Verified staged upload, no-overwrite promotion; original acquisition provenance retained",
        },
    )
    return canonical


def run_command(command, root):
    check_stop(root)
    child = subprocess.Popen([str(part) for part in command], cwd=PROJECT)
    try:
        while child.poll() is None:
            check_stop(root)
            time.sleep(15)
        if child.returncode:
            raise RuntimeError(f"Preparation child exited with code {child.returncode}")
    except BaseException:
        if child.poll() is None:
            child.terminate()
            child.wait()
        raise


def wait_bootstrap(root):
    runtime = root / "vision/runtime/runpod"
    python = PROJECT / ".venv-runpod/bin/python"
    protocol = root / "vision/research/protocol.json"
    while True:
        check_stop(root)
        pid_path = runtime / "bootstrap.pid"
        info = process_info(int(pid_path.read_text().strip())) if pid_path.exists() else None
        alive = bool(info and info["state"] != "Z")
        log = runtime / "bootstrap.log"
        footer = log.exists() and "Bootstrap complete." in log.read_text(errors="replace")[-65536:]
        if pid_path.exists() and not alive and not protocol.exists():
            raise RuntimeError("Bootstrap is dead or zombie and no frozen protocol exists")
        if python.exists() and protocol.exists() and (footer or not alive):
            run_command(
                [
                    python,
                    "-c",
                    "from pathlib import Path; import sys; from secom_quality.vision.protocol import checked_protocol; checked_protocol(Path(sys.argv[1]))",
                    root,
                ],
                root,
            )
            return python
        time.sleep(15)


def audit(root):
    destination = root / "vision/ad2_downloads"
    canonical = destination / "mvtec_ad_2.tar.gz"
    acquired = destination / "download.json"
    staged_candidates = (
        destination / "mvtec_ad_2.uploading.tar.gz",
        Path("/workspace/mvtec_ad_2.uploading.tar.gz"),
    )
    while True:
        check_stop(root)
        if canonical.exists() and acquired.exists():
            provenance(acquired)
            verified_hash(canonical, root)
            break
        if not canonical.exists():
            staged = next(
                (
                    path
                    for path in staged_candidates
                    if path.exists() and path.stat().st_size >= AD2_BYTES
                ),
                None,
            )
            if staged is not None:
                source_record = Path("/workspace/download.json")
                if source_record.exists():
                    canonical = promote_upload(root, staged, source_record)
                    break
        time.sleep(15)
    python = wait_bootstrap(root)
    manifest = root / "vision/ad2/source/manifest.json"
    if not manifest.exists():
        run_command(
            [
                python,
                "-m",
                "secom_quality.cli",
                "--workspace",
                root,
                "vision",
                "ad2-audit",
                "--archive",
                canonical,
            ],
            root,
        )
    if not manifest.exists():
        raise RuntimeError("AD2 source audit did not create its completion manifest")


def pilots(root):
    python = wait_bootstrap(root)
    runtime = root / "vision/runtime/runpod"
    for script, marker in (("pilot.py", "pilot.json"), ("steady_pilot.py", "steady-pilot.json")):
        path = runtime / marker
        if not path.exists():
            run_command(
                [python, "-u", PROJECT / "scripts/runpod" / script, "--workspace", root], root
            )
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(result, dict)
            or result.get("development_only") is not True
            or len(result.get("neural_pilots", [])) != 2
        ):
            raise ValueError("Pilot completion marker is invalid")
    atomic_json(
        runtime / "preflight.json",
        {
            "state": "passed",
            "phase": "preflight",
            "pid": os.getpid(),
            "scope": "development_training_pilots_only",
            "training_autolaunch": False,
        },
    )
    atomic_json(
        runtime / "status.json",
        {
            "state": "awaiting_review",
            "phase": "preflight",
            "pid": os.getpid(),
            "training_autolaunch": False,
        },
    )


def main(root, phase):
    root = Path(root).resolve()
    runtime = root / "vision/runtime/runpod"
    marker = runtime / ("ad2-preparation.json" if phase == "audit" else "pilot-preparation.json")
    record = {
        "state": "preparing",
        "pid": os.getpid(),
        "scope": "development_source_audit_only"
        if phase == "audit"
        else "development_training_pilots_only",
    }
    if phase == "audit":
        record["archive_sha256"] = AD2_SHA256
    atomic_json(marker, record)
    try:
        (audit if phase == "audit" else pilots)(root)
        atomic_json(marker, {**record, "state": "completed"})
    except BaseException as exc:
        atomic_json(marker, {**record, "state": "failed", "error_type": type(exc).__name__})
        atomic_json(
            runtime / "status.json",
            {
                "phase": f"prepare_{phase}",
                "state": "failed",
                "pid": os.getpid(),
                "error_type": type(exc).__name__,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--phase", choices=("audit", "pilots"), required=True)
    args = parser.parse_args()
    main(args.workspace, args.phase)
