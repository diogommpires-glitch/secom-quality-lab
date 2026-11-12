"""Verified development-image disk cache; canonical research manifests stay unchanged."""

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

CACHE = Path("/tmp/secom-development-cache/train")
EXPECTED_FILES = 4662
ARCHIVE_SHA256 = "edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8"
EXCLUDED = {
    "train/10301 (copy).png": "29ceeb9311153c1526ca7e623912d6f25da072738ffa62a8ccb37e2a7535403e",
    "train/10301_GT (copy).png": "d7bbd6bf21d1bbbc53acae6102699c43d960bbbc2b34ea3a53c614544e0366bb",
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def paths(root, cache):
    root = Path(root).resolve()
    cache = Path(cache)
    if cache != CACHE or not cache.is_absolute() or cache.is_symlink() or cache.parent.is_symlink():
        raise ValueError("Only the reviewed, ordinary local cache path is allowed")
    train = root / "vision/images/train"
    return (
        root,
        cache,
        train,
        train.with_name("train.network-original"),
        root / "vision/runtime/runpod",
    )


def expected(root):
    manifest = json.loads((root / "vision/source/manifest.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "source_fingerprint"}
    fingerprint = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if (
        manifest.get("source_fingerprint") != fingerprint
        or manifest.get("archive_sha256") != ARCHIVE_SHA256
    ):
        raise ValueError("Pinned source manifest checksum/archive differs")
    members = manifest["member_sha256"]
    if manifest.get("dataset_version_review", {}).get("excluded_exact_copy_files") != EXCLUDED:
        raise ValueError("Canonical duplicate exclusion differs from the reviewed source")
    files = {}
    for name, digest in members.items():
        member = PurePosixPath(name)
        if not name.startswith("train/"):
            continue
        if name in EXCLUDED:
            if digest != EXCLUDED[name]:
                raise ValueError("Excluded duplicate checksum differs")
            continue
        if (
            len(member.parts) != 2
            or member.suffix != ".png"
            or "\\" in name
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("Unexpected canonical development member")
        files[member.name] = digest
    if len(files) != EXPECTED_FILES:
        raise ValueError("Canonical development file count differs")
    return files, fingerprint


def check_members(directory, files):
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Expected an ordinary development directory")
    entries = list(directory.iterdir())
    if {path.name for path in entries} != set(files) or any(
        not path.is_file() or path.is_symlink() for path in entries
    ):
        raise ValueError("Development directory has missing or unexpected members")


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(directory, files):
    check_members(directory, files)
    for name, digest in files.items():
        if digest_file(directory / name) != digest:
            raise ValueError(f"Development byte checksum differs: {name}")


def record(root, cache, train, backup, fingerprint):
    return {
        "scope": "development_images_only",
        "root": str(root),
        "cache": str(cache),
        "canonical": str(train),
        "backup": str(backup),
        "source_fingerprint": fingerprint,
        "files": EXPECTED_FILES,
    }


def stage(root, cache=CACHE):
    root, cache, train, backup, runtime = paths(root, cache)
    files, fingerprint = expected(root)
    base = record(root, cache, train, backup, fingerprint)
    marker = runtime / "cache-stage.json"
    if cache.exists():
        prior = json.loads(marker.read_text()) if marker.exists() else {}
        if prior.get("state") != "completed" or any(
            prior.get(key) != value for key, value in base.items()
        ):
            raise FileExistsError(
                "Existing local cache is not a completed matching stage; preserve and inspect it"
            )
        verify(cache, files)
        return prior
    check_members(train, files)
    cache.mkdir(parents=True, exist_ok=False)
    progress = {**base, "state": "staging", "copied": 0, "started_unix": time.time()}
    atomic_json(runtime / "cache-progress.json", progress)

    def copy(name):
        source, destination = train / name, cache / name
        digest = hashlib.sha256()
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            while chunk := incoming.read(1024 * 1024):
                digest.update(chunk)
                outgoing.write(chunk)
        if digest.hexdigest() != files[name]:
            raise ValueError(f"Development byte checksum differs: {name}")
        shutil.copystat(source, destination, follow_symlinks=False)

    try:
        updated = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(copy, name) for name in files]
            for future in as_completed(futures):
                future.result()
                progress["copied"] += 1
                if time.monotonic() - updated >= 2:
                    atomic_json(runtime / "cache-progress.json", progress)
                    print(json.dumps(progress), flush=True)
                    updated = time.monotonic()
        verify(cache, files)
        progress.update(state="completed", finished_unix=time.time())
        atomic_json(marker, progress)
        atomic_json(runtime / "cache-progress.json", progress)
        return progress
    except BaseException as exc:
        progress.update(state="failed", error_type=type(exc).__name__, finished_unix=time.time())
        atomic_json(runtime / "cache-progress.json", progress)
        raise


def activate(root, cache=CACHE, *, processes_paused=False):
    if not processes_paused:
        raise ValueError("Pause all exact research/auditor process trees before activation")
    root, cache, train, backup, runtime = paths(root, cache)
    files, fingerprint = expected(root)
    base = record(root, cache, train, backup, fingerprint)
    staged = json.loads((runtime / "cache-stage.json").read_text())
    if staged.get("state") != "completed" or any(
        staged.get(key) != value for key, value in base.items()
    ):
        raise ValueError("Local cache does not have a matching completed stage")
    verify(cache, files)
    if train.is_symlink():
        state = json.loads((runtime / "cache-state.json").read_text())
        if (
            train.resolve(strict=False) != cache.resolve(strict=False)
            or state.get("state") != "active"
            or any(state.get(key) != value for key, value in base.items())
        ):
            raise ValueError("Canonical directory is not the recorded managed cache link")
        check_members(backup, files)
        return state
    check_members(train, files)
    if backup.exists() or backup.is_symlink():
        raise FileExistsError("Persistent original backup already exists; refusing overwrite")
    state = {**base, "state": "activating", "activated_unix": time.time()}
    atomic_json(runtime / "cache-state.json", state)
    train.rename(backup)
    try:
        train.symlink_to(cache, target_is_directory=True)
        state["state"] = "active"
        atomic_json(runtime / "cache-state.json", state)
    except BaseException:
        if train.is_symlink() and train.resolve(strict=False) == cache.resolve(strict=False):
            train.unlink()
        if not train.exists() and not train.is_symlink():
            backup.rename(train)
        raise
    return state


def recover(root, cache=CACHE):
    root, cache, train, backup, runtime = paths(root, cache)
    broken_link = train.is_symlink() and not train.exists()
    interrupted_activation = not train.exists() and not train.is_symlink() and backup.exists()
    if not broken_link and not interrupted_activation:
        return {"state": "recovery_not_needed"}
    files, fingerprint = expected(root)
    base = record(root, cache, train, backup, fingerprint)
    state = json.loads((runtime / "cache-state.json").read_text())
    if (
        state.get("state") not in {"active", "activating"}
        or any(state.get(key) != value for key, value in base.items())
        or (broken_link and train.resolve(strict=False) != cache.resolve(strict=False))
    ):
        raise ValueError("Broken symlink is not the recorded managed development cache")
    verify(backup, files)
    if broken_link:
        train.unlink()
    try:
        backup.rename(train)
    except BaseException:
        if broken_link:
            train.symlink_to(cache, target_is_directory=True)
        raise
    state.update(state="recovered_original", recovered_unix=time.time())
    atomic_json(runtime / "cache-state.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["stage", "activate", "recover"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--processes-paused", action="store_true")
    args = parser.parse_args()
    result = (
        activate(args.workspace, processes_paused=args.processes_paused)
        if args.action == "activate"
        else {"stage": stage, "recover": recover}[args.action](args.workspace)
    )
    print(json.dumps(result, indent=2), flush=True)
