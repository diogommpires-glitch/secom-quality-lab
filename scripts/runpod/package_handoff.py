"""Create a reviewable source-only handoff; never upload local environments or secrets."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect_files(root):
    root = Path(root).resolve()
    files = []
    for directory in ("src", "configs", "scripts", "docs"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            if any(
                part.startswith(".") or part in {"__pycache__", "node_modules"}
                for part in relative.parts
            ):
                continue
            allowed = {
                "src": {".py", ".json", ".txt", ".js", ".css", ".html", ".svg"},
                "configs": {".json", ".toml", ".in", ".lock"},
                "scripts": {".py", ".ps1", ".sh"},
                "docs": {".md"},
            }
            if path.suffix.lower() not in allowed[directory] and path.name != "LICENSE":
                continue
            if path.suffix.lower() in {
                ".pyc",
                ".pem",
                ".key",
                ".env",
                ".parquet",
                ".skops",
                ".safetensors",
            }:
                continue
            sensitive = any(
                word in path.name.lower()
                for word in (
                    "secret",
                    "token",
                    "password",
                    "credential",
                    "api-key",
                    "api_key",
                    "private-key",
                    "private_key",
                )
            )
            if "id_ed25519" in path.name or (sensitive and path.suffix != ".ps1"):
                continue
            files.append(path)
    files += [
        root / name
        for name in ("pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md")
        if (root / name).is_file()
    ]
    return sorted(set(files))


def build_handoff(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    members = {path.relative_to(root).as_posix(): path.read_bytes() for path in collect_files(root)}
    explicit = {
        "migration/windows_protocol.json": root / "workspace/vision/research/protocol.json",
        "migration/windows_stop.json": root / "workspace/vision/runtime/local-stop.json",
    }
    for name, path in explicit.items():
        if path.exists():
            members[name] = path.read_bytes()
    pins = {
        "KolektorSDD2": "edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8",
        "MVTecAD2": "c0ded99ef32bfc8e352d52beb44515e5b292b8598cb963aadfa91ca0763505e4",
    }
    members["migration/archive-pins.json"] = json.dumps(pins, sort_keys=True).encode()
    development = root / "workspace/vision/data/train.parquet"
    if development.exists():
        import pandas as pd

        identities = (
            pd.read_parquet(development)[["record_id", "fold"]]
            .sort_values("record_id")
            .to_dict("records")
        )
        encoded = json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
        members["migration/development-partitions.json"] = json.dumps(
            {"row_count": len(identities), "sha256": hashlib.sha256(encoded).hexdigest()}
        ).encode()
    manifest = {
        "schema": "secom-runpod-source-v1",
        "files": {name: hashlib.sha256(body).hexdigest() for name, body in members.items()},
        "contains_credentials": False,
        "contains_test_images": False,
        "study_policy": "Fresh Linux paths, environment and protocol; preserve Windows14fits separately. No pooled selection.",
        "lock_overlay": "Bootstrap copies Linuxrunpod-requirements.lock to canonical vision-requirements.lock inside the cloud copy only, before freeze.",
    }
    members["handoff-manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, body)
    return {
        "archive": str(output),
        "sha256": sha(output),
        "files": len(members),
        "bytes": output.stat().st_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output", type=Path, default=Path("release/runpod/secom_runpod_source.zip")
    )
    args = parser.parse_args()
    print(json.dumps(build_handoff(args.root, args.output), indent=2))
