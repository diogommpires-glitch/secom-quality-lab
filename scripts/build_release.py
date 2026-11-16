"""Build the local demo package from explicit source and frozen-artifact trees."""

import argparse
import json
import os
import re
import shutil
import stat
import zipfile
from pathlib import Path, PureWindowsPath

from secom_quality.common import sha256_file, utc_now, write_json

SKIP = {
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "test-results",
    "playwright-report",
    ".venv",
    ".bootstrap",
}
FRONTEND_FILES = {
    ".prettierignore",
    ".prettierrc.json",
    "index.html",
    "openapi.json",
    "package.json",
    "playwright.config.ts",
    "playwright.vision.config.ts",
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
    "tsconfig.json",
    "vite.config.ts",
}
PRIVATE_NAMES = {
    ".npmrc",
    ".pypirc",
    ".netrc",
    "_netrc",
    ".git-credentials",
    ".ssh",
    ".aws",
    ".gnupg",
    "credentials.json",
    "secrets.json",
    "runpod.json",
    "runpod-api-key.json",
    "runpod_api_key.json",
    "api-key.json",
    "api_key.json",
    "api-keys.json",
    "api_keys.json",
    "account-key.json",
    "account_key.json",
    "account-token.json",
    "account_token.json",
    "access-token.json",
    "access_token.json",
    "token.json",
    "tokens.json",
}
PRIVATE_SUFFIXES = {".env", ".pem", ".key", ".p12", ".pfx", ".ppk", ".jks", ".keystore"}


def private_input(relative: Path):
    for part in relative.parts:
        name = part.casefold()
        if (
            name == ".env"
            or name.startswith(".env.")
            or name.startswith(("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"))
            or name in PRIVATE_NAMES
            or Path(name).suffix in PRIVATE_SUFFIXES
        ):
            return True
    return False


def checked_source(root: Path, path: Path):
    relative = path.relative_to(root)
    for component in (relative, *relative.parents):
        candidate = root / component
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
        ):
            raise ValueError(f"Linked release input is forbidden: {relative}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Release input escapes the source root: {relative}")
    return path


def collect(tree: Path, root: Path):
    if not tree.exists() and not tree.is_symlink():
        return
    checked_source(root, tree)
    for path in tree.rglob("*"):
        if any(part in SKIP for part in path.relative_to(tree).parts):
            continue
        checked_source(root, path)
        if path.is_file() and path.suffix not in {".pyc", ".tsbuildinfo"}:
            yield path


def build(root: Path, name: str) -> dict:
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
        or name.endswith(".")
        or PureWindowsPath(name).is_reserved()
    ):
        raise ValueError("Release name must be one ordinary directory component")
    root = root.resolve()
    destination = root / "release" / name
    archive = root / "release" / f"{name}.zip"
    metadata = root / "release" / f"{name}.json"
    if (root / "release").exists() or (root / "release").is_symlink():
        checked_source(root, root / "release")
    for output in (destination, archive, metadata):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Preserve existing release; use another name: {output}")
    destination.mkdir(parents=True)
    entries = []
    for filename in (
        "README.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".gitignore",
        ".gitattributes",
        "AGENTS.md",
    ):
        entries.append((root / filename, Path(filename)))
    for directory in ("src", "tests", "scripts", "docs", "reports", "configs", ".github"):
        entries.extend((path, path.relative_to(root)) for path in collect(root / directory, root))
    for directory in ("frontend/src", "frontend/tests", "frontend/public"):
        entries.extend((path, path.relative_to(root)) for path in collect(root / directory, root))
    entries.extend(
        (path, path.relative_to(root))
        for filename in sorted(FRONTEND_FILES)
        if (path := root / "frontend" / filename).is_file()
    )
    for directory in (
        "workspace/bundles",
        "workspace/reports",
        "workspace/research",
        "evidence/verification",
        "evidence/release_install",
    ):
        entries.extend((path, path.relative_to(root)) for path in collect(root / directory, root))
    verification = root / "evidence/release_verification.json"
    if verification.is_file():
        entries.append((verification, verification.relative_to(root)))
    entries.extend(
        (path, path.relative_to(root))
        for path in (root / "evidence/demo").iterdir()
        if path.is_file()
    )
    for relative_tree in (
        "evidence/protocol_v1/source_snapshot",
        "evidence/protocol_v1/workspace/research",
    ):
        entries.extend(
            (path, path.relative_to(root)) for path in collect(root / relative_tree, root)
        )
    for path in (root / "release/wheels").glob("*.whl"):
        entries.append((path, Path("wheels") / path.name))
    checksums = {}
    for source, relative in sorted(set(entries), key=lambda item: str(item[1])):
        if private_input(relative):
            continue
        checked_source(root, source)
        if "data/raw" in relative.as_posix() or "runtime" in relative.parts:
            raise ValueError(
                f"Raw holdout measurements or mutable state cannot enter demo release: {relative}"
            )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        checksums[relative.as_posix()] = sha256_file(target)
    write_json(destination / "MANIFEST.sha256.json", checksums)
    with archive.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for path in sorted(destination.rglob("*")):
                if path.is_file():
                    zipped.write(path, Path(destination.name) / path.relative_to(destination))
        stream.flush()
        os.fsync(stream.fileno())
    result = {
        "created_at": utc_now(),
        "directory": str(destination),
        "archive": str(archive),
        "sha256": sha256_file(archive),
        "bytes": archive.stat().st_size,
        "files": len(checksums),
        "raw_archive_included": False,
        "notes": "Offline demo contains development-only replay measurements, verified models, frozen aggregate results and offline research predictions; no raw final-test measurements or runtime database.",
    }
    with metadata.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="secom_quality_lab-1.0.0")
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1], args.name), indent=2))
