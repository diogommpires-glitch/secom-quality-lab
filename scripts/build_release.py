"""Build the local demo package from explicit source and frozen-artifact trees."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

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


def collect(tree: Path):
    for path in tree.rglob("*"):
        if (
            path.is_file()
            and not any(part in SKIP for part in path.relative_to(tree).parts)
            and path.suffix not in {".pyc", ".tsbuildinfo"}
        ):
            yield path


def build(root: Path, name: str) -> dict:
    destination = root / "release" / name
    if destination.exists():
        raise FileExistsError(f"Preserve existing release; use another name: {destination}")
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
        entries.extend((path, path.relative_to(root)) for path in collect(root / directory))
    for directory in ("frontend/src", "frontend/tests", "frontend/public"):
        entries.extend((path, path.relative_to(root)) for path in collect(root / directory))
    entries.extend(
        (path, path.relative_to(root))
        for path in (root / "frontend").iterdir()
        if path.is_file() and path.suffix != ".tsbuildinfo"
    )
    for directory in (
        "workspace/bundles",
        "workspace/reports",
        "workspace/research",
        "evidence/verification",
        "evidence/release_install",
    ):
        entries.extend((path, path.relative_to(root)) for path in collect(root / directory))
    verification = root / "evidence/release_verification.json"
    if verification.is_file():
        entries.append((verification, verification.relative_to(root)))
    entries.extend(
        (path, path.relative_to(root))
        for path in (root / "evidence/demo").iterdir()
        if path.is_file()
    )
    for relative in (
        "evidence/protocol_v1/source_snapshot",
        "evidence/protocol_v1/workspace/research",
    ):
        entries.extend((path, path.relative_to(root)) for path in collect(root / relative))
    for path in (root / "release/wheels").glob("*.whl"):
        entries.append((path, Path("wheels") / path.name))
    checksums = {}
    for source, relative in sorted(set(entries), key=lambda item: str(item[1])):
        if "data/raw" in relative.as_posix() or "runtime" in relative.parts:
            raise ValueError(
                f"Raw holdout measurements or mutable state cannot enter demo release: {relative}"
            )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        checksums[relative.as_posix()] = sha256_file(target)
    write_json(destination / "MANIFEST.sha256.json", checksums)
    archive = root / "release" / f"{destination.name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                zipped.write(path, Path(destination.name) / path.relative_to(destination))
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
    write_json(root / "release" / f"{destination.name}.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="secom_quality_lab-1.0.0")
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1], args.name), indent=2))
