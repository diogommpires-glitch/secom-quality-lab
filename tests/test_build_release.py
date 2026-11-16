"""Release input boundaries use synthetic public files and secret sentinels only."""

import importlib.util
import os
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "build_release", Path(__file__).parents[1] / "scripts/build_release.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    for name in (
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
        (root / name).write_text("public fixture")
    for name in ("frontend", "evidence/demo"):
        (root / name).mkdir(parents=True)
    (root / "frontend/package.json").write_text('{"name":"public-fixture"}')
    return root


def test_release_frontend_allowlist_excludes_private_configuration(source):
    (source / "frontend/.env.local").write_text("SYNTHETIC_SECRET_DO_NOT_PACKAGE")
    (source / "frontend/.npmrc").write_text("SYNTHETIC_TOKEN_DO_NOT_PACKAGE")
    (source / "frontend/private-notes.txt").write_text("synthetic private note")
    result = release.build(source, "public-demo-1.0")
    with zipfile.ZipFile(result["archive"]) as archive:
        names = set(archive.namelist())
        assert "public-demo-1.0/frontend/package.json" in names
        assert not any(
            ".env" in name or ".npmrc" in name or "private-notes" in name for name in names
        )
        assert not any(b"SYNTHETIC_SECRET" in archive.read(name) for name in names)


@pytest.mark.parametrize(
    "name", ["../escape", "..\\escape", "/absolute", "C:\\absolute", ".", "..", "CON", "trailing."]
)
def test_release_name_cannot_escape_destination(source, name):
    with pytest.raises(ValueError, match="ordinary directory component"):
        release.build(source, name)
    assert not (source / "release").exists()


@pytest.mark.parametrize(
    "kind", ["explicit_file", "recursive_file", "recursive_tree", "explicit_parent"]
)
def test_linked_release_inputs_and_linked_ancestors_are_rejected(source, tmp_path, kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("SYNTHETIC_LINK_SECRET")
    if kind == "explicit_file":
        link, target, directory = source / "README.md", outside / "private.txt", False
        link.unlink()
    elif kind == "recursive_file":
        (source / "src").mkdir()
        link, target, directory = source / "src/module.py", outside / "private.txt", False
    elif kind == "recursive_tree":
        link, target, directory = source / "src", outside, True
    else:
        (outside / "package.json").write_text("synthetic private config")
        (source / "frontend/package.json").unlink()
        (source / "frontend").rmdir()
        link, target, directory = source / "frontend", outside, True
    link.symlink_to(target, target_is_directory=directory)
    with pytest.raises(ValueError, match="Linked release input"):
        release.build(source, "linked-demo")
    assert not (source / "release/linked-demo.zip").exists()


def test_private_filename_families_are_excluded_from_every_release_tree(source):
    private = [
        "frontend/src/.env.local",
        "configs/private.pem",
        "evidence/demo/.npmrc",
        "docs/.ENV.PRODUCTION",
        "reports/PRIVATE.KEY",
        "scripts/.NPMRC",
        "workspace/research/credentials.json",
        "evidence/verification/id_ed25519",
        "configs/nested/RUNPOD-API-KEY.JSON",
        "configs/api-key.json",
        "configs/API_KEY.JSON",
        "configs/runpod.json",
        "reports/account-token.json",
        "reports/TOKENS.JSON",
        "evidence/demo/.git-credentials",
        "docs/_NETRC",
    ]
    for name in private:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SYNTHETIC_PRIVATE_SENTINEL")
    result = release.build(source, "private-families")
    with zipfile.ZipFile(result["archive"]) as archive:
        assert not any(
            b"SYNTHETIC_PRIVATE_SENTINEL" in archive.read(name) for name in archive.namelist()
        )
        assert "private-families/frontend/package.json" in archive.namelist()


def test_credential_filename_filter_preserves_public_code_and_security_workflows(source):
    for name in ("scripts/runpod/credentials.ps1", ".github/workflows/secret-scan.yml"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("public acquisition code or workflow; no credentials")
    result = release.build(source, "public-security-code")
    with zipfile.ZipFile(result["archive"]) as archive:
        assert "public-security-code/scripts/runpod/credentials.ps1" in archive.namelist()
        assert "public-security-code/.github/workflows/secret-scan.yml" in archive.namelist()


@pytest.mark.parametrize("kind", ["archive_hardlink", "archive_symlink", "metadata", "destination"])
def test_existing_outputs_are_preserved_before_build(source, tmp_path, kind):
    directory = source / "release"
    directory.mkdir()
    original = tmp_path / "original.txt"
    original.write_text("preserve existing bytes")
    if kind == "archive_hardlink":
        os.link(original, directory / "preserved.zip")
    elif kind == "archive_symlink":
        (directory / "preserved.zip").symlink_to(original)
    elif kind == "metadata":
        (directory / "preserved.json").write_bytes(original.read_bytes())
    else:
        (directory / "preserved").mkdir()
    with pytest.raises(FileExistsError, match="Preserve existing release"):
        release.build(source, "preserved")
    assert original.read_text() == "preserve existing bytes"
    if kind != "destination":
        assert not (directory / "preserved").exists()
    if kind == "metadata":
        assert (directory / "preserved.json").read_bytes() == original.read_bytes()


@pytest.mark.parametrize("suffix", ["zip", "json"])
def test_output_creation_race_cannot_overwrite_another_file(source, tmp_path, monkeypatch, suffix):
    original = tmp_path / "outside.txt"
    original.write_text("preserve racing output")
    copyfile = release.shutil.copyfile
    planted = False

    def plant_link_then_copy(source_file, target):
        nonlocal planted
        if not planted:
            os.link(original, source / "release" / f"racing.{suffix}")
            planted = True
        return copyfile(source_file, target)

    monkeypatch.setattr(release.shutil, "copyfile", plant_link_then_copy)
    with pytest.raises(FileExistsError):
        release.build(source, "racing")
    assert planted and original.read_text() == "preserve racing output"
