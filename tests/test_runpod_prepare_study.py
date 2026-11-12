"""Offline source-promotion and preparation-guard checks; no processes are launched."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "runpod_prepare_study", Path(__file__).parents[1] / "scripts/runpod/prepare_study.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


@pytest.fixture
def upload(tmp_path, monkeypatch):
    body = b"synthetic pinned archive"
    monkeypatch.setattr(prepare, "AD2_BYTES", len(body))
    monkeypatch.setattr(prepare, "AD2_SHA256", hashlib.sha256(body).hexdigest())
    staged = tmp_path / "mvtec_ad_2.uploading.tar.gz"
    staged.write_bytes(body)
    record = tmp_path / "download.json"
    record.write_text(
        json.dumps({"state": "complete", "sha256": prepare.AD2_SHA256, "received_bytes": len(body)})
    )
    return staged, record


def test_bad_upload_hash_never_promotes_or_copies_provenance(tmp_path, upload):
    staged, record = upload
    staged.write_bytes(b"x" * prepare.AD2_BYTES)
    with pytest.raises(ValueError, match="SHA256"):
        prepare.promote_upload(tmp_path, staged, record)
    assert staged.exists()
    assert not (tmp_path / "vision/ad2_downloads/mvtec_ad_2.tar.gz").exists()
    assert not (tmp_path / "vision/ad2_downloads/download.json").exists()


def test_promotion_keeps_provenance_and_refuses_canonical_overwrite(tmp_path, upload, monkeypatch):
    staged, record = upload
    monkeypatch.setattr(prepare, "stop_owned_fetch", lambda root: [])

    def exclusive_link(source, destination):
        # Windows sandbox forbids hard links; model Linux atomic no-overwrite semantics.
        with destination.open("xb") as stream:
            stream.write(source.read_bytes())

    monkeypatch.setattr(prepare.os, "link", exclusive_link)
    canonical = prepare.promote_upload(tmp_path, staged, record)
    assert not staged.exists()
    assert (canonical.parent / "download.json").read_bytes() == record.read_bytes()
    assert (
        json.loads((canonical.parent / "cloud-transfer.json").read_text())["archive_sha256"]
        == prepare.AD2_SHA256
    )
    before = canonical.read_bytes()
    staged.write_bytes(b"replacement")
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare.promote_upload(tmp_path, staged, record)
    assert canonical.read_bytes() == before
    assert staged.read_bytes() == b"replacement"


def test_provenance_requires_strict_complete_marker(tmp_path, upload):
    _, record = upload
    value = json.loads(record.read_text())
    for name, invalid in (("state", "downloading"), ("sha256", "wrong"), ("received_bytes", True)):
        record.write_text(json.dumps({**value, name: invalid}))
        with pytest.raises(ValueError, match="provenance"):
            prepare.provenance(record)


def test_only_recorded_wrapper_fetch_child_can_be_signalled(tmp_path, monkeypatch):
    runtime = tmp_path / "vision/runtime/runpod"
    runtime.mkdir(parents=True)
    (runtime / "ad2-download.pid").write_text("100")
    proc = tmp_path / "proc"
    for pid, parent, command in (
        (201, 100, b"python\0/scripts/fetch_ad2.py\0"),
        (202, 101, b"python\0/scripts/fetch_ad2.py\0"),
        (203, 100, b"python\0/scripts/other.py\0"),
    ):
        directory = proc / str(pid)
        directory.mkdir(parents=True)
        (directory / "stat").write_text(f"{pid} (python worker) S {parent} 0 0")
        (directory / "cmdline").write_bytes(command)
    signalled = []

    def kill(pid, signal):
        signalled.append(pid)
        (proc / str(pid) / "stat").write_text(f"{pid} (python worker) Z 100 0 0")

    monkeypatch.setattr(prepare.os, "kill", kill)
    assert prepare.stop_owned_fetch(tmp_path, proc) == [201]
    assert signalled == [201]
