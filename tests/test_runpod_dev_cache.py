"""Operational cache byte identity and recovery; synthetic development files only."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "runpod_dev_cache", Path(__file__).parents[1] / "scripts/runpod/dev_cache.py"
)
assert SPEC and SPEC.loader
cache_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache_module)


@pytest.fixture
def example(tmp_path, monkeypatch):
    root = tmp_path / "study"
    train = root / "vision/images/train"
    train.mkdir(parents=True)
    payloads = {
        "001.png": b"normal-image",
        "001_GT.png": b"normal-mask",
        "002.png": b"defect-image",
        "002_GT.png": b"defect-mask",
    }
    members = dict(cache_module.EXCLUDED)
    for name, payload in payloads.items():
        (train / name).write_bytes(payload)
        members["train/" + name] = hashlib.sha256(payload).hexdigest()
    protected = root / "vision/protected/images"
    protected.mkdir(parents=True)
    (protected / "never-open.png").write_bytes(b"protected-content")
    members["test/never-open.png"] = hashlib.sha256(b"protected-content").hexdigest()
    manifest = {
        "archive_sha256": cache_module.ARCHIVE_SHA256,
        "member_sha256": members,
        "dataset_version_review": {"excluded_exact_copy_files": cache_module.EXCLUDED},
    }
    manifest["source_fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_dir = root / "vision/source"
    source_dir.mkdir()
    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    cache = tmp_path / "local-cache/train"
    monkeypatch.setattr(cache_module, "CACHE", cache)
    monkeypatch.setattr(cache_module, "EXPECTED_FILES", len(payloads))
    return root, train, cache, payloads, manifest_path, protected


def require_symlinks(tmp_path):
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(tmp_path / "absent")
        probe.unlink()
    except OSError:
        pytest.skip("This Windows account cannot create symlinks")


def test_stage_preserves_canonical_bytes_and_never_opens_protected(example, monkeypatch):
    root, train, cache, payloads, manifest_path, protected = example
    prior_manifest = manifest_path.read_bytes()
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert not path.is_relative_to(protected), "Protected pixels must never enter the cache"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = cache_module.stage(root, cache)
    assert result["state"] == "completed" and result["copied"] == len(payloads)
    assert {p.name for p in cache.iterdir()} == set(payloads)
    for name, payload in payloads.items():
        assert (train / name).read_bytes() == (cache / name).read_bytes() == payload
    assert manifest_path.read_bytes() == prior_manifest
    assert not train.is_symlink() and not train.with_name("train.network-original").exists()
    assert cache_module.stage(root, cache) == result


def test_changed_source_pin_fails_without_activation_or_original_overwrite(example):
    root, train, cache, _, _, _ = example
    (train / "001.png").write_bytes(b"corrupt-source")
    with pytest.raises(ValueError, match="checksum differs"):
        cache_module.stage(root, cache)
    assert (train / "001.png").read_bytes() == b"corrupt-source"
    assert not train.is_symlink() and not train.with_name("train.network-original").exists()
    progress = json.loads((root / "vision/runtime/runpod/cache-progress.json").read_text())
    assert progress["state"] == "failed"
    assert not (root / "vision/runtime/runpod/cache-stage.json").exists()


def test_unknown_members_and_partial_cache_are_preserved(example):
    root, train, cache, _, _, _ = example
    (train / "unknown.png").write_bytes(b"unknown")
    with pytest.raises(ValueError, match="unexpected members"):
        cache_module.stage(root, cache)
    assert not cache.exists()
    (train / "unknown.png").unlink()
    cache.mkdir(parents=True)
    (cache / "partial.png").write_bytes(b"partial")
    with pytest.raises(FileExistsError, match="preserve and inspect"):
        cache_module.stage(root, cache)
    assert (cache / "partial.png").read_bytes() == b"partial"


def test_manifest_corruption_and_unreviewed_cache_path_are_rejected(example):
    root, _, cache, _, manifest_path, _ = example
    with pytest.raises(ValueError, match="reviewed"):
        cache_module.stage(root, cache.with_name("unknown"))
    value = json.loads(manifest_path.read_text())
    value["member_sha256"]["train/001.png"] = "0" * 64
    manifest_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest checksum"):
        cache_module.stage(root, cache)
    assert not cache.exists()


def test_activate_requires_pause_and_verified_completed_cache(example):
    root, _, cache, _, _, _ = example
    with pytest.raises(ValueError, match="Pause all exact"):
        cache_module.activate(root, cache)
    cache_module.stage(root, cache)
    (cache / "002.png").write_bytes(b"corrupt-cache")
    with pytest.raises(ValueError, match="checksum differs"):
        cache_module.activate(root, cache, processes_paused=True)


def test_activation_keeps_manifest_paths_and_recovery_restores_original(example, tmp_path):
    require_symlinks(tmp_path)
    root, train, cache, payloads, manifest_path, _ = example
    prior_manifest = manifest_path.read_bytes()
    cache_module.stage(root, cache)
    state = cache_module.activate(root, cache, processes_paused=True)
    assert state["state"] == "active" and train.is_symlink()
    allowed = train.resolve()
    assert all((train / name).resolve().is_relative_to(allowed) for name in payloads)
    assert cache_module.activate(root, cache, processes_paused=True) == state
    assert manifest_path.read_bytes() == prior_manifest
    backup = train.with_name("train.network-original")
    assert not backup.is_symlink()
    for name, payload in payloads.items():
        assert (backup / name).read_bytes() == (train / name).read_bytes() == payload
        (cache / name).unlink()
    cache.rmdir()
    recovered = cache_module.recover(root, cache)
    assert recovered["state"] == "recovered_original"
    assert not train.is_symlink() and not backup.exists()
    assert manifest_path.read_bytes() == prior_manifest
    assert all((train / name).read_bytes() == payload for name, payload in payloads.items())


def test_activation_rolls_back_if_symlink_creation_fails(example, monkeypatch):
    root, train, cache, payloads, _, _ = example
    cache_module.stage(root, cache)

    def fail(*args, **kwargs):
        raise OSError("simulated symlink failure")

    monkeypatch.setattr(Path, "symlink_to", fail)
    with pytest.raises(OSError, match="simulated"):
        cache_module.activate(root, cache, processes_paused=True)
    assert train.is_dir() and not train.is_symlink()
    assert not train.with_name("train.network-original").exists()
    assert all((train / name).read_bytes() == payload for name, payload in payloads.items())


def test_recovery_refuses_unknown_broken_symlink(example, tmp_path):
    require_symlinks(tmp_path)
    root, train, cache, _, _, _ = example
    cache_module.stage(root, cache)
    cache_module.activate(root, cache, processes_paused=True)
    train.unlink()
    unknown = tmp_path / "unknown-missing"
    train.symlink_to(unknown)
    with pytest.raises(ValueError, match="not the recorded"):
        cache_module.recover(root, cache)
    assert train.is_symlink() and train.resolve(strict=False) == unknown.resolve(strict=False)
    assert train.with_name("train.network-original").is_dir()


def test_recovery_restores_interrupted_activation_without_canonical_link(example):
    root, train, cache, payloads, _, _ = example
    cache_module.stage(root, cache)
    _, _, _, backup, runtime = cache_module.paths(root, cache)
    _, fingerprint = cache_module.expected(root)
    state = cache_module.record(root, cache, train, backup, fingerprint)
    state["state"] = "activating"
    cache_module.atomic_json(runtime / "cache-state.json", state)
    train.rename(backup)
    assert not train.exists()
    recovered = cache_module.recover(root, cache)
    assert recovered["state"] == "recovered_original"
    assert not train.is_symlink() and not backup.exists()
    assert all((train / name).read_bytes() == payload for name, payload in payloads.items())
