"""Exercise the launcher's source-only preparation gate without starting research."""

import json
import os
import sys
from pathlib import Path

import pytest

LAUNCH = Path(__file__).parents[1] / "scripts/runpod/launch.sh"
SCRIPT = (
    LAUNCH.read_text(encoding="utf-8")
    .split("<<'AD2_PREPARATION_CHECK'\n", 1)[1]
    .split("\nAD2_PREPARATION_CHECK", 1)[0]
)
SHA256 = "c0ded99ef32bfc8e352d52beb44515e5b292b8598cb963aadfa91ca0763505e4"


def marker():
    return {
        "state": "preparing",
        "archive_sha256": SHA256,
        "scope": "development_source_audit_only",
        "pid": 123,
    }


def check(tmp_path, monkeypatch, preparation=None, *, enabled="1", error=None):
    if enabled is None:
        monkeypatch.delenv("SECOM_AD2_BACKGROUND_PREPARATION", raising=False)
    else:
        monkeypatch.setenv("SECOM_AD2_BACKGROUND_PREPARATION", enabled)
    monkeypatch.setattr(sys, "argv", ["-", str(tmp_path)])
    calls = []

    def kill(pid, signal):
        calls.append((pid, signal))
        if error is not None:
            raise error

    monkeypatch.setattr(os, "kill", kill)
    if preparation is not None:
        path = tmp_path / "vision/runtime/runpod/ad2-preparation.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            preparation if isinstance(preparation, str) else json.dumps(preparation),
            encoding="utf-8",
        )
    exec(compile(SCRIPT, str(LAUNCH), "exec"), {"__name__": "__main__"})
    return calls


def test_valid_pinned_source_only_preparation_permits_launch_without_skipping_ad2(
    tmp_path, monkeypatch, capsys
):
    assert check(tmp_path, monkeypatch, marker()) == [(123, 0)]
    assert "AD2 challenge remains enabled" in capsys.readouterr().out
    assert "skip-ad2" not in SCRIPT


@pytest.mark.parametrize("enabled", [None, "0", "true"])
def test_default_gate_rejects_missing_audited_source_even_with_live_marker(
    tmp_path, monkeypatch, enabled
):
    with pytest.raises(SystemExit, match="Acquire and audit AD2 first"):
        check(tmp_path, monkeypatch, marker(), enabled=enabled)


@pytest.mark.parametrize("preparation", [None, "{malformed", [], 123])
def test_missing_or_malformed_marker_is_rejected(tmp_path, monkeypatch, preparation):
    with pytest.raises(SystemExit, match=r"preparation marker|preparation needs"):
        check(tmp_path, monkeypatch, preparation)


@pytest.mark.parametrize(
    "changes",
    [
        {"archive_sha256": "c0de"},
        {"archive_sha256": "0" * 64},
        {"state": "completed"},
        {"state": "failed"},
        {"scope": "development_fit"},
        {"scope": "final_test"},
        {"pid": 0},
        {"pid": -1},
        {"pid": True},
        {"pid": "123"},
        {"pid": 1.0},
        {"pid": None},
    ],
)
def test_unpinned_nonpreparing_or_invalid_process_marker_is_rejected_before_liveness_check(
    tmp_path, monkeypatch, changes
):
    with pytest.raises(SystemExit, match="pinned source-only audit"):
        check(
            tmp_path, monkeypatch, marker() | changes, error=AssertionError("Liveness must not run")
        )


@pytest.mark.parametrize("error", [ProcessLookupError("dead"), PermissionError("inaccessible")])
def test_dead_or_unverifiable_auditor_process_is_rejected(tmp_path, monkeypatch, error):
    with pytest.raises(SystemExit, match="not verifiably alive"):
        check(tmp_path, monkeypatch, marker(), error=error)
