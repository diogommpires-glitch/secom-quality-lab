"""Pod-scoped CLI compatibility without provider calls or credentials."""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).parents[1] / "scripts/runpod/watchdog.py"
SPEC = importlib.util.spec_from_file_location("runpod_watchdog_cli", PATH)
assert SPEC and SPEC.loader
watchdog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watchdog)


def test_modern_stop_success_does_not_invoke_legacy_command():
    calls = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(returncode=0)

    watchdog.stop_compute("pod_123-abc", run=run)
    assert calls == [
        (
            ["runpodctl", "pod", "stop", "pod_123-abc"],
            {"capture_output": True, "text": True, "timeout": 60},
        )
    ]


def test_modern_command_failure_falls_back_once_to_legacy_stop():
    calls = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(returncode=2 if len(calls) == 1 else 0)

    watchdog.stop_compute("pod123", run=run)
    assert [arguments for arguments, _ in calls] == [
        ["runpodctl", "pod", "stop", "pod123"],
        ["runpodctl", "stop", "pod", "pod123"],
    ]
    assert all(
        options == {"capture_output": True, "text": True, "timeout": 60} for _, options in calls
    )


def test_both_cli_commands_fail_without_exposing_captured_output(capsys):
    calls = []
    secret = "SENTINEL_POD_SCOPED_KEY_NEVER_PRINT"

    def run(arguments, **kwargs):
        calls.append(arguments)
        return SimpleNamespace(returncode=1, stdout=secret, stderr=secret)

    with pytest.raises(RuntimeError, match="provider output is withheld") as error:
        watchdog.stop_compute("pod123", run=run)
    assert len(calls) == 2
    assert secret not in str(error.value)
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err


@pytest.mark.parametrize("pod_id", ["", "pod;bad", "pod id", "../pod", "pod\n", None, 123])
def test_invalid_pod_id_never_invokes_cli(pod_id):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid Pod ID reached the CLI")

    with pytest.raises(ValueError, match="Invalid Pod ID"):
        watchdog.stop_compute(pod_id, run=forbidden)


@pytest.mark.parametrize(
    "failure", [OSError("CLI unavailable"), subprocess.TimeoutExpired("runpodctl", 60)]
)
def test_invocation_or_timeout_failure_remains_available_for_monitor_retry(failure):
    calls = []

    def run(arguments, **kwargs):
        calls.append(arguments)
        raise failure

    with pytest.raises(type(failure)):
        watchdog.stop_compute("pod123", run=run)
    assert calls == [["runpodctl", "pod", "stop", "pod123"]]
