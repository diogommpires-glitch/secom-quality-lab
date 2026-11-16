"""Portable imports and filesystem publication contracts."""

import ctypes
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from secom_quality.common import read_json, write_json


def _publish_record(path, value, barrier, outcomes, publication_module):
    from importlib import import_module

    immutable_json = import_module(publication_module).immutable_json

    barrier.wait(timeout=20)
    try:
        immutable_json(Path(path), {"value": value, "padding": "x" * 200_000})
    except (ValueError, OSError) as error:
        outcomes.put((value, type(error).__name__))
    else:
        outcomes.put((value, "published"))


def test_imports_do_not_load_a_machine_specific_storage_adapter(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(source)!r}); "
        "from secom_quality.common import read_json, sha256_file; "
        "import secom_quality.api; import secom_quality.cli; "
        "assert '_dodo_ad2_runtime_binding' not in sys.modules; "
        "p=Path('record.json'); p.write_text('{\"value\": 1}'); "
        "assert read_json(p)=={'value':1}; assert len(sha256_file(p))==64"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=tmp_path,
        env={**os.environ, "SECOM_AD2_STORAGE_ADAPTER": str(tmp_path / "absent.py")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("publication_module", ["secom_quality.data", "secom_quality.vision.data"])
def test_concurrent_scientific_publication_keeps_exactly_one_payload(tmp_path, publication_module):
    context = multiprocessing.get_context("spawn")
    barrier, outcomes = context.Barrier(2), context.Queue()
    path = tmp_path / "record.json"
    processes = [
        context.Process(
            target=_publish_record, args=(str(path), value, barrier, outcomes, publication_module)
        )
        for value in (1, 2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    observed = [outcomes.get(timeout=5) for _ in processes]
    assert sorted(status for _, status in observed) == ["ValueError", "published"]
    winner = next(value for value, status in observed if status == "published")
    assert read_json(path) == {"value": winner, "padding": "x" * 200_000}
    assert [entry.name for entry in tmp_path.iterdir()] == ["record.json"]


def test_mutable_publishers_do_not_share_a_temporary_file(tmp_path):
    path = tmp_path / "status.json"
    barrier = threading.Barrier(2)
    errors = []

    def publish(value):
        barrier.wait(timeout=5)
        try:
            write_json(path, {"value": value, "padding": "x" * 200_000})
        except OSError as error:
            errors.append(error)

    writers = [threading.Thread(target=publish, args=(value,)) for value in (1, 2)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=5)
        assert not writer.is_alive()
    assert errors == []
    assert read_json(path) in [{"value": value, "padding": "x" * 200_000} for value in (1, 2)]
    assert [entry.name for entry in tmp_path.iterdir()] == ["status.json"]


@pytest.mark.skipif(os.name != "nt", reason="Native Windows delete-sharing boundary")
@pytest.mark.parametrize("release_reader", [True, False])
def test_windows_reader_retry_preserves_complete_json(tmp_path, release_reader):
    from ctypes import wintypes

    path = tmp_path / "status.json"
    write_json(path, {"old": True})
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
    assert handle != wintypes.HANDLE(-1).value
    closed = threading.Event()

    def release():
        time.sleep(0.08)
        assert kernel.CloseHandle(handle)
        closed.set()

    reader = threading.Thread(target=release) if release_reader else None
    if reader:
        reader.start()
    try:
        if release_reader:
            write_json(path, {"new": True})
            assert read_json(path) == {"new": True}
        else:
            with pytest.raises(PermissionError) as failure:
                write_json(path, {"new": True})
            assert failure.value.winerror in {5, 32}
            assert read_json(path) == {"old": True}
    finally:
        if reader:
            reader.join(timeout=5)
        if not closed.is_set():
            assert kernel.CloseHandle(handle)
    assert [entry.name for entry in tmp_path.iterdir()] == ["status.json"]
