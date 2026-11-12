"""Start the local application and browser, or stop its exact recorded server."""

import argparse
import errno
import json
import msvcrt
import socket
import subprocess
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from contextlib import contextmanager
from pathlib import Path

import psutil

PROJECT = Path(__file__).resolve().parent.parent
RUNTIME = PROJECT / "workspace" / "desktop-app"
IDENTITY = RUNTIME / "server.json"
PORT = 8767
URL = f"http://127.0.0.1:{PORT}"


def recorded_server():
    if not IDENTITY.exists():
        return None
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    try:
        process = psutil.Process(identity["pid"])
        command = process.cmdline()
        if (
            abs(process.create_time() - identity["created"]) > 0.001
            or process.exe() != identity["executable"]
            or command != identity["command"]
            or str(Path(__file__).resolve()) not in command
            or command[-1] != "serve"
        ):
            return None
        return process
    except psutil.NoSuchProcess:
        return None


def read_api(path, timeout=2):
    with urllib.request.urlopen(URL + path, timeout=timeout) as response:
        return json.load(response)


@contextmanager
def startup_lock():
    # Start and stop share this lock, including the child's first identity write.
    with (RUNTIME / "startup.lock").open("a+b") as lock:
        if lock.seek(0, 2) == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        deadline = time.monotonic() + 120
        while True:
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as error:
                if error.errno != errno.EACCES:
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Another app startup or shutdown is still in progress."
                    ) from None
                time.sleep(0.25)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def start(open_browser):
    with startup_lock():
        process = recorded_server()
        child = None
        if process is None:
            with socket.socket() as connection:
                connection.settimeout(0.5)
                if connection.connect_ex(("127.0.0.1", PORT)) == 0:
                    raise RuntimeError(f"Port {PORT} belongs to another server. Close it first.")
            python = PROJECT / ".venv-vision" / "Scripts" / "python.exe"
            with (RUNTIME / "server.log").open("ab") as log:
                child = subprocess.Popen(
                    [str(python), "-B", str(Path(__file__).resolve()), "serve"],
                    cwd=PROJECT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
        deadline = time.monotonic() + 60
        while True:
            if child is not None and child.poll() is not None:
                raise RuntimeError("The application exited during startup. See server.log.")
            try:
                ready = read_api("/health/ready").get("status") == "ready"
            except (urllib.error.URLError, TimeoutError):
                ready = False
            if ready and recorded_server() is not None:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "The application was not ready within 60 seconds. See server.log."
                )
            time.sleep(0.25)
        if not read_api("/api/v2/vision/status", timeout=45).get("ready"):
            raise RuntimeError("The vision bundle is unavailable. See server.log.")
        process = recorded_server()
        if process is None:
            raise RuntimeError("The application exited before startup completed. See server.log.")
        server_pid = process.pid
    if open_browser and not webbrowser.open(URL):
        raise RuntimeError(f"The application is ready. Open {URL} in your browser.")
    print(json.dumps({"state": "ready", "url": URL, "pid": server_pid}))


def serve():
    process = psutil.Process()
    identity = {
        "pid": process.pid,
        "created": process.create_time(),
        "executable": process.exe(),
        "command": process.cmdline(),
    }
    temporary = IDENTITY.with_suffix(".tmp")
    temporary.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    temporary.replace(IDENTITY)
    import uvicorn

    from secom_quality.api import create_app

    uvicorn.run(create_app(PROJECT / "workspace"), host="127.0.0.1", port=PORT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "serve", "stop"], default="start", nargs="?")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    try:
        if args.action == "start":
            start(not args.no_browser)
        elif args.action == "serve":
            serve()
        else:
            with startup_lock():
                process = recorded_server()
                if process is not None:
                    process.terminate()
                    process.wait(timeout=10)
            print(json.dumps({"state": "stopped"}))
    except Exception as error:
        with (RUNTIME / "launcher.log").open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        if not args.no_browser and args.action != "serve":
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                0, f"{error}\n\nLogs: {RUNTIME}", "SECOM Quality Lab", 0x10
            )
        raise


if __name__ == "__main__":
    main()
