"""Creation-bound Linux process ownership for the Pod coordinator and guard."""

import json
from pathlib import Path


def process(pid, proc=Path("/proc")):
    try:
        if type(pid) is not int or pid <= 1:
            return None
        directory = proc / str(pid)
        fields = (directory / "stat").read_text().rpartition(") ")[2].split()
        command = [
            part.decode(errors="replace")
            for part in (directory / "cmdline").read_bytes().split(b"\0")
            if part
        ]
        return {
            "pid": pid,
            "start_ticks": int(fields[19]),
            "state": fields[0],
            "command": command,
        }
    except (OSError, ValueError, IndexError):
        return None


def matches(identity, current):
    return bool(
        isinstance(identity, dict)
        and current
        and current["state"] not in {"Z", "X"}
        and all(identity.get(key) == current[key] for key in ("pid", "start_ticks", "command"))
    )


def option(command, flag):
    positions = [
        index for index, part in enumerate(command) if part == flag or part.startswith(flag + "=")
    ]
    if len(positions) != 1:
        raise ValueError(f"Exactly one {flag} argument is required")
    index = positions[0]
    if command[index] == flag:
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise ValueError(f"Missing {flag} value")
        value = command[index + 1]
    else:
        value = command[index][len(flag) + 1 :]
    if not value:
        raise ValueError(f"Missing {flag} value")
    return value


def workspace_matches(record, root):
    try:
        return Path(option(record["command"], "--workspace")).resolve() == root.resolve()
    except (KeyError, ValueError, IndexError, TypeError):
        return False


def coordinator(root, proc=Path("/proc")):
    runtime = root / "vision/runtime/runpod"
    try:
        identity = json.loads((runtime / "job-identity.json").read_text())
        pid = int((runtime / "job.pid").read_text())
        current = process(pid, proc)
        if (
            matches(identity, current)
            and workspace_matches(current, root)
            and any(Path(part).name == "parallel_study.py" for part in current["command"])
            and not any(
                part == "--worker" or part.startswith("--worker=") for part in current["command"]
            )
        ):
            return current
    except (OSError, ValueError, TypeError):
        pass
    return None
