"""Native AD2 storage, with an explicitly configured external routing adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from functools import lru_cache
from pathlib import Path, PurePosixPath
from types import ModuleType


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@lru_cache(maxsize=1)
def _load_adapter(path: Path) -> tuple[ModuleType, str]:
    digest = _digest(path)
    spec = importlib.util.spec_from_file_location("_secom_explicit_ad2_storage", path)
    if spec is None or spec.loader is None:
        raise ImportError("AD2 storage adapter cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if any(
        not callable(getattr(module, name, None))
        for name in ("resolve", "member", "member_root", "names")
    ):
        raise ValueError("AD2 storage adapter must provide resolve/member/member_root/names")
    return module, digest


def _adapter() -> tuple[ModuleType, str] | None:
    configured = os.environ.get("SECOM_AD2_STORAGE_ADAPTER")
    if not configured:
        return None
    path = Path(configured)
    if not path.is_absolute() or path.suffix.lower() != ".py":
        raise ValueError("SECOM_AD2_STORAGE_ADAPTER requires an absolute Python module path")
    path = path.resolve(strict=True)
    module, digest = _load_adapter(path)
    if _digest(path) != digest:
        raise ValueError("AD2 storage adapter changed after loading; use a new process/protocol")
    return module, digest


def provenance() -> dict:
    selected = _adapter()
    return (
        {"mode": "native-filesystem"}
        if selected is None
        else {"mode": "explicit-adapter", "module_sha256": selected[1]}
    )


def resolve(path) -> Path:
    selected = _adapter()
    return Path(path) if selected is None else Path(selected[0].resolve(path))


def member(base: Path, relative: str) -> Path:
    selected = _adapter()
    if selected is not None:
        return Path(selected[0].member(base, relative))
    name = str(relative)
    parts = PurePosixPath(name)
    if (
        not name
        or parts.is_absolute()
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or any(character in name for character in ("\\", ":", "\0"))
    ):
        raise ValueError("Unsafe AD2 storage member path")
    base = Path(base)
    candidate = base.joinpath(*parts.parts)
    if not candidate.resolve().is_relative_to(base.resolve()):
        raise ValueError("Unsafe AD2 storage member location")
    for current in (
        base,
        *[base.joinpath(*parts.parts[:i]) for i in range(1, len(parts.parts) + 1)],
    ):
        if current.is_symlink() or current.is_junction():
            raise ValueError("Unsafe AD2 storage link or junction")
    return candidate


def member_root(base: Path, relative: str) -> Path:
    selected = _adapter()
    return Path(base) if selected is None else Path(selected[0].member_root(base, relative))


def names(root: Path) -> set[str]:
    selected = _adapter()
    if selected is not None:
        return set(selected[0].names(root))
    root = Path(root)
    if root.is_symlink() or root.is_junction():
        raise ValueError("Unsafe AD2 storage root link or junction")
    found = set()
    for path in root.rglob("*"):
        if path.is_symlink() or path.is_junction():
            raise ValueError("Unsafe AD2 storage link or junction")
        if path.is_file():
            found.add(path.relative_to(root).as_posix())
    return found
