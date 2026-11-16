"""AD2 storage routing is explicit, scoped, lazy and protocol-visible."""

import hashlib

import pytest

from secom_quality.common import read_json as common_read_json
from secom_quality.common import sha256_file as common_sha256
from secom_quality.vision import ad2, ad2_storage


def test_native_ad2_storage_needs_no_private_machine_module(tmp_path, monkeypatch):
    monkeypatch.delenv("SECOM_AD2_STORAGE_ADAPTER", raising=False)
    path = tmp_path / "development/manifest.json"
    path.parent.mkdir()
    path.write_text('{"normal":true}')
    assert ad2.read_json(path) == {"normal": True}
    assert ad2.sha256_file(path) == common_sha256(path)
    assert ad2_storage.provenance() == {"mode": "native-filesystem"}
    assert ad2_storage.names(tmp_path) == {"development/manifest.json"}
    with pytest.raises(ValueError, match="Unsafe AD2"):
        ad2_storage.member(tmp_path, "../outside")


def test_explicit_adapter_routes_ad2_only_and_pins_its_code(tmp_path, monkeypatch):
    logical = tmp_path / "logical.json"
    physical = tmp_path / "physical.json"
    logical.write_text('{"location":"local"}')
    physical.write_text('{"location":"external"}')
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "from pathlib import Path\n"
        f"LOGICAL = Path({str(logical)!r})\nPHYSICAL = Path({str(physical)!r})\n"
        "def resolve(path):\n    return PHYSICAL if Path(path)==LOGICAL else Path(path)\n"
        "def member(base,name):\n    return resolve(Path(base)/name)\n"
        "def member_root(base,name):\n    return Path(base)\n"
        "def names(root):\n    return set()\n"
    )
    monkeypatch.setenv("SECOM_AD2_STORAGE_ADAPTER", str(adapter))
    assert ad2.read_json(logical) == {"location": "external"}
    assert common_read_json(logical) == {"location": "local"}
    assert ad2.sha256_file(logical) == hashlib.sha256(physical.read_bytes()).hexdigest()
    assert common_sha256(logical) == hashlib.sha256(logical.read_bytes()).hexdigest()
    assert ad2_storage.provenance() == {
        "mode": "explicit-adapter",
        "module_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
    }
    adapter.write_text(adapter.read_text() + "\n# changed after first load\n")
    with pytest.raises(ValueError, match="adapter changed after loading"):
        ad2.read_json(logical)


def test_configured_missing_adapter_fails_instead_of_using_native_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("SECOM_AD2_STORAGE_ADAPTER", str(tmp_path / "missing.py"))
    with pytest.raises(FileNotFoundError):
        ad2_storage.provenance()
