"""Future speed package parity; never train on or open benchmark evaluation data."""

import ast
import hashlib
import importlib.util
import json
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "future_source_optimizer", ROOT / "scripts/runpod/optimize_source.py"
)
optimizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(optimizer)


def model_module(body, name):
    module = types.ModuleType(name)
    module.__package__ = "secom_quality.vision"
    exec(compile(body, name, "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("seed", [0, 7, 20261026])
def test_full_patchcore_fit_preserves_candidate_order_bank_and_metadata(monkeypatch, seed):
    torch = pytest.importorskip("torch")
    original = (ROOT / optimizer.MODELS_PATH).read_bytes()
    future = optimizer.patch_priority_reservoir(
        optimizer.patch_models(original), optimizer.PRIORITY_SOURCE.read_bytes()
    )
    reference = model_module(original, "secom_quality.vision.reference_fit_test")
    candidate = model_module(future, "secom_quality.vision.future_fit_test")
    train = pd.DataFrame(
        {
            "image_path": [f"/synthetic/train/{i}.png" for i in range(12)],
            "label": [int(i % 5 == 0) for i in range(12)],
        }
    )
    geometry = SimpleNamespace(target_height=6, target_width=4, top=1, left=0, height=4, width=3)
    inputs = []
    coreset = reference.approximate_coreset

    def record_coreset(features, fraction, limit, value):
        inputs.append(features.copy())
        return coreset(features, fraction, limit, value)

    def batch(paths, resolution, device):
        return paths, [geometry] * len(paths)

    def features(network, paths):
        rows = [
            np.arange(18, dtype=np.float32).reshape(3, 3, 2) / 8 + int(Path(path).stem)
            for path in paths
        ]
        return torch.from_numpy(np.stack(rows))

    predictors = []
    for module in (reference, candidate):
        monkeypatch.setattr(module, "MAX_CANDIDATE_PATCHES", 7)
        monkeypatch.setattr(module, "_batch", batch)
        monkeypatch.setattr(module, "_patch_features", features)
        monkeypatch.setattr(module, "approximate_coreset", record_coreset)
        predictor = SimpleNamespace(
            config={"coreset_fraction": 0.5, "max_memory_patches": 20000},
            network=None,
            device="cpu",
            resolution=(6, 4),
            training={},
            memory=None,
        )
        module._fit_patchcore(predictor, train, seed)
        predictors.append(predictor)
    assert len(inputs) == 2 and inputs[0].tobytes() == inputs[1].tobytes()
    assert inputs[0].flags.c_contiguous and inputs[1].flags.c_contiguous
    assert torch.equal(predictors[0].memory, predictors[1].memory)
    assert predictors[0].training == predictors[1].training
    assert predictors[0].training["normal_images"] == 9
    assert predictors[0].training["original_patches"] == 18


def make_source(path):
    members = {
        optimizer.MODELS_PATH: (ROOT / optimizer.MODELS_PATH).read_bytes(),
        "configs/vision-requirements.lock": b"unchanged pinned dependency lock",
    }
    manifest = {
        "schema": "secom-runpod-source-v1",
        "files": {name: hashlib.sha256(body).hexdigest() for name, body in members.items()},
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
        archive.writestr("handoff-manifest.json", json.dumps(manifest))
    return path.read_bytes()


def test_future_package_hashes_all_runtime_code_and_preserves_input(tmp_path):
    original = tmp_path / "source.zip"
    before = make_source(original)
    output = tmp_path / "future.zip"
    result = optimizer.optimize_handoff(original, output, future_priority_reservoir=True)
    assert original.read_bytes() == before
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with zipfile.ZipFile(output) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members.pop("handoff-manifest.json"))
    assert manifest["files"] == {
        name: hashlib.sha256(body).hexdigest() for name, body in members.items()
    }
    record = json.loads(members["migration/cloud-optimizations.json"])
    future = record["future_priority_reservoir"]
    assert future["status"] == "candidate_requires_fresh_workspace_and_protocol"
    assert future["runtime_code_hashed_in"] == optimizer.MODELS_PATH
    assert (
        future["candidate_source_sha256"]
        == hashlib.sha256(members["migration/priority-reservoir-candidate.py"]).hexdigest()
    )
    assert record["optimized_models_sha256"] == manifest["files"][optimizer.MODELS_PATH]
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "PriorityReservoir"
        for node in ast.parse(members[optimizer.MODELS_PATH]).body
    )
    assert members["configs/vision-requirements.lock"] == b"unchanged pinned dependency lock"


def test_future_patch_refuses_repeated_patch_and_unexpected_helpers():
    original = optimizer.patch_models((ROOT / optimizer.MODELS_PATH).read_bytes())
    helper = optimizer.PRIORITY_SOURCE.read_bytes()
    future = optimizer.patch_priority_reservoir(original, helper)
    with pytest.raises(ValueError, match="original PatchCore"):
        optimizer.patch_priority_reservoir(future, helper)
    with pytest.raises(ValueError, match="self-contained"):
        optimizer.patch_priority_reservoir(original, b"import os\n" + helper)
    with pytest.raises(ValueError, match="implementation changed"):
        optimizer.patch_priority_reservoir(
            original.replace(b"np.argpartition(keys, -MAX_CANDIDATE_PATCHES)", b"np.argsort(keys)"),
            helper,
        )


def test_combined_future_profile_hashes_both_helpers(tmp_path):
    original = tmp_path / "source.zip"
    make_source(original)
    output = tmp_path / "future-both.zip"
    optimizer.optimize_handoff(
        original, output, future_priority_reservoir=True, future_prefetch=True
    )
    with zipfile.ZipFile(output) as archive:
        record = json.loads(archive.read("migration/cloud-optimizations.json"))
        source = archive.read(optimizer.MODELS_PATH)
        metadata = json.loads(archive.read("handoff-manifest.json"))
        helper = archive.read("migration/ordered-prefetch-candidate.py")
    assert record["optimized_models_sha256"] == hashlib.sha256(source).hexdigest()
    assert (
        record["future_ssn_prefetch"]["candidate_source_sha256"]
        == hashlib.sha256(helper).hexdigest()
    )
    assert metadata["files"][optimizer.MODELS_PATH] == hashlib.sha256(source).hexdigest()
    assert b"prepared_batches.close()" in source
    assert b"tensor.pin_memory()" in source and b"non_blocking=" in source
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "ordered_prefetch"
        for node in ast.parse(source).body
    )


def test_prefetch_patch_refuses_repeated_or_changed_training_loop():
    original = (ROOT / optimizer.MODELS_PATH).read_bytes()
    helper = optimizer.PREFETCH_SOURCE.read_bytes()
    candidate = optimizer.patch_ssn_prefetch(original, helper)
    with pytest.raises(ValueError, match="original SSN"):
        optimizer.patch_ssn_prefetch(candidate, helper)
    with pytest.raises(ValueError, match="batch preparation changed"):
        optimizer.patch_ssn_prefetch(
            original.replace(
                b"for index in order[start : start + batch_size]",
                b"for index in reversed(order[start : start + batch_size])",
            ),
            helper,
        )
