"""Cloud-only preprocessing parity and source-archive integrity, using synthetic images."""

import ast
import hashlib
import importlib.util
import json
import os
import random
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "runpod_optimize_source", ROOT / "scripts/runpod/optimize_source.py"
)
optimizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(optimizer)


@pytest.fixture
def patched():
    source = (ROOT / optimizer.MODELS_PATH).read_bytes()
    module = types.ModuleType("secom_quality.vision.cloud_cache_test")
    module.__package__ = "secom_quality.vision"
    exec(compile(optimizer.patch_models(source), "cloud_models.py", "exec"), module.__dict__)
    return module


def sample(tmp_path, name="image", defect=True, shape=(61, 25)):
    height, width = shape
    pixels = np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3)
    image = tmp_path / f"{name}.png"
    Image.fromarray((pixels % 256).astype(np.uint8)).save(image)
    mask_path = tmp_path / f"{name}-mask.png"
    if defect:
        mask = np.zeros(shape, dtype=np.uint8)
        mask[3:10, 3:9] = 255
        mask[-12:-3, -10:-2] = 255
        Image.fromarray(mask).save(mask_path)
    return SimpleNamespace(image_path=str(image), mask_path=str(mask_path), label=int(defect))


def assert_arrays_equal(actual, expected):
    for first, second in zip(actual, expected, strict=True):
        assert first.shape == second.shape
        assert first.dtype == second.dtype
        assert first.tobytes() == second.tobytes()
        assert first.flags.c_contiguous and first.flags.writeable


@pytest.mark.parametrize("shape", [(61, 25), (25, 61)])
@pytest.mark.parametrize("defect", [False, True])
def test_all_flips_are_byte_equal_and_return_independent_arrays(patched, tmp_path, shape, defect):
    row = sample(tmp_path, defect=defect, shape=shape)
    resolution = (80, 40)
    original = patched._training_sample_uncached
    calls = []

    def counted(row, resolution, flip):
        calls.append((row.image_path, tuple(resolution), flip))
        return original(row, resolution, flip)

    patched._training_sample_uncached = counted
    for flip in [3, 0, 2, 1, 3]:
        expected = original(row, resolution, flip)
        actual = patched._training_sample(row, resolution, flip)
        assert_arrays_equal(actual, expected)
        for array in actual:
            array.fill(99)
        assert_arrays_equal(patched._training_sample(row, resolution, flip), expected)
    assert calls == [(row.image_path, resolution, 0)]
    assert all(
        not array.flags.writeable
        for base in patched._TRAINING_SAMPLE_CACHE.values()
        for array in base
    )
    assert patched._TRAINING_SAMPLE_CACHE_BYTES == 80 * 40 * 24


def test_cache_preserves_rng_and_ordered_batch_inputs(patched, tmp_path):
    torch = pytest.importorskip("torch")
    rows = [sample(tmp_path, f"image-{index}", defect=index != 1) for index in range(3)]
    order = [(2, 1), (0, 3), (1, 0), (2, 2), (0, 0), (1, 3)]
    expected = [
        patched._training_sample_uncached(rows[index], (48, 24), flip) for index, flip in order
    ]
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    local_rng = np.random.default_rng(52)
    local_state = local_rng.bit_generator.state
    actual = [patched._training_sample(rows[index], (48, 24), flip) for index, flip in order]
    for first, second in zip(actual, expected, strict=True):
        assert_arrays_equal(first, second)
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    assert np.array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert local_rng.bit_generator.state == local_state


def test_lru_eviction_is_byte_bounded_and_cache_hits_refresh_recency(patched, tmp_path):
    rows = [sample(tmp_path, f"image-{index}", defect=False) for index in range(3)]
    sample_bytes = 32 * 16 * 24
    patched._TRAINING_SAMPLE_CACHE_MAX_BYTES = 2 * sample_bytes
    for index in [0, 1, 0, 2]:
        patched._training_sample(rows[index], (32, 16), 0)
        assert 2 * sample_bytes >= patched._TRAINING_SAMPLE_CACHE_BYTES
    paths = [key[0][0] for key in patched._TRAINING_SAMPLE_CACHE]
    assert paths == [
        str(Path(rows[0].image_path).resolve()),
        str(Path(rows[2].image_path).resolve()),
    ]
    assert 2 * sample_bytes == patched._TRAINING_SAMPLE_CACHE_BYTES


def test_oversized_samples_are_never_retained(patched, tmp_path):
    row = sample(tmp_path)
    patched._TRAINING_SAMPLE_CACHE_MAX_BYTES = 1
    assert_arrays_equal(
        patched._training_sample(row, (48, 24), 3),
        patched._training_sample_uncached(row, (48, 24), 3),
    )
    assert not patched._TRAINING_SAMPLE_CACHE
    assert patched._TRAINING_SAMPLE_CACHE_BYTES == 0


@pytest.mark.parametrize("changed", ["image_path", "mask_path"])
def test_file_stat_changes_invalidate_cached_preprocessing(patched, tmp_path, changed):
    row = sample(tmp_path)
    before = patched._training_sample(row, (80, 40), 0)
    path = Path(getattr(row, changed))
    status = path.stat()
    if changed == "image_path":
        Image.new("RGB", (25, 61), (7, 11, 13)).save(path)
    else:
        Image.new("L", (25, 61), 255).save(path)
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000))
    after = patched._training_sample(row, (80, 40), 0)
    assert_arrays_equal(after, patched._training_sample_uncached(row, (80, 40), 0))
    assert any(
        first.tobytes() != second.tobytes() for first, second in zip(before, after, strict=True)
    )
    assert len(patched._TRAINING_SAMPLE_CACHE) == 2


def make_handoff(path, *, tamper=None):
    members = {
        optimizer.MODELS_PATH: (ROOT / optimizer.MODELS_PATH).read_bytes(),
        "configs/vision-requirements.lock": b"preserved Windows lock",
        "migration/windows_protocol.json": b'{"preserved": true}',
    }
    manifest = {
        "schema": "secom-runpod-source-v1",
        "files": {name: hashlib.sha256(body).hexdigest() for name, body in members.items()},
        "contains_credentials": False,
        "contains_test_images": False,
    }
    members["handoff-manifest.json"] = json.dumps(manifest).encode()
    if tamper:
        members[tamper] = b"not the committed member"
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return members


def test_archive_patch_preserves_original_and_regenerates_every_member_hash(tmp_path):
    original = tmp_path / "source.zip"
    source_members = make_handoff(original)
    before = original.read_bytes()
    output = tmp_path / "cloud.zip"
    result = optimizer.optimize_handoff(original, output)
    assert original.read_bytes() == before
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert Path(result["sha256_file"]).read_text().strip() == f"{result['sha256']}  cloud.zip"
    with zipfile.ZipFile(output) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members.pop("handoff-manifest.json"))
    assert manifest["files"] == {
        name: hashlib.sha256(body).hexdigest() for name, body in members.items()
    }
    for name in ("configs/vision-requirements.lock", "migration/windows_protocol.json"):
        assert members[name] == source_members[name]
    record = json.loads(members["migration/cloud-optimizations.json"])
    assert record["default_cache_max_bytes"] == 24 * 1024**3
    assert (
        record["original_models_sha256"]
        == hashlib.sha256(source_members[optimizer.MODELS_PATH]).hexdigest()
    )
    assert record["optimized_models_sha256"] == manifest["files"][optimizer.MODELS_PATH]
    original_tree = ast.parse(source_members[optimizer.MODELS_PATH])
    patched_tree = ast.parse(members[optimizer.MODELS_PATH])
    original_sample = next(
        node
        for node in original_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_training_sample"
    )
    retained_sample = next(
        node
        for node in patched_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_training_sample_uncached"
    )
    retained_sample.name = "_training_sample"
    assert ast.dump(original_sample) == ast.dump(retained_sample)
    second_output = tmp_path / "cloud-again.zip"
    assert optimizer.optimize_handoff(original, second_output)["sha256"] == result["sha256"]


@pytest.mark.parametrize("tamper", [optimizer.MODELS_PATH, "unlisted.txt"])
def test_unverified_members_are_rejected_without_output(tmp_path, tamper):
    original = tmp_path / "source.zip"
    make_handoff(original, tamper=tamper)
    output = tmp_path / "cloud.zip"
    with pytest.raises(ValueError, match="hashes do not match"):
        optimizer.optimize_handoff(original, output)
    assert not output.exists()


def test_repeated_patching_and_overwriting_original_are_rejected(tmp_path):
    original = tmp_path / "source.zip"
    make_handoff(original)
    with pytest.raises(ValueError, match="separate"):
        optimizer.optimize_handoff(original, original)
    output = tmp_path / "cloud.zip"
    optimizer.optimize_handoff(original, output)
    with pytest.raises(ValueError, match="existing cloud optimization"):
        optimizer.optimize_handoff(output, tmp_path / "twice.zip")
    with pytest.raises(FileExistsError, match="fresh output"):
        optimizer.optimize_handoff(original, output)


def test_cache_keys_isolate_labels_resolution_and_mask_identity(patched, tmp_path):
    defect = sample(tmp_path)
    normal = SimpleNamespace(image_path=defect.image_path, mask_path=defect.mask_path, label=0)
    other_mask = tmp_path / "other-mask.png"
    Image.new("L", (25, 61), 255).save(other_mask)
    another = SimpleNamespace(image_path=defect.image_path, mask_path=str(other_mask), label=1)
    for row, resolution in [
        (defect, (80, 40)),
        (normal, (80, 40)),
        (another, (80, 40)),
        (defect, (40, 80)),
    ]:
        assert_arrays_equal(
            patched._training_sample(row, resolution, 0),
            patched._training_sample_uncached(row, resolution, 0),
        )
    assert len(patched._TRAINING_SAMPLE_CACHE) == 4


@pytest.mark.parametrize("member", ["../outside.py", "C:/outside.py", "directory\\outside.py"])
def test_unsafe_archive_paths_are_rejected(tmp_path, member):
    original = tmp_path / "source.zip"
    make_handoff(original)
    with zipfile.ZipFile(original, "a") as archive:
        archive.writestr(member, b"invalid")
    with pytest.raises(ValueError):
        optimizer.optimize_handoff(original, tmp_path / "cloud.zip")


def test_duplicate_archive_members_are_rejected(tmp_path):
    original = tmp_path / "source.zip"
    make_handoff(original)
    with (
        zipfile.ZipFile(original, "a") as archive,
        pytest.warns(UserWarning, match="Duplicate name"),
    ):
        archive.writestr(optimizer.MODELS_PATH, b"duplicate")
    with pytest.raises(ValueError, match="Duplicate ZIP"):
        optimizer.optimize_handoff(original, tmp_path / "cloud.zip")
