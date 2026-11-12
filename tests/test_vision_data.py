"""Tests of image boundaries, canonical source review and deterministic folds."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from secom_quality.vision.data import EXCLUDED_COPIES, audit_pairs, load_protected, safe_members
from secom_quality.vision.protocol import RESOLUTIONS, candidate_configs


def test_candidates_are_bounded_and_portrait():
    configs = candidate_configs()
    assert len(configs) == 13
    assert len({row["config_id"] for row in configs}) == 13
    assert len([row for row in configs if row["family"] == "patchcore"]) == 4
    assert RESOLUTIONS == ([704, 256], [1056, 384])
    assert all(row["resolution"][0] > row["resolution"][1] for row in configs)


@pytest.mark.parametrize(
    "name",
    [
        "../escape.png",
        "train/../../escape.png",
        "C:/escape.png",
        "train/nested/image.png",
        "unexpected.pyb",
    ],
)
def test_archive_rejects_unexpected_paths(name):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr(name, b"anything")
    with zipfile.ZipFile(content) as archive, pytest.raises(ValueError):
        safe_members(archive)


def test_opaque_pickle_is_not_loaded_and_names_are_restricted():
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("split_weakly_0.pyb", b"this is not valid pickle")
        archive.writestr("train/1.png", b"anything")
    with zipfile.ZipFile(content) as archive:
        assert len(safe_members(archive)) == 2
    assert set(EXCLUDED_COPIES) == {"train/10301 (copy).png", "train/10301_GT (copy).png"}


def make_pairs(root: Path):
    paths = []
    rng = np.random.default_rng(17)
    for index in range(12):
        rgb = rng.integers(0, 256, size=(48, 16, 3), dtype=np.uint8)
        mask = np.zeros((48, 16), dtype=np.uint8)
        if index % 2:
            mask[10:12, 5:7] = 255
        image = root / f"{index}.png"
        target = root / f"{index}_GT.png"
        Image.fromarray(rgb).save(image)
        Image.fromarray(mask).save(target)
        paths.append((f"train-{index}", image, target))
    return paths


def test_masks_labels_alignment_and_fold_reproduction(tmp_path):
    paths = make_pairs(tmp_path)
    first, audit = audit_pairs(paths)
    second, _ = audit_pairs(paths)
    assert first.fold.tolist() == second.fold.tolist()
    assert audit["failures"] == 6
    assert set(first.fold) == {0, 1, 2}
    assert first.groupby("group_id").fold.nunique().max() == 1
    Image.new("L", (8, 8)).save(paths[0][2])
    with pytest.raises(ValueError, match="dimensions"):
        audit_pairs(paths)


def test_exact_duplicates_stay_in_same_fold(tmp_path):
    paths = make_pairs(tmp_path)
    original = paths[0]
    duplicate_image = tmp_path / "copy.png"
    duplicate_mask = tmp_path / "copy_GT.png"
    duplicate_image.write_bytes(original[1].read_bytes())
    duplicate_mask.write_bytes(original[2].read_bytes())
    paths.append(("duplicate", duplicate_image, duplicate_mask))
    frame, audit = audit_pairs(paths)
    assert audit["exact_duplicate_pairs"] == [["train-0", "duplicate"]]
    assert frame.iloc[0].fold == frame.iloc[-1].fold


def test_final_access_requires_recorded_matching_selection(tmp_path):
    for authorization in ({}, {"selection_hash": "wrong", "access_recorded": True}):
        with pytest.raises(ValueError, match="frozen selection"):
            load_protected(tmp_path, selection_hash="selected", authorization=authorization)


def test_final_access_rejects_forged_in_memory_authorization(tmp_path):
    with pytest.raises(ValueError, match="durable"):
        load_protected(
            tmp_path,
            selection_hash="selected",
            authorization={"selection_hash": "selected", "access_recorded": True},
        )
