"""Pinned KolektorSDD2 acquisition and development-only image manifests.

Official test images and masks stay in a separate protected directory. Development
loaders validate hashes without decoding protected pixels or opening test labels.
"""

from __future__ import annotations

import hashlib
import os
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold

from ..common import fingerprint, read_json, sha256_file, utc_now, write_json

ARCHIVE_URL = "https://go.vicos.si/kolektorsdd2"
ARCHIVE_SHA256 = "edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8"
AUTHOR_MD5 = "1ed0f628122776ace4965afa30dfe316"
SEED = 20261026
EXPECTED_COUNTS = {"train": 2331, "test": 1004}
OPAQUE_METADATA = {f"split_weakly_{value}.pyb" for value in (0, 16, 53, 126, 246)}
EXCLUDED_COPIES = {
    "train/10301 (copy).png": "29ceeb9311153c1526ca7e623912d6f25da072738ffa62a8ccb37e2a7535403e",
    "train/10301_GT (copy).png": "d7bbd6bf21d1bbbc53acae6102699c43d960bbbc2b34ea3a53c614544e0366bb",
}
COLUMNS = ["record_id", "image_path", "mask_path", "label", "group_id", "fold"]


def vision_root(root: Path) -> Path:
    return Path(root).resolve() / "vision"


def immutable_json(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable artifact conflicts: {path.name}")
        return
    write_json(path, value)


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Accept only the publisher's two image trees; never call extractall."""
    files = []
    seen: set[str] = set()
    total = 0
    for member in archive.infolist():
        name = member.orig_filename
        path = PurePosixPath(name)
        if name in OPAQUE_METADATA:
            files.append(member)
            continue
        if (
            "\\" in name
            or path.is_absolute()
            or ".." in path.parts
            or ":" in name
            or not path.parts
            or path.parts[0] not in {"train", "test"}
        ):
            raise ValueError(f"Unexpected archive path: {name}")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError("Archive symlinks are forbidden")
        if member.is_dir():
            if len(path.parts) != 1:
                raise ValueError("Unexpected nested archive directory")
            continue
        if len(path.parts) != 2 or path.suffix.lower() != ".png":
            raise ValueError(f"Unexpected archive file: {name}")
        if name.casefold() in seen:
            raise ValueError("Duplicate archive filename")
        seen.add(name.casefold())
        total += member.file_size
        if member.file_size > 20 * 1024 * 1024 or total > 3 * 1024**3:
            raise ValueError("Archive exceeds declared extraction limits")
        files.append(member)
    return files


def fetch_data(root: Path) -> dict:
    base = vision_root(root)
    source = base / "source"
    source.mkdir(parents=True, exist_ok=True)
    target = source / "KolektorSDD2.zip"
    temporary = target.with_suffix(".zip.download")
    if not target.exists():
        if not temporary.exists():
            request = urllib.request.Request(
                ARCHIVE_URL, headers={"User-Agent": "SECOM-Quality-Lab/2.0"}
            )
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as output,
            ):
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > 2 * 1024**3:
                        raise ValueError("Archive download exceeds 2 GiB")
                    output.write(block)
        if sha256_file(temporary) != ARCHIVE_SHA256:
            raise ValueError("KSDD2 archive hash mismatch; dataset-version review required")
        os.replace(temporary, target)
    if sha256_file(target) != ARCHIVE_SHA256:
        raise ValueError("KSDD2 archive hash mismatch; dataset-version review required")
    with zipfile.ZipFile(target) as archive:
        members = safe_members(archive)
        member_hashes = {}
        split_images: dict[str, dict[str, str]] = {"train": {}, "test": {}}
        for member in members:
            content = archive.read(member)
            digest = hashlib.sha256(content).hexdigest()
            member_hashes[member.filename] = digest
            if member.filename in OPAQUE_METADATA:
                continue  # Untrusted pickle metadata is hashed, never extracted/deserialized.
            split, name = PurePosixPath(member.filename).parts
            if member.filename in EXCLUDED_COPIES:
                if digest != EXCLUDED_COPIES[member.filename]:
                    raise ValueError("Reviewed duplicate source bytes changed")
                continue
            destination = (
                base / "images" / "train" / name
                if split == "train"
                else base / "protected" / "images" / name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256_file(destination) != digest:
                    raise ValueError(f"Extracted source changed: {member.filename}")
            else:
                temporary_image = destination.with_suffix(".png.partial")
                temporary_image.write_bytes(content)
                os.replace(temporary_image, destination)
            member_hashes[member.filename] = digest
            split_images[split][name] = str(destination)
        for split, paths in split_images.items():
            ids = sorted(name[:-4] for name in paths if not name.endswith("_GT.png"))
            if len(ids) != EXPECTED_COUNTS[split] or len(paths) != 2 * len(ids):
                raise ValueError(f"Unexpected {split} image/mask count")
            if any(f"{identifier}_GT.png" not in paths for identifier in ids):
                raise ValueError(f"Unpaired {split} image/mask")
        train_hashes = {
            value: key
            for key, value in member_hashes.items()
            if key.startswith("train/") and not key.endswith("_GT.png")
        }
        overlap = [
            name
            for name, value in member_hashes.items()
            if name.startswith("test/") and not name.endswith("_GT.png") and value in train_hashes
        ]
        if overlap:
            raise ValueError(
                "Exact image-byte duplicates across official partitions; review required"
            )
    manifest = {
        "dataset": "KolektorSDD2",
        "source_url": ARCHIVE_URL,
        "publisher_url": "https://www.vicos.si/resources/kolektorsdd2/",
        "archive_sha256": ARCHIVE_SHA256,
        "author_md5": AUTHOR_MD5,
        "license": "CC-BY-NC-SA-4.0",
        "expected_image_counts": EXPECTED_COUNTS,
        "member_sha256": member_hashes,
        "dataset_version_review": {
            "review_id": "ksdd2-canonical-v1",
            "raw_training_pairs": 2332,
            "canonical_training_pairs": 2331,
            "excluded_exact_copy_files": EXCLUDED_COPIES,
            "reason": "Publisher archive includes a byte-identical10301 image/mask copy pair; remove only that proven duplication to match official counts, preserving archive and hashes.",
            "opaque_metadata_not_loaded": sorted(OPAQUE_METADATA),
        },
        "protected_pixels_decoded": False,
        "test_label_counts": {"source": "publisher_metadata", "positive": 110, "negative": 894},
    }
    manifest["source_fingerprint"] = fingerprint(manifest)
    immutable_json(source / "manifest.json", manifest)
    return {key: value for key, value in manifest.items() if key != "member_sha256"}


def _perceptual_hash(rgb: np.ndarray) -> int:
    image = Image.fromarray(rgb).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    spectrum = dctn(np.asarray(image, dtype=float), type=2, norm="ortho")[:8, :8]
    bits = spectrum.flatten() > np.median(spectrum.flatten()[1:])
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def audit_pairs(paths: list[tuple[str, Path, Path]]) -> tuple[pd.DataFrame, dict]:
    """Decode training pairs only; conservative near duplicates form shared groups."""
    rows: list[dict] = []
    hashes: list[int] = []
    thumbnails: list[np.ndarray] = []
    parents = list(range(len(paths)))
    exact: dict[str, int] = {}
    exact_pairs = []
    near_pairs = []
    dimensions: dict[str, int] = {}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, (record_id, image_path, mask_path) in enumerate(paths):
        with Image.open(image_path) as image:
            if image.format != "PNG":
                raise ValueError("Expected PNG source images")
            rgb = np.asarray(image.convert("RGB"))
            size = image.size
        with Image.open(mask_path) as mask:
            if mask.size != size:
                raise ValueError(f"Image/mask dimensions differ: {record_id}")
            mask_array = np.asarray(mask.convert("L"))
        if not set(np.unique(mask_array)).issubset({0, 255}):
            raise ValueError(f"Nonbinary source mask: {record_id}")
        label = int(np.any(mask_array))
        digest = hashlib.sha256(rgb.tobytes() + str(rgb.shape).encode()).hexdigest()
        if digest in exact:
            other = exact[digest]
            if rows[other]["label"] != label:
                raise ValueError("Duplicate image has conflicting labels")
            parents[find(index)] = find(other)
            exact_pairs.append([rows[other]["record_id"], record_id])
        else:
            exact[digest] = index
        phash = _perceptual_hash(rgb)
        thumbnail = np.asarray(
            Image.fromarray(rgb).resize((64, 64), Image.Resampling.BOX), dtype=np.float32
        )
        for other, earlier_hash in enumerate(hashes):
            if (phash ^ earlier_hash).bit_count() <= 4:
                rmse = float(np.sqrt(np.mean((thumbnail - thumbnails[other]) ** 2)))
                if rmse <= 1.0 and rows[other]["pixel_sha256"] != digest:
                    if rows[other]["label"] != label:
                        raise ValueError(
                            "Conservative near duplicate has conflicting labels; review required"
                        )
                    parents[find(index)] = find(other)
                    near_pairs.append([rows[other]["record_id"], record_id, rmse])
        hashes.append(phash)
        thumbnails.append(thumbnail)
        key = f"{size[0]}x{size[1]}"
        dimensions[key] = dimensions.get(key, 0) + 1
        rows.append(
            {
                "record_id": record_id,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "label": label,
                "pixel_sha256": digest,
            }
        )
    for index, row in enumerate(rows):
        row["group_id"] = rows[find(index)]["record_id"]
    frame = pd.DataFrame(rows)
    frame["fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (_, validation) in enumerate(
        splitter.split(frame, frame["label"], frame["group_id"])
    ):
        frame.loc[validation, "fold"] = fold
    audit = {
        "count": len(frame),
        "failures": int(frame["label"].sum()),
        "dimensions": dimensions,
        "exact_duplicate_pairs": exact_pairs,
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_rule": "64-bit pHash distance <=4 AND 64x64 RGB thumbnail RMSE <=1 of255",
        "physical_item_groups": "not documented; group_id links only verified exact/conservative near duplicates",
        "folds": [
            {
                "fold": fold,
                "count": int((frame.fold == fold).sum()),
                "failures": int(frame.loc[frame.fold == fold, "label"].sum()),
            }
            for fold in range(3)
        ],
    }
    return frame, audit


def audit_data(root: Path) -> dict:
    base = vision_root(root)
    audit_path = base / "data" / "audit.json"
    if audit_path.exists():
        load_development(root)
        return read_json(audit_path)
    source = fetch_data(root)
    directory = base / "images" / "train"
    paths = [
        (f"ksdd2-train-{image.stem}", image, image.with_name(f"{image.stem}_GT.png"))
        for image in sorted(directory.glob("*.png"))
        if not image.stem.endswith("_GT")
    ]
    frame, details = audit_pairs(paths)
    if details["count"] != 2331 or details["failures"] != 246:
        raise ValueError("Training label count differs from official KSDD2; review required")
    output = base / "data"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "train.parquet"
    staging = output / "train.parquet.staging"
    frame[[*COLUMNS, "pixel_sha256"]].to_parquet(staging, index=False)
    os.replace(staging, destination)
    protected = []
    for image in sorted((base / "protected" / "images").glob("*.png")):
        if not image.stem.endswith("_GT"):
            protected.append(
                {
                    "record_id": f"ksdd2-test-{image.stem}",
                    "image_path": str(image.resolve()),
                    "mask_path": str(image.with_name(f"{image.stem}_GT.png").resolve()),
                    "label": None,
                    "group_id": None,
                    "fold": None,
                }
            )
    test_path = base / "protected" / "test.parquet"
    pd.DataFrame(protected)[COLUMNS].to_parquet(test_path, index=False)
    audit = {
        "dataset": "KolektorSDD2",
        "source_fingerprint": source["source_fingerprint"],
        "archive_sha256": ARCHIVE_SHA256,
        "development": details,
        "train_manifest_sha256": sha256_file(destination),
        "protected_manifest_sha256": sha256_file(test_path),
        "test_count": len(protected),
        "protected_pixels_decoded": False,
        "test_labels": "unread; official positive110/negative894 from publisher metadata",
        "cross_partition_exact_image_byte_duplicates": 0,
        "cross_partition_near_duplicates": "not assessed; protected pixels remain unopened",
        "grouping_limitation": "No published physical-item or capture-session mapping has been verified; no temporal/deployment generalisation claim.",
        "created_at": utc_now(),
    }
    audit["audit_fingerprint"] = fingerprint(audit)
    immutable_json(audit_path, audit)
    return audit


def load_development(root: Path) -> pd.DataFrame:
    base = vision_root(root)
    audit = read_json(base / "data" / "audit.json")
    if (
        fingerprint({key: value for key, value in audit.items() if key != "audit_fingerprint"})
        != audit["audit_fingerprint"]
    ):
        raise ValueError("Development audit checksum failed")
    path = base / "data" / "train.parquet"
    if sha256_file(path) != audit["train_manifest_sha256"]:
        raise ValueError("Development manifest checksum failed")
    frame = pd.read_parquet(path)
    required = set(COLUMNS)
    if not required.issubset(frame.columns) or len(frame) != 2331:
        raise ValueError("Development manifest schema/count mismatch")
    allowed = (base / "images" / "train").resolve()
    for key in ("image_path", "mask_path"):
        if any(not Path(value).resolve().is_relative_to(allowed) for value in frame[key]):
            raise ValueError("Development manifest references protected or external paths")
    if frame.record_id.duplicated().any() or not set(frame.label).issubset({0, 1}):
        raise ValueError("Development identity/label contract failed")
    if set(frame.fold) != {0, 1, 2} or frame.groupby("group_id").fold.nunique().max() != 1:
        raise ValueError("Development fold/group contract failed")
    manifest = read_json(base / "source" / "manifest.json")
    if (
        fingerprint({key: value for key, value in manifest.items() if key != "source_fingerprint"})
        != manifest["source_fingerprint"]
        or manifest["source_fingerprint"] != audit["source_fingerprint"]
    ):
        raise ValueError("Source manifest checksum failed")
    for row in frame.itertuples():
        for value in (row.image_path, row.mask_path):
            path = Path(value)
            if sha256_file(path) != manifest["member_sha256"][f"train/{path.name}"]:
                raise ValueError("Development image/mask checksum failed")
    return frame


def load_protected(root: Path, *, selection_hash: str, authorization: dict) -> pd.DataFrame:
    """Final-stage only: caller must durably record access before passing authorization."""
    if (
        not selection_hash
        or authorization.get("selection_hash") != selection_hash
        or not authorization.get("access_recorded")
    ):
        raise ValueError("Protected data requires matching frozen selection and recorded access")
    base = vision_root(root)
    research = base / "research"
    if (
        not (research / "selection.json").is_file()
        or not (research / "final-access.json").is_file()
    ):
        raise ValueError("Protected data requires a durable frozen selection and access ledger")
    selection = read_json(research / "selection.json")
    actual_authorization = read_json(research / "final-access.json")
    expected_selection = fingerprint(
        {key: value for key, value in selection.items() if key != "selection_hash"}
    )
    if (
        selection.get("selection_hash") != expected_selection
        or expected_selection != selection_hash
        or authorization != actual_authorization
        or authorization.get("protocol_hash") != selection.get("protocol_hash")
    ):
        raise ValueError("Protected access does not match durable frozen selection/ledger")
    audit = read_json(base / "data" / "audit.json")
    path = base / "protected" / "test.parquet"
    if sha256_file(path) != audit["protected_manifest_sha256"]:
        raise ValueError("Protected manifest checksum failed")
    frame = pd.read_parquet(path)
    labels = []
    manifest = read_json(base / "source" / "manifest.json")
    allowed = (base / "protected" / "images").resolve()
    for row in frame.itertuples():
        for value in (row.image_path, row.mask_path):
            path = Path(value).resolve()
            if (
                not path.is_relative_to(allowed)
                or sha256_file(path) != manifest["member_sha256"][f"test/{path.name}"]
            ):
                raise ValueError("Protected image/mask path or checksum failed")
        with Image.open(row.mask_path) as mask:
            labels.append(int(np.any(np.asarray(mask.convert("L")))))
    frame["label"] = labels
    frame["group_id"] = frame[
        "record_id"
    ]  # Physical groups unknown: explicitly use images as units.
    if len(frame) != 1004 or sum(labels) != 110:
        raise ValueError(
            "Protected count/labels differ from official data; version review required"
        )
    return frame
