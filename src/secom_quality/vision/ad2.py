"""Independent MVTec AD2 challenge: normal-only fitting and native tiled maps.

No registration, download, or external submission occurs in this module. The
publisher archive is supplied locally; test pixels stay behind a frozen protocol.
"""

from __future__ import annotations

import gc
import hashlib
import math
import re
import shutil
import stat
import tarfile
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label as connected_components
from sklearn.metrics import average_precision_score, roc_auc_score

from secom_quality.common import fingerprint, read_json, sha256_file, utc_now, write_json

from .data import immutable_json

CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
# Publisher counts: train, validation, public-good, public-bad, EACH private split.
EXPECTED_COUNTS = {
    "can": (412, 46, 72, 90, 321),
    "fabric": (387, 43, 66, 90, 314),
    "fruit_jelly": (263, 37, 20, 60, 255),
    "rice": (313, 35, 42, 90, 277),
    "sheet_metal": (137, 19, 24, 90, 142),
    "vial": (291, 41, 35, 105, 276),
    "wallplugs": (293, 33, 60, 90, 232),
    "walnuts": (432, 48, 60, 90, 228),
}
TILE_SIZE, OVERLAP = 512, 128
MAX_TOTAL_BYTES, MAX_MEMBER_BYTES = 100 * 1024**3, 100 * 1024**2
MAX_MEMBERS = 100_000
SOURCE = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2"


def _base(root: Path) -> Path:
    return Path(root).resolve() / "vision" / "ad2"


def _relative(name: str) -> Path:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or any(char in name for char in ("\\", ":", "\0"))
        or not path.parts
    ):
        raise ValueError("Unsafe AD2 archive path")
    parts = path.parts
    if parts[0] in {"mvtec_ad_2", "MVTecAD2"}:
        parts = parts[1:]
    if len(parts) == 1 and parts[0].lower().startswith(("readme", "license")):
        return Path("source", "metadata", parts[0])
    if len(parts) < 3 or parts[0] not in CATEGORIES:
        raise ValueError(f"Unsupported official AD2 layout: {name}")
    split = parts[1]
    if split in {"train", "validation"}:
        valid = len(parts) == 4 and parts[2] == "good"
    elif split == "test_public":
        valid = (len(parts) == 4 and parts[2] in {"good", "bad"}) or (
            len(parts) == 5 and parts[2:4] == ("ground_truth", "bad")
        )
    else:
        valid = split in {"test_private", "test_private_mixed"} and len(parts) == 3
    if not valid or not re.fullmatch(r"[0-9]{3}[^/]*\.png", parts[-1]):
        raise ValueError(f"Unsupported official AD2 member: {name}")
    return Path("development" if split in {"train", "validation"} else "protected", *parts)


def _extract(archive: Path, staging: Path, files: dict[str, str]) -> None:
    total = sum((staging / name).stat().st_size for name in files)
    seen = {name.casefold() for name in files}

    def save(name, size, source):
        nonlocal total
        relative = _relative(name)
        key = relative.as_posix()
        if key.casefold() in seen:
            # Publisher category archives may repeat their identical licence.
            if relative.parts[:2] != ("source", "metadata"):
                raise ValueError("Duplicate AD2 archive member")
            body = source.read(MAX_MEMBER_BYTES + 1)
            if len(body) != size or hashlib.sha256(body).hexdigest() != files[key]:
                raise ValueError("Conflicting AD2 archive metadata")
            return
        total += size
        if (
            size < 0
            or size > MAX_MEMBER_BYTES
            or total > MAX_TOTAL_BYTES
            or len(files) >= MAX_MEMBERS
        ):
            raise ValueError("AD2 extraction budget exceeded")
        seen.add(key.casefold())
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest, copied = hashlib.sha256(), 0
        with target.open("xb") as output:
            while chunk := source.read(1024 * 1024):
                copied += len(chunk)
                if copied > size:
                    raise ValueError("AD2 archive member size mismatch")
                digest.update(chunk)
                output.write(chunk)
        if copied != size:
            raise ValueError("Truncated AD2 archive member")
        files[key] = digest.hexdigest()

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zipped:
            if len(zipped.infolist()) > MAX_MEMBERS:
                raise ValueError("Too many AD2 archive members")
            for member in zipped.infolist():
                if member.is_dir():
                    continue
                if stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1:
                    raise ValueError("AD2 links or encrypted archives are forbidden")
                with zipped.open(member) as stream:
                    save(member.orig_filename, member.file_size, stream)
    else:
        with tarfile.open(archive, "r|*") as tar:
            for index, tar_member in enumerate(tar):
                if index >= MAX_MEMBERS:
                    raise ValueError("Too many AD2 archive members")
                if tar_member.isdir():
                    continue
                if not tar_member.isfile():
                    raise ValueError("AD2 links and special files are forbidden")
                tar_stream = tar.extractfile(tar_member)
                if tar_stream is None:
                    raise ValueError("Unreadable AD2 archive member")
                with tar_stream:
                    save(tar_member.name, tar_member.size, tar_stream)


def _index(staging: Path) -> tuple[list[dict], list[dict], dict]:
    development, protected, counts = [], [], {}
    for category in CATEGORIES:
        found = []
        for split, kind in (
            ("train", "good"),
            ("validation", "good"),
            ("test_public", "good"),
            ("test_public", "bad"),
            ("test_private", None),
            ("test_private_mixed", None),
        ):
            partition = "development" if split in {"train", "validation"} else "protected"
            directory = staging / partition / category / split
            if kind:
                directory /= kind
            images = sorted(directory.glob("*.png"))
            if split in {"test_private", "test_private_mixed"}:
                suffix = "regular" if split == "test_private" else "mixed"
                expected = {
                    f"{index:03d}_{suffix}.png" for index in range(EXPECTED_COUNTS[category][4])
                }
                if {image.name for image in images} != expected:
                    raise ValueError(
                        "AD2 private filenames differ from the publisher's submission contract"
                    )
            found.append(len(images))
            for path in images:
                identity = path.relative_to(staging / partition).as_posix()
                row = {
                    "record_id": identity,
                    "group_id": identity,
                    "category": category,
                    "split": split,
                    "image_path": path.relative_to(staging).as_posix(),
                    "mask_path": None,
                    "label": 0 if kind == "good" else 1 if kind == "bad" else None,
                }
                if partition == "development":
                    with Image.open(path) as image:
                        if image.format != "PNG" or image.width * image.height > 20_000_000:
                            raise ValueError("Invalid AD2 normal image")
                        image.verify()
                    development.append(row)
                else:
                    if kind == "bad":
                        mask = directory.parent / "ground_truth" / "bad" / (path.stem + "_mask.png")
                        if not mask.is_file():
                            raise ValueError("Missing public AD2 mask")
                        row["mask_path"] = mask.relative_to(staging).as_posix()
                    protected.append(row)
        train, val, good, bad, private = EXPECTED_COUNTS[category]
        if found != [train, val, good, bad, private, private]:
            raise ValueError(
                f"AD2 {category} counts differ from the publisher: {found}; review dataset version"
            )
        counts[category] = dict(
            zip(
                ("train", "validation", "public_good", "public_bad", "private", "private_mixed"),
                found,
                strict=True,
            )
        )
    return development, protected, counts


def audit_ad2(root: Path, archive: Path) -> dict:
    """Safely ingest one complete archive or a directory of eight category archives."""
    archive = Path(archive).resolve()
    archives = (
        sorted(
            path
            for path in archive.iterdir()
            if path.is_file() and path.name.endswith((".zip", ".tar", ".tar.gz", ".tgz"))
        )
        if archive.is_dir()
        else [archive]
    )
    if not archives or any(not path.is_file() for path in archives):
        raise FileNotFoundError(
            f"AD2 challenge is pending external data. Obtain official archives from {SOURCE} and pass their local path."
        )
    if sum(path.stat().st_size for path in archives) > MAX_TOTAL_BYTES:
        raise ValueError("AD2 archive input exceeds100GiB")
    hashes = {path.name: sha256_file(path) for path in archives}
    base = _base(root)
    if (base / "source" / "manifest.json").exists():
        manifest = _verify_source(base)
        if manifest["archives"] != hashes:
            raise ValueError("Changed AD2 archives require a versioned workspace")
        return manifest
    staging = base.with_name("ad2.staging-" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    try:
        files: dict[str, str] = {}
        for path in archives:
            _extract(path, staging, files)
        development, protected, counts = _index(staging)
        for partition, rows in (("development", development), ("protected", protected)):
            path = staging / partition / "manifest.json"
            write_json(path, rows)
            files[path.relative_to(staging).as_posix()] = sha256_file(path)
        for archive in archives:
            copy = staging / "source" / "archives" / archive.name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive, copy)
            if sha256_file(copy) != hashes[archive.name]:
                raise ValueError("AD2 source changed during ingestion")
            files[copy.relative_to(staging).as_posix()] = hashes[archive.name]
        manifest = {
            "version": "ad2-source-v1",
            "dataset": "MVTec AD2",
            "source": SOURCE,
            "license": "CC BY-NC-SA 4.0",
            "archives": hashes,
            "files": files,
            "counts": counts,
            "test_pixels_decoded": False,
            "physical_groups": "Unknown public physical-item grouping; private/mixed scenes are paired by publisher filenames.",
        }
        manifest["manifest_hash"] = fingerprint(manifest)
        write_json(staging / "source" / "manifest.json", manifest)
        if base.exists():
            raise ValueError(
                "Preserve incomplete AD2 destination and use a clean versioned workspace"
            )
        staging.rename(base)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_source(base: Path) -> dict:
    path = base / "source" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"AD2 challenge is pending external data. Audit official archives from {SOURCE} first."
        )
    manifest = read_json(path)
    if manifest.get("manifest_hash") != fingerprint(
        {k: v for k, v in manifest.items() if k != "manifest_hash"}
    ):
        raise ValueError("AD2 source fingerprint mismatch")
    actual = {
        path.relative_to(base).as_posix()
        for partition in ("source", "development", "protected")
        for path in (base / partition).rglob("*")
        if path.is_file()
    }
    if actual != set(manifest["files"]) | {"source/manifest.json"}:
        raise ValueError("AD2 source has missing or unexpected files")
    for name, digest in manifest["files"].items():
        candidate = base / name
        if (
            candidate.is_symlink()
            or not candidate.resolve().is_relative_to(base.resolve())
            or sha256_file(candidate) != digest
        ):
            raise ValueError("AD2 source checksum mismatch")
    return manifest


def freeze_ad2(root: Path, source: dict | None = None) -> dict:
    from .protocol import dependency_versions

    source = source or _verify_source(_base(root))
    path = Path(root) / "vision" / "research" / "selection.json"
    if not path.exists():
        raise ValueError("Freeze the KSDD2 development selection before the AD2 challenge")
    selection = read_json(path)
    if selection.get("selection_hash") != fingerprint(
        {k: v for k, v in selection.items() if k != "selection_hash"}
    ):
        raise ValueError("KSDD2 selection fingerprint mismatch")
    candidates = [
        config
        for config in [selection["config"], *selection.get("reference_configs", [])]
        if config["family"] == "patchcore"
    ]
    if len(candidates) != 1:
        raise ValueError("KSDD2 selection must identify exactly one frozen PatchCore reference")
    protocol = {
        "version": "ad2-tiled-patchcore-v1",
        "selection_hash": selection["selection_hash"],
        "source_hash": source["manifest_hash"],
        "ksdd2_config": candidates[0],
        "config": {**candidates[0], "resolution": [TILE_SIZE, TILE_SIZE]},
        "resolution_amendment": "AD2 native512x512 tiles replace KSDD2 portrait letterboxing, frozen before AD2 test access.",
        "seed": selection["seeds"][0],
        "categories": list(CATEGORIES),
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "stitching": "Original coordinates, uniform overlap average; image score maximum stitched pixel score.",
        "fit": "Normal train images only; validation never enters memory bank.",
        "metrics": "All8category image AP/AUROC and201quantile AU-PRO30 approximation; macro average; no official private score claim.",
        "dependencies": dependency_versions(),
        "code": {
            name: sha256_file(Path(__file__).parent / name)
            for name in ("ad2.py", "models.py", "transforms.py")
        },
    }
    protocol["protocol_hash"] = fingerprint(protocol)
    immutable_json(_base(root) / "research" / "protocol.json", protocol)
    return protocol


def tile_coordinates(width: int, height: int, size: int = TILE_SIZE, overlap: int = OVERLAP):
    if min(width, height, size) <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid tiling dimensions")

    def starts(length):
        end = max(0, length - size)
        positions = list(range(0, end + 1, size - overlap))
        return positions if positions[-1] == end else [*positions, end]

    return [
        (x, y, min(size, width - x), min(size, height - y))
        for y in starts(height)
        for x in starts(width)
    ]


def _tiles(image: Image.Image, directory: Path, identity: str) -> list[dict]:
    rows = []
    token = hashlib.sha256(identity.encode()).hexdigest()[:24]
    for index, (x, y, width, height) in enumerate(tile_coordinates(*image.size)):
        tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (124, 116, 104))
        tile.paste(image.crop((x, y, x + width, y + height)), (0, 0))
        path = directory / f"{token}-{index:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        tile.save(path, format="PNG")
        rows.append(
            {
                "record_id": identity + f":tile:{index}",
                "group_id": identity,
                "image_path": str(path),
                "label": 0,
                "partition": "train",
                "mask_path": None,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )
    return rows


def predict_tiled(predictor: Any, image_path: Path, scratch: Path) -> tuple[float, np.ndarray]:
    with Image.open(image_path) as original:
        image = original.convert("RGB")
    rows = _tiles(image, scratch, str(image_path))
    total = np.zeros((image.height, image.width), dtype=np.float64)
    counts = np.zeros(total.shape, dtype=np.uint16)
    try:
        for start in range(0, len(rows), 4):
            current = rows[start : start + 4]
            _, maps = predictor.predict([Path(row["image_path"]) for row in current])
            if len(maps) != len(current):
                raise ValueError("AD2 tile prediction count mismatch")
            for row, values in zip(current, maps, strict=True):
                values = np.asarray(values, dtype=np.float32)
                if values.shape != (TILE_SIZE, TILE_SIZE) or not np.isfinite(values).all():
                    raise ValueError("AD2 tile maps must have finite native-size pixels")
                x, y, width, height = (row[key] for key in ("x", "y", "width", "height"))
                total[y : y + height, x : x + width] += values[:height, :width]
                counts[y : y + height, x : x + width] += 1
        if not counts.all():
            raise ValueError("AD2 tiling left uncovered pixels")
        result = (total / counts).astype(np.float32)
        return float(result.max()), result
    finally:
        for row in rows:
            Path(row["image_path"]).unlink(missing_ok=True)


def au_pro_30(records: list[dict], base: Path) -> dict:
    """Memory-bounded quantile integration; explicitly not the official evaluator."""
    if not records:
        raise ValueError("AD2 localisation metrics require image records")
    samples = []
    budget = max(1, 1_000_000 // len(records))
    for row in records:
        score = np.load(row["map_path"], mmap_mode="r", allow_pickle=False)
        samples.append(np.asarray(score).ravel()[:: max(1, math.ceil(score.size / budget))])
    edges = np.unique(np.quantile(np.concatenate(samples), np.linspace(0, 1, 201)))
    normal_hist, region_hist = np.zeros(len(edges) + 1), np.zeros(len(edges) + 1)
    regions = normal_pixels = 0
    for row in records:
        score = np.load(row["map_path"], mmap_mode="r", allow_pickle=False)
        if not np.isfinite(score).all():
            raise ValueError("Nonfinite AD2 map")
        if row["mask_path"] is None:
            mask = np.zeros(score.shape, dtype=bool)
        else:
            with Image.open(base / row["mask_path"]) as image:
                mask = np.asarray(image.convert("L")) > 0
        if mask.shape != score.shape:
            raise ValueError("AD2 mask/map alignment mismatch")
        bins = np.searchsorted(edges, score, side="right")
        normal_hist += np.bincount(bins[~mask], minlength=len(edges) + 1)
        normal_pixels += int((~mask).sum())
        components, count = connected_components(mask, structure=np.ones((3, 3), dtype=np.uint8))
        if count:
            sizes = np.bincount(components.ravel())
            region_hist += np.bincount(
                bins[mask], weights=1.0 / sizes[components[mask]], minlength=len(edges) + 1
            )
            regions += count
    method = "201 empirical quantile thresholds;8-connected regions; original-size pixels; local approximation, not official server scores"
    if not regions or not normal_pixels:
        return {
            "au_pro_30": None,
            "regions": regions,
            "normal_pixels": normal_pixels,
            "method": method,
        }
    fpr = np.cumsum(normal_hist[::-1])[::-1][1:] / normal_pixels
    pro = np.cumsum(region_hist[::-1])[::-1][1:] / regions
    # Preserve vertical curve segments at tied FPRs. Taking their maximum would
    # inflate area for reversed or tied anomaly scores.
    xs, ys = np.asarray(
        sorted([(0.0, 0.0), (1.0, 1.0), *zip(fpr.tolist(), pro.tolist(), strict=True)])
    ).T
    keep = xs < 0.3
    x, y = np.r_[xs[keep], 0.3], np.r_[ys[keep], np.interp(0.3, xs, ys)]
    return {
        "au_pro_30": float(np.trapezoid(y, x) / 0.3),
        "regions": regions,
        "normal_pixels": normal_pixels,
        "method": method,
    }


def _verify_files(directory: Path, files: dict) -> None:
    for name, digest in files.items():
        path = directory / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or sha256_file(path) != digest
        ):
            raise ValueError("AD2 research artifact checksum mismatch")


def _verify_fingerprint(value: dict, key: str) -> None:
    if value.get(key) != fingerprint({name: item for name, item in value.items() if name != key}):
        raise ValueError("AD2 research metadata fingerprint mismatch")


def _fit_category(base: Path, category: str, development: list[dict], protocol: dict, device: str):
    from .models import fit_model, load_predictor

    if (base / "research" / "public_test_access.json").exists():
        raise ValueError("Frozen AD2 models cannot be fitted after public-test access")
    destination = base / "research" / "models" / category
    metadata = destination / "ad2_fit.json"
    if metadata.exists():
        saved = read_json(metadata)
        _verify_fingerprint(saved, "fit_hash")
        if saved["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("AD2 category fit protocol differs")
        _verify_files(destination, saved["files"])
        return load_predictor(destination, device=device)
    cache = base / "cache" / "train" / category
    rows = []
    for row in development:
        if row["category"] == category and row["split"] == "train" and row["label"] == 0:
            with Image.open(base / row["image_path"]) as image:
                rows.extend(_tiles(image.convert("RGB"), cache, row["record_id"]))
    if not rows or len(rows) > 50_000:
        raise ValueError("AD2 normal training tile count must be1..50,000percategory")
    staging = destination.with_name(category + ".staging")
    if staging.exists():
        raise ValueError(
            "Preserve interrupted AD2 model staging and use a clean versioned workspace"
        )
    predictor = fit_model(protocol["config"], pd.DataFrame(rows), protocol["seed"], staging, device)
    files = {
        path.relative_to(staging).as_posix(): sha256_file(path)
        for path in staging.rglob("*")
        if path.is_file()
    }
    fit = {
        "protocol_hash": protocol["protocol_hash"],
        "files": files,
        "image_count": sum(
            row["category"] == category and row["split"] == "train" for row in development
        ),
        "tile_count": len(rows),
    }
    fit["fit_hash"] = fingerprint(fit)
    write_json(staging / "ad2_fit.json", fit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    shutil.rmtree(cache)
    return predictor


def _development(base: Path) -> list[dict]:
    rows = read_json(base / "development" / "manifest.json")
    if any(
        row["split"] not in {"train", "validation"}
        or row["label"] != 0
        or not row["image_path"].startswith("development/")
        for row in rows
    ):
        raise ValueError("AD2 development forbids anomalous or test records")
    return rows


def _model_bindings(base: Path, protocol: dict) -> dict:
    """Verify every frozen model, including its fit metadata and complete file set."""
    bindings = {}
    try:
        for category in CATEGORIES:
            directory = base / "research" / "models" / category
            metadata = directory / "ad2_fit.json"
            if (
                directory.is_symlink()
                or metadata.is_symlink()
                or not directory.resolve().is_relative_to(base.resolve())
            ):
                raise ValueError("Unsafe frozen AD2 model location")
            fit = read_json(metadata)
            _verify_fingerprint(fit, "fit_hash")
            if fit["protocol_hash"] != protocol["protocol_hash"]:
                raise ValueError("AD2 fitted model belongs to another protocol")
            actual = {
                path.relative_to(directory).as_posix()
                for path in directory.rglob("*")
                if path.is_file()
            }
            if actual != set(fit["files"]) | {"ad2_fit.json"}:
                raise ValueError("Frozen AD2 model has missing or unexpected files")
            _verify_files(directory, fit["files"])
            bindings[category] = {**fit["files"], "ad2_fit.json": sha256_file(metadata)}
    except OSError as exc:
        raise ValueError(
            "Frozen AD2 model files are missing or unreadable; refitting after test access is forbidden"
        ) from exc
    return bindings


def _verify_access(base: Path, protocol: dict) -> dict:
    ledger = base / "research" / "public_test_access.json"
    if not ledger.is_file():
        raise ValueError(
            "AD2 public-test access record is missing; cached results cannot be trusted"
        )
    access = read_json(ledger)
    _verify_fingerprint(access, "access_hash")
    if access["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("AD2 test was exposed to another protocol")
    bindings = _model_bindings(base, protocol)
    if access.get("models") != bindings or access.get("models_hash") != fingerprint(bindings):
        raise ValueError("Frozen AD2 model bytes changed after test access; refitting is forbidden")
    return access


def _verify_report(base: Path, protocol: dict, access: dict) -> dict:
    research = base / "research"
    report = read_json(research / "report.json")
    _verify_fingerprint(report, "report_hash")
    if (
        report["protocol_hash"] != protocol["protocol_hash"]
        or report.get("access_hash") != access["access_hash"]
    ):
        raise ValueError("AD2 report belongs to another frozen model/access record")
    actual = {
        path.relative_to(research).as_posix()
        for path in (research / "public").rglob("*")
        if path.is_file()
    }
    if actual != set(report["files"]):
        raise ValueError("AD2 completed public results have missing or unexpected files")
    _verify_files(research, report["files"])
    return report


def run_ad2(root: Path, device: str = "cuda") -> dict:
    """Fit normal-only PatchCore and evaluate all public categories once."""
    from .study import exclusive

    base = _base(root)
    source = _verify_source(base)
    with exclusive(base / "research" / "challenge.lock"):
        return _run_ad2(root, device, source)


def _run_ad2(root: Path, device: str, source: dict) -> dict:
    from .models import load_predictor

    base = _base(root)
    protocol = freeze_ad2(root, source)
    research = base / "research"
    ledger = research / "public_test_access.json"
    if ledger.exists():
        access = _verify_access(base, protocol)
    else:
        if (research / "report.json").exists() or (research / "public").exists():
            raise ValueError(
                "AD2 public results exist without their model-bound test access record"
            )
        development = _development(base)
        # All eight model byte sets must exist before opening any test manifest.
        # Release each predictor before fitting the next category to bound GPU use.
        for category in CATEGORIES:
            predictor = _fit_category(base, category, development, protocol, device)
            del predictor
            gc.collect()
        bindings = _model_bindings(base, protocol)
        access = {
            "protocol_hash": protocol["protocol_hash"],
            "models": bindings,
            "models_hash": fingerprint(bindings),
            "accessed_at": utc_now(),
            "purpose": "Frozen all-eight-category public evaluation; no tuning or refitting.",
        }
        access["access_hash"] = fingerprint(access)
        immutable_json(ledger, access)
    if (research / "report.json").exists():
        return _verify_report(base, protocol, access)
    protected = read_json(base / "protected" / "manifest.json")
    summaries = []
    with tempfile.TemporaryDirectory(prefix="ad2-tiles-") as scratch:
        for category in CATEGORIES:
            complete = research / "public" / category / "metrics.json"
            if complete.exists():
                summary = read_json(complete)
                _verify_fingerprint(summary, "result_hash")
                if summary["protocol_hash"] != protocol["protocol_hash"]:
                    raise ValueError("AD2 category result protocol differs")
                if summary.get("access_hash") != access["access_hash"]:
                    raise ValueError("AD2 category result belongs to different frozen model bytes")
                actual = {
                    path.relative_to(complete.parent).as_posix()
                    for path in complete.parent.rglob("*")
                    if path.is_file()
                }
                if actual != set(summary["files"]) | {"metrics.json"}:
                    raise ValueError("AD2 category result has missing or unexpected files")
                _verify_files(complete.parent, summary["files"])
                summaries.append(summary)
                continue
            predictor = load_predictor(research / "models" / category, device=device)
            records = []
            for row in protected:
                if row["category"] != category or row["split"] != "test_public":
                    continue
                score, values = predict_tiled(predictor, base / row["image_path"], Path(scratch))
                path = (
                    complete.parent
                    / "maps"
                    / (hashlib.sha256(row["record_id"].encode()).hexdigest() + ".npy")
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, values, allow_pickle=False)
                records.append({**row, "risk_score": score, "map_path": str(path)})
            labels, scores = (
                [row["label"] for row in records],
                [row["risk_score"] for row in records],
            )
            predictions = complete.with_name("predictions.json")
            write_json(predictions, records)
            files = {
                path.relative_to(complete.parent).as_posix(): sha256_file(path)
                for path in complete.parent.rglob("*")
                if path.is_file()
            }
            summary = {
                "category": category,
                "protocol_hash": protocol["protocol_hash"],
                "access_hash": access["access_hash"],
                "image_count": len(records),
                "defect_count": sum(labels),
                "image_average_precision": float(average_precision_score(labels, scores)),
                "image_auroc": float(roc_auc_score(labels, scores)),
                "files": files,
                **au_pro_30(records, base),
            }
            summary["result_hash"] = fingerprint(summary)
            write_json(complete, summary)
            summaries.append(summary)
            del predictor
    files = {
        path.relative_to(research).as_posix(): sha256_file(path)
        for path in (research / "public").rglob("*")
        if path.is_file()
    }
    report = {
        "status": "public_complete_private_pending",
        "protocol_hash": protocol["protocol_hash"],
        "access_hash": access["access_hash"],
        "categories": summaries,
        "files": files,
        "macro": {
            key: float(np.mean([row[key] for row in summaries]))
            if all(row[key] is not None for row in summaries)
            else None
            for key in ("image_average_precision", "image_auroc", "au_pro_30")
        },
        "limitations": [
            "Separate normal-only challenge; not transfer validation of supervised KSDD2.",
            "Different anomaly prevalence; no direct cross-dataset AP or inspection comparison.",
            "Local AU-PRO approximation; official private scores require publisher evaluation.",
        ],
        "external_submission_performed": False,
    }
    report["report_hash"] = fingerprint(report)
    immutable_json(research / "report.json", report)
    return report


def export_ad2(root: Path, device: str = "cuda") -> dict:
    """Prepare the publisher's float16TIFF layout locally, without upload."""
    from .study import exclusive

    base = _base(root)
    source = _verify_source(base)
    with exclusive(base / "research" / "challenge.lock"):
        return _export_ad2(root, device, source)


def _export_ad2(root: Path, device: str, source: dict) -> dict:
    import tifffile

    from .models import load_predictor

    base = _base(root)
    protocol = freeze_ad2(root, source)
    research = base / "research"
    if not (research / "report.json").exists():
        raise ValueError("Complete the frozen public AD2 evaluation before private export")
    access = _verify_access(base, protocol)
    _verify_report(base, protocol, access)
    complete = research / "private_export.json"
    if complete.exists():
        result = read_json(complete)
        _verify_fingerprint(result, "export_hash")
        if (
            result["protocol_hash"] != protocol["protocol_hash"]
            or result.get("public_access_hash") != access["access_hash"]
        ):
            raise ValueError("AD2 private export protocol differs")
        _verify_files(research / "private_submission", result["files"])
        if sha256_file(Path(result["archive"])) != result["archive_sha256"]:
            raise ValueError("AD2 private export archive checksum mismatch")
        return result
    staging = research / "private_submission.staging"
    target = research / "private_submission"
    if staging.exists() or target.exists():
        raise ValueError("Preserve interrupted AD2 export; use a clean versioned workspace")
    immutable_json(
        research / "private_test_access.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "public_access_hash": access["access_hash"],
            "purpose": "Local map export only; no external submission.",
        },
    )
    protected = read_json(base / "protected" / "manifest.json")
    files = {}
    with tempfile.TemporaryDirectory(prefix="ad2-private-tiles-") as scratch:
        for category in CATEGORIES:
            model = research / "models" / category
            fit = read_json(model / "ad2_fit.json")
            _verify_fingerprint(fit, "fit_hash")
            if fit["protocol_hash"] != protocol["protocol_hash"]:
                raise ValueError("AD2 private predictor protocol differs")
            _verify_files(model, fit["files"])
            predictor = load_predictor(model, device=device)
            for row in protected:
                if row["category"] != category or row["split"] not in {
                    "test_private",
                    "test_private_mixed",
                }:
                    continue
                _, values = predict_tiled(predictor, base / row["image_path"], Path(scratch))
                if np.abs(values).max() > np.finfo(np.float16).max:
                    raise ValueError("AD2 float16 map overflow; clipping is forbidden")
                relative = Path(
                    "anomaly_images", category, row["split"], Path(row["image_path"]).stem + ".tiff"
                )
                path = staging / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(path, values.astype(np.float16), photometric="minisblack")
                files[relative.as_posix()] = sha256_file(path)
            del predictor
    staging.rename(target)
    archive = research / "private_submission.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name in sorted(files):
            output.add(target / name, arcname="private_submission/" + name, recursive=False)
    result = {
        "protocol_hash": protocol["protocol_hash"],
        "public_access_hash": access["access_hash"],
        "directory": str(target),
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "files": files,
        "format": "Original-size single-channel float16TIFF; thresholded masks omitted.",
        "publisher_checker_required_before_submission": True,
        "external_submission_performed": False,
    }
    result["export_hash"] = fingerprint(result)
    immutable_json(complete, result)
    return result
