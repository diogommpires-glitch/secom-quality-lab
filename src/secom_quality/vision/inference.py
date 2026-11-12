"""Image validation and authoritative finite-batch inspection policies."""

import csv
import hashlib
import io
import math
import shutil
import stat
import uuid
import warnings
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from secom_quality.common import fingerprint, utc_now
from secom_quality.inference import ApplicationError

from .storage import VisionStore

MAX_BYTES = 20 * 1024 * 1024
MAX_IMAGES = 100
MAX_PIXELS = 16_000_000
MAX_DECODED_BYTES = 100 * 1024 * 1024


def decode_image(content: bytes, filename: str) -> Any:
    """Decode verified PNG/JPEG pixels; metadata never becomes a model feature."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not content or len(content) > MAX_BYTES:
        raise ApplicationError("image_size", "Each image must contain 1 byte to 20 MiB.")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ApplicationError("image_format", "Only PNG and JPEG images are supported.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                expected = "PNG" if suffix == ".png" else "JPEG"
                if image.format != expected or getattr(image, "n_frames", 1) != 1:
                    raise ApplicationError(
                        "image_format",
                        "The image format must match its filename and contain one frame.",
                    )
                width, height = image.size
                if min(width, height) < 16 or width * height > MAX_PIXELS:
                    raise ApplicationError(
                        "image_dimensions",
                        "Images need sides of at least 16 pixels and at most 16 million pixels.",
                    )
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except ApplicationError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ApplicationError(
            "image_decode", "The image is corrupt or cannot be decoded safely."
        ) from exc


def parse_upload(content: bytes, filename: str) -> list[tuple[str, Any]]:
    if len(content) > MAX_BYTES:
        raise ApplicationError("upload_too_large", "Upload exceeds 20 MiB.", 413)
    files: list[tuple[str, bytes]] = []
    if Path(filename).suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                members = archive.infolist()
                if not 1 <= len(members) <= MAX_IMAGES:
                    raise ApplicationError(
                        "image_count", "ZIP batches require 1 to 100 image files."
                    )
                expanded = 0
                for member in members:
                    path = PurePosixPath(member.filename)
                    mode = member.external_attr >> 16
                    if (
                        member.is_dir()
                        or len(path.parts) != 1
                        or path.is_absolute()
                        or "\\" in member.filename
                        or ":" in member.filename
                        or path.name in {".", ".."}
                        or stat.S_ISLNK(mode)
                        or member.flag_bits & 1
                    ):
                        raise ApplicationError(
                            "archive_member",
                            "Use a flat ZIP containing only unencrypted PNG/JPEG files.",
                        )
                    expanded += member.file_size
                    if expanded > MAX_DECODED_BYTES or member.file_size > MAX_BYTES:
                        raise ApplicationError(
                            "archive_size", "Expanded ZIP content exceeds the image budget.", 413
                        )
                    files.append((member.filename, archive.read(member)))
        except ApplicationError:
            raise
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError) as exc:
            raise ApplicationError(
                "archive_decode", "The ZIP archive cannot be read safely."
            ) from exc
    else:
        files = [(filename, content)]
    decoded = []
    identities: set[str] = set()
    total_pixels = 0
    for name, body in files:
        record_id = Path(name).stem
        if not record_id.strip() or len(record_id) > 200 or record_id.casefold() in identities:
            raise ApplicationError(
                "record_id",
                "Image names must have distinct nonempty stems of at most 200 characters.",
            )
        if any(ord(char) < 32 for char in record_id):
            raise ApplicationError("record_id", "Image names cannot contain control characters.")
        identities.add(record_id.casefold())
        image = decode_image(body, name)
        total_pixels += image.width * image.height
        if total_pixels * 3 > MAX_DECODED_BYTES:
            raise ApplicationError(
                "decoded_size", "Decoded RGB images exceed the 100 MiB batch budget.", 413
            )
        decoded.append((record_id, image))
    return decoded


def load_active(root: Path) -> dict[str, Any]:
    try:
        from .bundle import load_verified_bundle

        return load_verified_bundle(root / "vision" / "bundles" / "main")
    except Exception as exc:
        raise ApplicationError(
            "vision_unavailable", "A verified, trained vision bundle is not available.", 503
        ) from exc


def process_batch(
    root: Path, content: bytes, filename: str, request_id: str, capacity: float = 0.1
) -> dict[str, Any]:
    if capacity not in {0.1, 0.2}:
        raise ApplicationError("capacity", "Use inspection capacity 0.1 or 0.2.")
    if not request_id.strip() or len(request_id) > 200:
        raise ApplicationError(
            "client_request_id", "A request ID from 1 to 200 characters is required."
        )
    request_hash = fingerprint(
        {"content": hashlib.sha256(content).hexdigest(), "filename": filename, "capacity": capacity}
    )
    store = VisionStore(root)
    previous = store.retry(request_id, request_hash)
    if previous:
        return previous
    images = parse_upload(content, filename)
    bundle = load_active(root)
    batch_id = uuid.uuid4().hex
    directory = root / "vision" / "assets" / batch_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        paths, identities = [], []
        for index, (record_id, image) in enumerate(images):
            asset_id = f"{index:03d}"
            path = directory / f"{asset_id}.png"
            image.save(path, format="PNG")
            paths.append(path)
            identities.append((record_id, asset_id))
        scores, maps = bundle["predictor"].predict(paths)
        if len(scores) != len(paths) or len(maps) != len(paths):
            raise ApplicationError(
                "prediction_shape", "The predictor returned an invalid batch shape.", 503
            )
        manifest = bundle["manifest"]
        predictions = []
        from PIL import Image

        for index, ((record_id, asset_id), raw_score, raw_map) in enumerate(
            zip(identities, scores, maps, strict=True)
        ):
            image = images[index][1]
            score = float(raw_score)
            heat = np.asarray(raw_map, dtype=np.float32)
            reasons = []
            if not math.isfinite(score):
                reasons.append("nonfinite_score")
            if heat.shape != (image.height, image.width) or not np.isfinite(heat).all():
                reasons.append("invalid_localisation_map")
            overlay_url = None
            if not reasons:
                lower, upper = float(heat.min()), float(heat.max())
                scaled = (heat - lower) / (upper - lower) if upper > lower else np.zeros_like(heat)
                colors = np.stack([scaled, 0.3 * scaled, 1 - scaled], axis=-1) * 255
                blend = np.asarray(image, dtype=np.float32) * 0.6 + colors * 0.4
                Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8)).save(
                    directory / f"{asset_id}-overlay.png"
                )
                overlay_url = f"/api/v2/vision/batches/{batch_id}/assets/{asset_id}/overlay"
            predictions.append(
                {
                    "prediction_id": uuid.uuid4().hex,
                    "record_id": record_id,
                    "model_id": manifest.get("model_id", "unrecorded"),
                    "bundle_id": manifest.get("bundle_id", "unrecorded"),
                    "feature_schema_version": "industrial-image-v1",
                    "risk_score": score if not reasons else None,
                    "score_kind": "uncalibrated",
                    "calibration_status": "not_validated",
                    "quality_status": "abstain" if reasons else "valid",
                    "quality_reason_codes": reasons,
                    "input_fingerprint": hashlib.sha256(paths[index].read_bytes()).hexdigest(),
                    "image_url": f"/api/v2/vision/batches/{batch_id}/assets/{asset_id}/image",
                    "overlay_url": overlay_url,
                    "width": image.width,
                    "height": image.height,
                    "rank": None,
                    "review_recommended": False,
                }
            )
        eligible = sorted(
            (p for p in predictions if p["risk_score"] is not None),
            key=lambda p: (-p["risk_score"], hashlib.sha256(p["record_id"].encode()).hexdigest()),
        )
        count = math.floor(capacity * len(eligible))
        for index, prediction in enumerate(eligible, 1):
            prediction["rank"] = index
            prediction["review_recommended"] = index <= count
        result = {
            "batch_id": batch_id,
            "created_at": utc_now(),
            "row_count": len(predictions),
            "eligible_count": len(eligible),
            "abstain_count": len(predictions) - len(eligible),
            "review_capacity": count,
            "capacity": capacity,
            "predictions": predictions,
            "model_id": manifest.get("model_id"),
            "bundle_id": manifest.get("bundle_id"),
            "queue_policy_id": "capacity-desc-score-sha256-id-v1",
            "replayed": False,
            "map_interpretation": "Per-image display normalisation; model localisation evidence, not confidence or physical cause.",
        }
        saved = store.save(request_id, request_hash, result)
        if saved["batch_id"] != batch_id:
            shutil.rmtree(directory)
        return saved
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def export_csv(batch: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    fields = [
        "record_id",
        "risk_score",
        "rank",
        "review_recommended",
        "quality_status",
        "bundle_id",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for prediction in batch["predictions"]:
        row = {field: prediction.get(field) for field in fields}
        for field, value in row.items():
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                row[field] = "'" + value
        writer.writerow(row)
    return output.getvalue()
