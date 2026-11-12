"""Bounded RGB decoding and reversible, aspect-preserving geometry."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class Geometry:
    original_height: int
    original_width: int
    height: int
    width: int
    top: int
    left: int
    target_height: int
    target_width: int

    def restore(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (self.target_height, self.target_width):
            values = np.asarray(
                Image.fromarray(values).resize(
                    (self.target_width, self.target_height), Image.Resampling.BILINEAR
                )
            )
        crop = values[self.top : self.top + self.height, self.left : self.left + self.width]
        return np.asarray(
            Image.fromarray(crop).resize(
                (self.original_width, self.original_height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )


def read_rgb(path: Path) -> Image.Image:
    if Path(path).stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("image_too_large")
    try:
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG", "BMP", "TIFF"}:
                raise ValueError("unsupported_image_format")
            if image.width * image.height > MAX_IMAGE_PIXELS or min(image.size) < 8:
                raise ValueError("unsupported_image_dimensions")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("multi_frame_image")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid_image") from exc


def letterbox(image: Image.Image, resolution: tuple[int, int]) -> tuple[np.ndarray, Geometry]:
    target_height, target_width = resolution
    scale = min(target_height / image.height, target_width / image.width)
    height = max(1, round(image.height * scale))
    width = max(1, round(image.width * scale))
    top, left = (target_height - height) // 2, (target_width - width) // 2
    canvas = np.full((target_height, target_width, 3), (124, 116, 104), dtype=np.uint8)
    canvas[top : top + height, left : left + width] = np.asarray(
        image.resize((width, height), Image.Resampling.BILINEAR)
    )
    geometry = Geometry(
        image.height, image.width, height, width, top, left, target_height, target_width
    )
    return canvas, geometry


def image_array(path: Path, resolution: tuple[int, int]) -> tuple[np.ndarray, Geometry]:
    image, geometry = letterbox(read_rgb(path), resolution)
    values = image.astype(np.float32) / 255
    values = (values - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return values.transpose(2, 0, 1).copy(), geometry


def mask_array(path: Path | str | None, geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    shape = (geometry.target_height, geometry.target_width)
    mask = np.zeros(shape, dtype=np.float32)
    valid = np.zeros(shape, dtype=np.float32)
    region = (
        slice(geometry.top, geometry.top + geometry.height),
        slice(geometry.left, geometry.left + geometry.width),
    )
    valid[region] = 1
    if path is not None and str(path):
        with Image.open(path) as source:
            if source.size != (geometry.original_width, geometry.original_height):
                raise ValueError("mask_image_alignment")
            mask[region] = (
                np.asarray(
                    source.convert("L").resize(
                        (geometry.width, geometry.height), Image.Resampling.NEAREST
                    )
                )
                > 0
            )
    return mask, valid
