"""Vision predictors shared by offline experiments and serving.

Torch is optional for the released sensor application. Neural weights are always
saved as safetensors, including the frozen ImageNet extractor for offline use.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .transforms import image_array, mask_array

if TYPE_CHECKING:
    from torch import Tensor

SOURCE_COMMIT = "98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d"
MODEL_SCHEMA = "vision-predictor-v1"
MAX_CANDIDATE_PATCHES = 100_000


def seed_runtime(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Exact bitwise equality across GPU architectures is not promised.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device(device: str) -> str:
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; explicitly request device='cpu' for CPU work.")
    return device


def _backbone(kind: str, pretrained: bool):
    import torch
    from torchvision.models import (
        ResNet18_Weights,
        Wide_ResNet50_2_Weights,
        resnet18,
        wide_resnet50_2,
    )
    from torchvision.models.feature_extraction import create_feature_extractor

    if kind == "resnet18_logistic":
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        extractor = create_feature_extractor(network, return_nodes={"layer4": "embedding"})
    else:
        network = wide_resnet50_2(
            weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        extractor = create_feature_extractor(network, return_nodes=["layer2", "layer3"])
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    extractor.eval()
    if not all(torch.isfinite(p).all() for p in extractor.parameters()):
        raise ValueError("Nonfinite backbone weights")
    return extractor


def _batch(paths: list[Path], resolution: tuple[int, int], device: str):
    import torch

    arrays, geometries = zip(*(image_array(Path(path), resolution) for path in paths), strict=True)
    return torch.from_numpy(np.stack(arrays)).to(device), geometries


def _patch_features(extractor: Any, images: Any):
    import torch
    import torch.nn.functional as functional

    features = list(extractor(images).values())
    features = [functional.avg_pool2d(value, 3, 1, 1) for value in features]
    return torch.cat(
        [
            features[0],
            functional.interpolate(
                features[1], size=features[0].shape[-2:], mode="bilinear", align_corners=False
            ),
        ],
        dim=1,
    )


class VisionPredictor:
    """One self-contained, reviewed architecture with uncalibrated outputs."""

    def __init__(self, config: dict[str, Any], network: Any, device: str = "cpu"):
        self.config = dict(config)
        self.family = str(config["family"])
        resolution = config.get("resolution", [704, 256])
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.device = _device(device)
        self.network = network.to(self.device).eval()
        self.head: dict[str, Any] = {}
        self.memory: Tensor | None = None
        self.training: dict[str, Any] = {}

    def predict(self, image_paths: list[Path]) -> tuple[np.ndarray, list[np.ndarray]]:
        import torch
        import torch.nn.functional as functional
        from scipy.ndimage import gaussian_filter

        scores: list[float] = []
        maps: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(image_paths), 4):
                images, geometries = _batch(
                    image_paths[start : start + 4], self.resolution, self.device
                )
                if self.family == "resnet18_logistic":
                    features = self.network(images)["embedding"]
                    coefficients = torch.tensor(
                        self.head["coefficients"], device=self.device, dtype=features.dtype
                    )
                    logits = (features * coefficients[None, :, None, None]).sum(dim=1)
                    logits += self.head["intercept"]
                    batch_scores = logits.mean(dim=(1, 2))
                    batch_maps = functional.interpolate(
                        logits[:, None], self.resolution, mode="bilinear", align_corners=False
                    )[:, 0]
                elif self.family == "supersimplenet":
                    batch_maps, batch_scores = self.network(images)
                    batch_maps = batch_maps[:, 0]
                elif self.family == "patchcore":
                    features = _patch_features(self.network, images)
                    batch_size, channels, height, width = features.shape
                    flat = features.permute(0, 2, 3, 1).reshape(-1, channels)
                    assert self.memory is not None
                    bank = self.memory.to(self.device, dtype=flat.dtype)
                    distance_chunks, nearest_chunks = [], []
                    for chunk in flat.split(512):
                        distance = torch.cdist(chunk, bank)
                        values, indices = distance.min(dim=1)
                        distance_chunks.append(values)
                        nearest_chunks.append(indices)
                    distances = torch.cat(distance_chunks).reshape(batch_size, -1)
                    nearest = torch.cat(nearest_chunks).reshape(batch_size, -1)
                    highest, index = distances.max(dim=1)
                    batch_scores = highest.clone()
                    # PatchCore's image score downweights isolated memory matches.
                    for row in range(batch_size):
                        closest = bank[nearest[row, index[row]]]
                        neighbor_indices = (
                            torch.cdist(closest[None], bank)
                            .topk(k=min(9, len(bank)), largest=False)
                            .indices[0]
                        )
                        query = flat.reshape(batch_size, -1, channels)[row, index[row]]
                        neighbor_distances = torch.linalg.vector_norm(
                            bank[neighbor_indices] - query, dim=1
                        )
                        weight = 1 - torch.softmax(neighbor_distances, dim=0)[0]
                        batch_scores[row] *= weight
                    batch_maps = functional.interpolate(
                        distances.reshape(batch_size, 1, height, width),
                        self.resolution,
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                else:
                    raise ValueError(f"Unsupported model family: {self.family}")
                raw_maps = batch_maps.detach().float().cpu().numpy()
                raw_scores = batch_scores.detach().float().cpu().numpy().reshape(-1)
                if not np.isfinite(raw_scores).all() or not np.isfinite(raw_maps).all():
                    raise FloatingPointError("nonfinite_model_output")
                for score, spatial, geometry in zip(raw_scores, raw_maps, geometries, strict=True):
                    if self.family == "patchcore":
                        spatial = gaussian_filter(spatial, sigma=4)
                    scores.append(float(score))
                    maps.append(geometry.restore(spatial))
        return np.asarray(scores, dtype=np.float64), maps

    def save(self, output: Path) -> None:
        from safetensors.torch import save_file

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in self.network.state_dict().items()
            },
            output / "weights.safetensors",
        )
        if self.memory is not None:
            save_file(
                {"memory": self.memory.detach().cpu().contiguous()}, output / "memory.safetensors"
            )
        metadata = {
            "schema": MODEL_SCHEMA,
            "config": self.config,
            "head": self.head,
            "training": self.training,
            "score_kind": "uncalibrated",
            "source_commit": SOURCE_COMMIT if self.family == "supersimplenet" else None,
        }
        (output / "predictor.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )


def load_predictor(path: Path, device: str = "cpu") -> VisionPredictor:
    from safetensors.torch import load_file

    path = Path(path)
    metadata = json.loads((path / "predictor.json").read_text(encoding="utf-8"))
    if metadata["schema"] != MODEL_SCHEMA:
        raise ValueError("Unsupported vision predictor schema")
    config = metadata["config"]
    family = config["family"]
    if family == "supersimplenet":
        from .vendor.supersimplenet import SuperSimpleNet

        network = SuperSimpleNet(tuple(config["resolution"]), _ssn_config(config, pretrained=False))
    elif family in {"patchcore", "resnet18_logistic"}:
        network = _backbone(family, pretrained=False)
    else:
        raise ValueError("Unexpected model family")
    network.load_state_dict(load_file(path / "weights.safetensors"), strict=True)
    predictor = VisionPredictor(config, network, device)
    predictor.head = metadata["head"]
    predictor.training = metadata["training"]
    if family == "patchcore":
        predictor.memory = load_file(path / "memory.safetensors")["memory"]
        if not 1 <= len(predictor.memory) <= 20_000:
            raise ValueError("Invalid memory bank size")
    return predictor


def _ssn_config(config: dict[str, Any], pretrained: bool = True) -> dict[str, Any]:
    multiplier = float(config.get("lr_multiplier", 1))
    return {
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": False,
        "adapt_cls_feat": False,
        "noise_std": 0.015,
        "perlin_thr": 0.6,
        "seg_lr": 0.0002 * multiplier,
        "dec_lr": 0.0002 * multiplier,
        "adapt_lr": 0.0001 * multiplier,
        "gamma": 0.4,
        "stop_grad": False,
        "epochs": int(config.get("epochs", 100)),
        "pretrained": pretrained,
    }


def _fit_linear(predictor: VisionPredictor, train: pd.DataFrame) -> None:
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    embeddings = []
    paths = list(map(Path, train.image_path))
    with torch.inference_mode():
        for start in range(0, len(paths), 8):
            images, _ = _batch(paths[start : start + 8], predictor.resolution, predictor.device)
            embeddings.append(predictor.network(images)["embedding"].mean((2, 3)).cpu().numpy())
    features = np.concatenate(embeddings)
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=float(predictor.config["C"]), class_weight="balanced", max_iter=5000, tol=1e-4
    ).fit(scaler.transform(features), train.label)
    if int(model.n_iter_.max()) >= 5000:
        raise RuntimeError("Logistic regression did not converge")
    if model.classes_.tolist() != [0, 1]:
        raise ValueError("Failure class encoding mismatch")
    coefficients = model.coef_[0] / scaler.scale_
    predictor.head = {
        "coefficients": coefficients.tolist(),
        "intercept": float(model.intercept_[0] - coefficients @ scaler.mean_),
    }


def approximate_coreset(features: np.ndarray, fraction: float, limit: int, seed: int) -> np.ndarray:
    """Seeded approximate greedy k-center on a bounded uniform patch pool.

    The fraction applies to the bounded pool, not every original training patch.
    This explicit approximation bounds both memory and greedy-search complexity.
    """
    if not 0 < fraction <= 1 or len(features) == 0:
        raise ValueError("Invalid coreset input")
    rng = np.random.default_rng(seed)
    count = max(1, min(limit, math.ceil(len(features) * fraction)))
    projection = rng.normal(size=(features.shape[1], min(128, features.shape[1]))).astype(
        np.float32
    )
    projection /= math.sqrt(projection.shape[1])
    reduced = np.asarray(features, dtype=np.float32) @ projection
    squared = np.einsum("ij,ij->i", reduced, reduced)
    anchors = rng.choice(len(features), size=min(10, len(features)), replace=False)
    minimum = np.full(len(features), np.inf, dtype=np.float32)
    for anchor in anchors:
        minimum = np.minimum(minimum, squared + squared[anchor] - 2 * reduced @ reduced[anchor])
    selected = []
    for _ in range(count):
        index = int(np.argmax(minimum))
        selected.append(index)
        minimum = np.minimum(minimum, squared + squared[index] - 2 * reduced @ reduced[index])
        minimum[selected] = -np.inf
    return np.asarray(selected)


def _fit_patchcore(predictor: VisionPredictor, train: pd.DataFrame, seed: int) -> None:
    import torch

    paths = list(map(Path, train.loc[train.label == 0, "image_path"]))
    if not paths:
        raise ValueError("PatchCore needs normal training images")
    rng = np.random.default_rng(seed)
    reservoir: NDArray[np.float32] | None = None
    priorities: NDArray[np.float64] | None = None
    seen = 0
    with torch.inference_mode():
        for start in range(0, len(paths), 4):
            images, geometries = _batch(
                paths[start : start + 4], predictor.resolution, predictor.device
            )
            features = _patch_features(predictor.network, images)
            for value, geometry in zip(features, geometries, strict=True):
                # Retain only patch centres within observed image, never letterbox padding.
                _channels, height, width = value.shape
                ys = (np.arange(height) + 0.5) * geometry.target_height / height
                xs = (np.arange(width) + 0.5) * geometry.target_width / width
                valid = (
                    (ys[:, None] >= geometry.top)
                    & (ys[:, None] < geometry.top + geometry.height)
                    & (xs[None, :] >= geometry.left)
                    & (xs[None, :] < geometry.left + geometry.width)
                )
                chunk = value.permute(1, 2, 0).cpu().numpy()[valid].astype(np.float32)
                seen += len(chunk)
                keys = rng.random(len(chunk))
                if reservoir is not None:
                    assert priorities is not None
                    chunk = np.concatenate((reservoir, chunk))
                    keys = np.concatenate((priorities, keys))
                if len(chunk) > MAX_CANDIDATE_PATCHES:
                    keep = np.argpartition(keys, -MAX_CANDIDATE_PATCHES)[-MAX_CANDIDATE_PATCHES:]
                    chunk, keys = chunk[keep], keys[keep]
                reservoir, priorities = chunk, keys
    assert reservoir is not None
    indices = approximate_coreset(
        reservoir,
        float(predictor.config["coreset_fraction"]),
        min(20_000, int(predictor.config.get("max_memory_patches", 20_000))),
        seed,
    )
    predictor.memory = torch.from_numpy(reservoir[indices].copy())
    predictor.training.update(
        {
            "normal_images": len(paths),
            "original_patches": seen,
            "uniform_candidate_patches": len(reservoir),
            "memory_patches": len(indices),
            "projection_dimensions": 128,
            "coreset_algorithm": "uniform100k_then_approximate_greedy_kcenter",
        }
    )


def fit_model(
    config: dict[str, Any], train: pd.DataFrame, seed: int, output: Path, device: str = "cuda"
) -> VisionPredictor:
    """Fit exclusively on supplied training rows, never retrieve a held-out set."""
    seed_runtime(seed)
    _device(device)
    if len(train) == 0 or not {"image_path", "label"}.issubset(train.columns):
        raise ValueError("Training table is empty or lacks required columns")
    if not set(train.label.unique()).issubset({0, 1}):
        raise ValueError("Invalid training labels")
    if "partition" in train and not set(train.partition).issubset({"train", "development"}):
        raise ValueError("Held-out partitions cannot enter fit_model")
    family = config["family"]
    start = time.perf_counter()
    if family == "supersimplenet":
        from .vendor.supersimplenet import SuperSimpleNet

        network = SuperSimpleNet(tuple(config["resolution"]), _ssn_config(config))
    elif family in {"resnet18_logistic", "patchcore"}:
        network = _backbone(family, pretrained=True)
    else:
        raise ValueError(f"Unknown model family: {family}")
    predictor = VisionPredictor(config, network, device)
    predictor.training = {
        "seed": seed,
        "training_rows": len(train),
        "training_record_ids": train.record_id.astype(str).tolist(),
        "pretraining": "ImageNet1K_V1",
        "device": device,
    }
    if family == "resnet18_logistic":
        _fit_linear(predictor, train)
    elif family == "patchcore":
        _fit_patchcore(predictor, train, seed)
    else:
        _fit_ssn(predictor, train, seed, Path(output))
    predictor.training["training_seconds"] = time.perf_counter() - start
    predictor.network.eval()
    predictor.save(output)
    return predictor


def _training_sample(row: Any, resolution: tuple[int, int], flip: int):
    from PIL import Image, ImageFilter
    from scipy.ndimage import distance_transform_edt, label

    values, geometry = image_array(Path(row.image_path), resolution)
    mask, valid = mask_array(None, geometry)
    if int(row.label):
        with Image.open(row.mask_path) as source:
            native_image = source.convert("L").filter(ImageFilter.MaxFilter(7))
            native_mask = (
                np.asarray(
                    native_image.resize((geometry.width, geometry.height), Image.Resampling.NEAREST)
                )
                > 0
            )
        mask[
            geometry.top : geometry.top + geometry.height,
            geometry.left : geometry.left + geometry.width,
        ] = native_mask
    weights = np.ones_like(mask)
    regions, count = label(mask)
    for region in range(1, count + 1):
        distance = distance_transform_edt(regions == region)
        if distance.max() > 0:
            distance /= distance.max()
            weights[regions == region] = (3 * distance**2)[regions == region]
    if flip in {1, 3}:
        values, mask, valid, weights = (
            np.flip(values, axis=1),
            np.flip(mask, axis=0),
            np.flip(valid, axis=0),
            np.flip(weights, axis=0),
        )
    if flip in {2, 3}:
        values, mask, valid, weights = (
            np.flip(values, axis=2),
            np.flip(mask, axis=1),
            np.flip(valid, axis=1),
            np.flip(weights, axis=1),
        )
    return tuple(array.copy() for array in (values, mask, valid, weights))


def _save_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    rng: Any,
    frequencies: np.ndarray,
) -> None:
    import torch
    from safetensors.torch import save_file

    tensors: dict[str, Any] = {}

    def encode(value):
        if isinstance(value, torch.Tensor):
            key = f"tensor_{len(tensors)}"
            tensors[key] = value.detach().cpu().contiguous()
            return {"__tensor__": key}
        if isinstance(value, np.ndarray):
            return {"__ndarray__": encode(torch.from_numpy(value.copy()))}
        if isinstance(value, tuple):
            return {"__tuple__": [encode(item) for item in value]}
        if isinstance(value, dict):
            return {"__dict__": [[encode(key), encode(item)] for key, item in value.items()]}
        if isinstance(value, list):
            return [encode(item) for item in value]
        return value

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "numpy_rng": rng.bit_generator.state,
        "frequencies": frequencies,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_global_rng": np.random.get_state(),
    }
    metadata = encode(state)
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path / "checkpoint.tmp.safetensors")
    checksum = hashlib.sha256((path / "checkpoint.tmp.safetensors").read_bytes()).hexdigest()
    (path / "checkpoint.tmp.json").write_text(
        json.dumps({"safetensors_sha256": checksum, "state": metadata}), encoding="utf-8"
    )
    (path / "checkpoint.tmp.safetensors").replace(path / "checkpoint.safetensors")
    (path / "checkpoint.tmp.json").replace(path / "checkpoint.json")


def _restore_checkpoint(path: Path, model: Any, optimizer: Any, scheduler: Any, rng: Any):
    import torch
    from safetensors.torch import load_file

    if not (path / "checkpoint.json").exists():
        return 0, None
    checkpoint = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    checksum = hashlib.sha256((path / "checkpoint.safetensors").read_bytes()).hexdigest()
    if checksum != checkpoint.get("safetensors_sha256"):
        raise ValueError("Checkpoint pair is incomplete or corrupt; refusing unsafe resume")
    # Detach mapped storage: Windows cannot replace an open mapped checkpoint,
    # and CPU optimizer state / NumPy views would otherwise retain that mapping.
    tensors = {
        key: value.clone() for key, value in load_file(path / "checkpoint.safetensors").items()
    }

    def decode(value):
        if isinstance(value, dict):
            if "__tensor__" in value:
                return tensors[value["__tensor__"]]
            if "__ndarray__" in value:
                return decode(value["__ndarray__"]).numpy()
            if "__tuple__" in value:
                return tuple(decode(item) for item in value["__tuple__"])
            if "__dict__" in value:
                return {decode(key): decode(item) for key, item in value["__dict__"]}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value

    state = decode(checkpoint["state"])
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng.bit_generator.state = state["numpy_rng"]
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_global_rng"])
    return int(state["epoch"]), state["frequencies"]


def _fit_ssn(predictor: VisionPredictor, train: pd.DataFrame, seed: int, output: Path) -> None:
    import torch
    import torch.nn.functional as functional

    from .vendor.loss import focal_loss

    output.mkdir(parents=True, exist_ok=True)
    model = predictor.network
    positive = train.loc[train.label == 1].reset_index(drop=True)
    negative = train.loc[train.label == 0].reset_index(drop=True)
    if positive.empty or negative.empty:
        raise ValueError("Supervised SSN needs defective and normal training images")
    optimizer, scheduler = model.get_optimizers()
    rng = np.random.default_rng(seed)
    epoch_start, frequencies = _restore_checkpoint(output, model, optimizer, scheduler, rng)
    if frequencies is None:
        frequencies = np.zeros(len(negative), dtype=np.float64)
    epochs = int(predictor.config.get("epochs", 100))
    history_path = output / "training_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    batch_size = 8
    for epoch in range(epoch_start, epochs):
        started = time.perf_counter()
        model.train()
        # Match the source's four explicit positive flips and equally many normals.
        count = 4 * len(positive)
        if frequencies.max() > 0:
            probability = 1 - frequencies / frequencies.max()
            probability = (probability - np.median(probability) + 1) ** (np.log(len(negative)) * 4)
            probability /= probability.sum()
        else:
            probability = np.full(len(negative), 1 / len(negative))
        normal_indices = rng.choice(
            len(negative), count, replace=count > len(negative), p=probability
        )
        frequencies += np.bincount(normal_indices, minlength=len(negative))
        examples = [
            (positive.iloc[index], flip) for flip in range(4) for index in range(len(positive))
        ]
        examples += [(negative.iloc[index], 0) for index in normal_indices]
        order = rng.permutation(len(examples))
        losses = []
        for start in range(0, len(order), batch_size):
            samples, labels = [], []
            for index in order[start : start + batch_size]:
                row, flip = examples[index]
                samples.append(_training_sample(row, predictor.resolution, flip))
                labels.append(float(row.label))
            arrays = [
                torch.from_numpy(np.stack(items)).to(predictor.device)
                for items in zip(*samples, strict=True)
            ]
            images, masks, valid, weights = arrays
            size = (model.fh, model.fw)
            masks = functional.interpolate(
                masks[:, None], size=size, mode="bilinear", align_corners=True
            )
            masks = (masks >= 0.5).float()
            valid = functional.interpolate(valid[:, None], size=size, mode="nearest")
            weights = functional.interpolate(
                weights[:, None], size=size, mode="bilinear", align_corners=True
            )
            labels_tensor = torch.tensor(labels, device=predictor.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=predictor.device.startswith("cuda"),
            ):
                anomaly_map, scores, generated_masks, generated_labels = model(
                    images, masks, labels_tensor
                )
            anomaly_map, scores = anomaly_map.float(), scores.float()
            valid = torch.cat((valid, valid))
            weights = torch.cat((weights, weights))
            # Synthetic anomalies confined to letterbox padding cannot make a defect label.
            generated_labels = torch.maximum(
                torch.cat((labels_tensor, labels_tensor)),
                (generated_masks * valid).flatten(1).any(1).float(),
            )
            focal = (
                focal_loss(torch.sigmoid(anomaly_map), generated_masks, reduction=None) * weights
            )
            normal = (generated_masks == 0) & (valid > 0)
            abnormal = (generated_masks > 0) & (valid > 0)
            good_loss = (
                ((anomaly_map + 0.5).clamp(min=0) * weights)[normal].mean()
                if normal.any()
                else anomaly_map.sum() * 0
            )
            bad_loss = (
                ((-anomaly_map + 0.5).clamp(min=0) * weights)[abnormal].mean()
                if abnormal.any()
                else anomaly_map.sum() * 0
            )
            loss = good_loss + bad_loss + focal[valid > 0].mean()
            loss += focal_loss(torch.sigmoid(scores), generated_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite_training_loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "epochs": epochs,
                "mean_loss": float(np.mean(losses)),
                "seconds": time.perf_counter() - started,
                "examples": len(examples),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated()
                if predictor.device.startswith("cuda")
                else None,
            }
        )
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        progress = {
            "state": "training" if epoch + 1 < epochs else "completed",
            "config_id": predictor.config.get("config_id"),
            **history[-1],
        }
        (output / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
        _save_checkpoint(output, model, optimizer, scheduler, epoch + 1, rng, frequencies)
        print(json.dumps(progress), flush=True)
    predictor.training.update(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "training_precision": "cuda_bfloat16_autocast_fp32_master"
            if predictor.device.startswith("cuda")
            else "float32",
            "inference_precision": "float32",
            "deterministic_algorithms": "warn_only; adaptive maxpool CUDA backward is nondeterministic",
            "source_commit": SOURCE_COMMIT,
            "recipe": _ssn_config(predictor.config),
            "adaptations": [
                "100 epochs and batch8 as frozen protocol",
                "CUDA BF16 forward, FP32 master weights and losses; FP32 inference",
                "letterboxpadding excluded from segmentation loss",
                "distance weights computed after mask letterboxing",
                "no test evaluation or batch-wise score normalization",
            ],
        }
    )
