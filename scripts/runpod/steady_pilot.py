"""Three-epoch development-training throughput probe; never read held-out images."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch

from secom_quality.common import write_json
from secom_quality.vision import models
from secom_quality.vision.data import load_development
from secom_quality.vision.protocol import checked_protocol

EPOCHS = 3
RESOLUTIONS = ((704, 256), (1056, 384))


def main(root):
    root = Path(root).resolve()
    report_path = root / "vision/runtime/runpod/steady-pilot.json"
    directories = [root / f"vision/pilots/steady-{height}" for height, _ in RESOLUTIONS]
    if report_path.exists() or any(directory.exists() for directory in directories):
        raise ValueError("Steady pilot evidence exists; preserve it and choose a fresh workspace")
    protocol = checked_protocol(root)
    development = load_development(root)
    train = development.loc[development.fold != 0].copy()
    positives = int((train.label == 1).sum())
    negatives = int((train.label == 0).sum())
    if train.empty or not positives or not negatives:
        raise ValueError("Frozen fold-0 training rows require both classes")
    if "partition" in train and not set(train.partition).issubset({"train", "development"}):
        raise ValueError("Only development-training rows may enter the pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("The steady pilot requires CUDA GPU 0")
    device = "cuda:0"
    expected_examples = 8 * positives
    results = []
    for resolution, directory in zip(RESOLUTIONS, directories, strict=True):
        config = {
            "config_id": f"cloud-steady-{resolution[0]}",
            "family": "supersimplenet",
            "resolution": list(resolution),
            "lr_multiplier": 1.0,
            "epochs": EPOCHS,
        }
        # Synchronize first to initialize the CUDA context before memory-stat calls.
        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        started = perf_counter()
        predictor = models.fit_model(config, train, protocol["seeds"][0], directory, device=device)
        torch.cuda.synchronize(0)
        fit_elapsed = perf_counter() - started
        fit_peak_cuda_bytes = torch.cuda.max_memory_allocated(0)
        history = json.loads((directory / "training_history.json").read_text(encoding="utf-8"))
        if len(history) != EPOCHS or any(
            row.get("epoch") != index + 1
            or row.get("epochs") != EPOCHS
            or row.get("examples") != expected_examples
            or not np.isfinite(row.get("seconds", float("nan")))
            or row.get("seconds", 0) <= 0
            for index, row in enumerate(history)
        ):
            raise ValueError(
                "Steady pilot history is incomplete or differs from the training recipe"
            )
        if not np.isfinite(fit_elapsed) or fit_elapsed <= 0:
            raise ValueError("Invalid complete-fit timing")
        scores, maps = predictor.predict([Path(train.iloc[0].image_path)])
        if np.size(scores) != 1 or len(maps) != 1 or not np.size(maps[0]):
            raise FloatingPointError("Training-image inference returned an invalid shape")
        if not np.isfinite(scores).all() or not np.isfinite(maps[0]).all():
            raise FloatingPointError("Training-image inference is nonfinite")
        seconds = [row["seconds"] for row in history]
        result = {
            "resolution": list(resolution),
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0),
            "training_rows": len(train),
            "positive_training_rows": positives,
            "negative_training_rows": negatives,
            "epochs": EPOCHS,
            "batch_size": 8,
            "seed": protocol["seeds"][0],
            "actual_examples_per_epoch": [row["examples"] for row in history],
            "fit_elapsed_seconds_including_checkpoints_and_final_save": fit_elapsed,
            "epoch_training_seconds_excluding_checkpoint_save": seconds,
            "first_epoch_seconds": seconds[0],
            "later_epoch_mean_seconds": float(np.mean(seconds[1:])),
            "cold_warm_interpretation": "First epoch and later two epochs are descriptive only; no steady-state speed claim from three epochs",
            "history": history,
            "fit_peak_cuda_bytes": fit_peak_cuda_bytes,
            "rss_bytes_at_end": psutil.Process().memory_info().rss,
            "preprocessing_cache_enabled": hasattr(models, "_TRAINING_SAMPLE_CACHE_BYTES"),
            "preprocessing_cache_bytes_at_end": getattr(models, "_TRAINING_SAMPLE_CACHE_BYTES", 0),
            "estimated_100_epoch_fold_hours": fit_elapsed / EPOCHS * 100 / 3600,
            "estimate_method": "Complete three-epoch fit elapsed / 3 * 100; includes cold setup, all checkpoints and final model save",
            "finite_prediction_on_one_training_image": True,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        del predictor, maps, scores
        torch.cuda.empty_cache()
    report = {
        "development_only": True,
        "fold": 0,
        "training_filter": "development.fold != 0",
        "protocol_hash": protocol["protocol_hash"],
        "neural_pilots": results,
        "interpretation": "Three-epoch throughput probes, not a benchmark or predictive evidence; no held-out metrics or images accessed",
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    main(parser.parse_args().workspace)
