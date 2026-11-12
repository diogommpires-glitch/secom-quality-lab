"""Frozen development-only corruption probes; these never select a replacement."""

import gc
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from secom_quality.common import read_json, write_json

from .data import load_development
from .metrics import image_metrics
from .study import _ordering, checked_protocol, exclusive

SCENARIOS = ("clean", "brightness_0p7", "brightness_1p3", "blur_1px", "jpeg_q40", "translate_5pct")


def corrupt(image, scenario):
    if scenario.startswith("brightness_"):
        return ImageEnhance.Brightness(image).enhance(0.7 if scenario.endswith("0p7") else 1.3)
    if scenario == "blur_1px":
        return image.filter(ImageFilter.GaussianBlur(1))
    if scenario == "translate_5pct":
        output = Image.new("RGB", image.size, (124, 116, 104))
        output.paste(image, (round(image.width * 0.05), round(image.height * 0.05)))
        return output
    return image


def run_stress(root: Path, device="cuda"):
    from .models import load_predictor

    root = Path(root)
    protocol = checked_protocol(root)
    research = root / "vision/research"
    target = research / "robustness.json"
    if target.exists():
        return read_json(target)
    development = read_json(research / "development.json")
    winner = sorted([row for row in development["finalists"] if row["valid"]], key=_ordering)[0]
    config_id = winner["config"]["config_id"]
    seed = protocol["seeds"][0]
    data = load_development(root)
    results = []
    with exclusive(research / "experiment.lock"):
        temporary = research / "stress-inputs"
        temporary.mkdir(exist_ok=True)
        for fold in range(3):
            subset = data[data.fold == fold]
            predictor = load_predictor(
                research / "experiments" / config_id / str(seed) / f"fold-{fold}" / "model",
                device=device,
            )
            for scenario in SCENARIOS:
                paths = []
                for row in subset.itertuples():
                    if scenario == "clean":
                        paths.append(Path(row.image_path))
                        continue
                    destination = (
                        temporary
                        / f"{row.record_id.replace(':', '_')}.{'jpg' if scenario == 'jpeg_q40' else 'png'}"
                    )
                    with Image.open(row.image_path) as source:
                        image = corrupt(source.convert("RGB"), scenario)
                        image.save(
                            destination, quality=40
                        ) if scenario == "jpeg_q40" else image.save(destination)
                    paths.append(destination)
                scores, maps = predictor.predict(paths)
                results.append(
                    {
                        "fold": fold,
                        "scenario": scenario,
                        "metrics": image_metrics(subset.label, scores, subset.record_id),
                    }
                )
                del maps
            del predictor
            gc.collect()
        if not temporary.resolve().is_relative_to(research.resolve()):
            raise ValueError("Stress cleanup escaped its research workspace")
        shutil.rmtree(temporary)
    report = {
        "protocol_hash": protocol["protocol_hash"],
        "model_id": config_id,
        "seed": seed,
        "development_only": True,
        "results": results,
        "meaning": "Synthetic corruption sensitivity; no factory robustness or model-reselection claim.",
    }
    write_json(target, report)
    return report
