"""Explicit synthetic browser fixture. Never use this server as a research demonstration.

The business routes and persistence are the production application. Only the bundle
loader is replaced, using an obviously named synthetic predictor without learned weights.
"""

import argparse
import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import uvicorn
from fastapi.responses import Response
from fastapi.routing import APIRoute
from PIL import Image

from secom_quality.api import create_app
from secom_quality.vision import api as vision_api
from secom_quality.vision import inference


class SyntheticBrowserPredictor:
    """Deterministic synthetic scores; no scientific performance interpretation."""

    def predict(self, paths: list[Path]):
        scores, maps = [], []
        for path in paths:
            with Image.open(path) as image:
                values = np.asarray(image, dtype=np.float32)
            scores.append(float(values[:, :, 0].mean() / 255))
            maps.append(values[:, :, 0] / 255)
        return np.asarray(scores), maps


def synthetic_bundle(root: Path):
    return {
        "manifest": {
            "bundle_id": "synthetic-browser-fixture",
            "model_id": "synthetic-browser-fixture-not-trained",
            "family": "SYNTHETIC TEST FIXTURE ONLY",
            "protocol_hash": "synthetic-no-research-protocol",
        },
        "predictor": SyntheticBrowserPredictor(),
        "report": {"fixture_only": True, "scientific_performance": None},
    }


def fixture_zip():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for index in range(10):
            image = np.zeros((32, 24, 3), dtype=np.uint8)
            image[:, :, 0] = np.arange(24)[None, :] + index * 20
            image[:, :, 1:] = 100
            stream = io.BytesIO()
            Image.fromarray(image).save(stream, format="PNG")
            archive.writestr(f"synthetic-item-{index:02d}.png", stream.getvalue())
    return Response(output.getvalue(), media_type="application/zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--synthetic-ready", action="store_true")
    arguments = parser.parse_args()
    if arguments.synthetic_ready:
        inference.load_active = synthetic_bundle
        vision_api.load_active = synthetic_bundle
    with tempfile.TemporaryDirectory(prefix="secom-synthetic-browser-") as directory:
        application = create_app(Path(directory))
        application.router.routes.insert(0, APIRoute("/__test__/inputs.zip", fixture_zip))
        uvicorn.run(application, host="127.0.0.1", port=arguments.port, log_level="warning")
