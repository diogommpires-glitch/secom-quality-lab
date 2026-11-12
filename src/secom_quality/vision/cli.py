"""Vision lifecycle; all commands use the same scientific and serving services."""

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="Independent, frozen industrial vision study.")


def show(value):
    typer.echo(json.dumps(value, indent=2, default=str, allow_nan=False))


@app.command("doctor")
def doctor():
    import platform

    from .bundle import dependency_versions

    result = {"python": platform.python_version(), "dependencies": dependency_versions()}
    import torch

    result["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        x = torch.ones((32, 32), device="cuda")
        result.update(
            gpu=torch.cuda.get_device_name(),
            cuda=torch.version.cuda,
            smoke_test=bool(torch.isfinite(x @ x).all()),
        )
    show(result)


@app.command("status")
def status(ctx: typer.Context):
    from secom_quality.common import read_json

    base = Path(ctx.obj) / "vision"
    path = base / "runtime/run.json"
    result = read_json(path) if path.exists() else {"state": "not_started"}
    progress = sorted(
        (base / "research/experiments").rglob("progress.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if progress:
        result["latest_training"] = read_json(progress[0])
    completed = list((base / "research/experiments").rglob("result.json"))
    result["recorded_fits"] = len(completed)
    result["failed_fits"] = sum(read_json(p).get("status") == "failed" for p in completed)
    result["final_access_recorded"] = (base / "research/final-access.json").exists()
    show(result)


@app.command("data-fetch")
def data_fetch(ctx: typer.Context):
    from .data import fetch_data

    show(fetch_data(ctx.obj))


@app.command("data-audit")
def data_audit(ctx: typer.Context):
    from .data import audit_data

    show(audit_data(ctx.obj))


@app.command("protocol-freeze")
def protocol_freeze(ctx: typer.Context):
    from .protocol import freeze_protocol

    show(freeze_protocol(ctx.obj))


@app.command("experiment-run")
def experiment_run(ctx: typer.Context, device: str = "cuda", config_id: str | None = None):
    from .study import run_experiments

    result = run_experiments(ctx.obj, device, config_id)
    show(
        {
            "completed": True,
            "results": str(Path(ctx.obj) / "vision/research"),
            "complete_development": isinstance(result, dict),
        }
    )


@app.command("experiment-compare")
def experiment_compare(ctx: typer.Context):
    from .study import comparison

    show([{k: v for k, v in row.items() if k != "results"} for row in comparison(ctx.obj)])


@app.command("selection-freeze")
def selection_freeze(ctx: typer.Context):
    from .study import freeze_selection

    show(freeze_selection(ctx.obj))


@app.command("final-evaluate")
def final_evaluate(ctx: typer.Context, device: str = "cuda"):
    from .study import evaluate_final

    show(evaluate_final(ctx.obj, device))


@app.command("bundle-build")
def bundle_build(ctx: typer.Context):
    from .bundle import build_bundle

    show(build_bundle(ctx.obj))


@app.command("bundle-verify")
def bundle_verify(ctx: typer.Context):
    from .bundle import load_verified_bundle

    show(load_verified_bundle(Path(ctx.obj) / "vision/bundles/main")["manifest"])


@app.command("predict")
def predict(
    ctx: typer.Context,
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    request_id: Annotated[str, typer.Option()],
    capacity: float = 0.1,
):
    """Validate, score and persist images using the identical API service."""
    from .inference import MAX_BYTES, process_batch

    if input_path.stat().st_size > MAX_BYTES:
        raise typer.BadParameter("Image input exceeds the 20 MiB limit")
    show(
        process_batch(Path(ctx.obj), input_path.read_bytes(), input_path.name, request_id, capacity)
    )


@app.command("report-build")
def report_build(ctx: typer.Context):
    from .report import build_report

    show(build_report(ctx.obj))


@app.command("stress")
def stress(ctx: typer.Context, device: str = "cuda"):
    from .robustness import run_stress

    show(run_stress(ctx.obj, device))


@app.command("ad2-audit")
def ad2_audit(ctx: typer.Context, archive: Annotated[Path, typer.Option(exists=True)]):
    from .ad2 import audit_ad2

    show(audit_ad2(ctx.obj, archive))


@app.command("ad2-freeze")
def ad2_freeze(ctx: typer.Context):
    from .ad2 import freeze_ad2

    show(freeze_ad2(ctx.obj))


@app.command("ad2-run")
def ad2_run(ctx: typer.Context, device: str = "cuda"):
    from .ad2 import run_ad2

    show(run_ad2(ctx.obj, device))


@app.command("ad2-export")
def ad2_export(ctx: typer.Context, device: str = "cuda"):
    from .ad2 import export_ad2

    show(export_ad2(ctx.obj, device))
