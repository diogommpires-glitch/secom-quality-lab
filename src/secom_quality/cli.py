"""Offline study lifecycle and local application commands."""

import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Annotated, Any

import typer

from secom_quality.contracts import LIMITATION, workspace_path
from secom_quality.vision.cli import app as vision_app

app = typer.Typer(
    no_args_is_help=True, help="SECOM Quality Lab: retrospective inspection research."
)
app.add_typer(vision_app, name="vision")
data_app = typer.Typer(no_args_is_help=True, help="Pinned source acquisition and structural audit.")
protocol_app = typer.Typer(no_args_is_help=True)
experiment_app = typer.Typer(no_args_is_help=True)
selection_app = typer.Typer(no_args_is_help=True)
final_app = typer.Typer(no_args_is_help=True)
report_app = typer.Typer(no_args_is_help=True)
bundle_app = typer.Typer(no_args_is_help=True)
for name, subapp in (
    ("data", data_app),
    ("protocol", protocol_app),
    ("experiment", experiment_app),
    ("selection", selection_app),
    ("final", final_app),
    ("report", report_app),
    ("bundle", bundle_app),
):
    app.add_typer(subapp, name=name)


@app.callback()
def main(
    ctx: typer.Context,
    workspace: Annotated[
        Path | None,
        typer.Option(envvar="SECOM_WORKSPACE", help="Mutable data and artifacts directory."),
    ] = None,
) -> None:
    ctx.obj = workspace_path(workspace)


def show(result: Any) -> None:
    typer.echo(json.dumps(result, indent=2, default=str, allow_nan=False))


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Print environment and installation checks without changing data."""
    packages = ("secom-quality-lab", "numpy", "pandas", "scipy", "scikit-learn", "fastapi", "skops")
    show(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "workspace": str(ctx.obj),
            "dependencies": {name: importlib.metadata.version(name) for name in packages},
            "frontend_packaged": (Path(__file__).parent / "web" / "index.html").exists(),
            "limitation": LIMITATION,
        }
    )


@data_app.command("fetch")
def data_fetch(ctx: typer.Context) -> None:
    from secom_quality.data import fetch_data

    show(fetch_data(ctx.obj))


@data_app.command("verify")
def data_verify(ctx: typer.Context) -> None:
    from secom_quality.data import verify_data

    show(verify_data(ctx.obj))


@data_app.command("audit")
def data_audit(ctx: typer.Context) -> None:
    from secom_quality.data import audit_data

    result = audit_data(ctx.obj)
    show(
        {
            key: value
            for key, value in result.items()
            if not key.startswith("protected_") and key != "missingness_by_feature"
        }
    )


@protocol_app.command("freeze")
def protocol_freeze(ctx: typer.Context) -> None:
    from secom_quality.study import freeze_protocol

    result = freeze_protocol(ctx.obj)
    show(
        {
            "protocol_hash": result["protocol_hash"],
            "folds": result["folds"],
            "candidate_count": len(result["candidate_configs"]),
            "selection_rule": result["selection_rule"],
        }
    )


@experiment_app.command("run")
def experiment_run(
    ctx: typer.Context,
    workers: Annotated[
        int, typer.Option(min=1, max=4, help="Maximum parallel training workers.")
    ] = 4,
) -> None:
    from secom_quality.study import run_experiments

    result = run_experiments(ctx.obj, max_workers=workers)
    show(
        {
            key: result[key]
            for key in (
                "protocol_hash",
                "planned_fits",
                "completed_fits",
                "invalid_fits",
                "selection_scope",
            )
        }
    )


@experiment_app.command("compare")
def experiment_compare(ctx: typer.Context) -> None:
    from secom_quality.study import compare_experiments

    show(compare_experiments(ctx.obj))


@selection_app.command("freeze")
def selection_freeze(ctx: typer.Context) -> None:
    from secom_quality.study import freeze_selection

    show(freeze_selection(ctx.obj))


@final_app.command("evaluate")
def final_evaluate(ctx: typer.Context) -> None:
    """Evaluate the frozen selection once, with 2,000 paired day bootstrap resamples."""
    from secom_quality.study import evaluate_final

    result = evaluate_final(ctx.obj, bootstrap_samples=2000)
    show(
        {
            "final_status": result["final_status"],
            "final_metrics": result["final_metrics"],
            "interpretation": result["interpretation"],
            "report_command": "secom report build",
        }
    )


@report_app.command("build")
def report_build(ctx: typer.Context) -> None:
    from secom_quality.study import build_report

    show({"report": str(build_report(ctx.obj))})


@bundle_app.command("build")
def bundle_build(ctx: typer.Context) -> None:
    from secom_quality.artifacts import build_bundles

    results = build_bundles(ctx.obj)
    show(
        {
            name: {key: value.get(key) for key in ("bundle_id", "purpose", "status")}
            for name, value in results.items()
        }
    )


@bundle_app.command("verify")
def bundle_verify(ctx: typer.Context, path: Annotated[Path | None, typer.Option()] = None) -> None:
    from secom_quality.artifacts import load_bundle

    bundle = load_bundle(path or ctx.obj / "bundles" / "main")
    show(
        {
            "verified": True,
            "path": str(bundle.path),
            "bundle_id": bundle.manifest["bundle_id"],
            "model_id": bundle.manifest["model_id"],
            "purpose": bundle.manifest["purpose"],
        }
    )


@app.command()
def predict(
    ctx: typer.Context,
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    bundle: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Use the same validation and inference path as the HTTP API."""
    from secom_quality.inference import predict_csv

    show(predict_csv(ctx.obj, input_csv, bundle_path=bundle, capacity=0.1))


@app.command()
def replay(
    ctx: typer.Context,
    scenario: Annotated[str, typer.Option(help="Approved demonstration scenario.")] = "baseline",
    steps: Annotated[int, typer.Option(min=1, max=40)] = 1,
) -> None:
    """Run virtual replay through the same application services as the console."""
    from secom_quality.monitoring import ReplayService

    service = ReplayService(ctx.obj)
    result = service.create(scenario)
    stream_id = str(result["stream_id"])
    for _ in range(steps):
        result = service.advance(stream_id, expected_cursor=result.get("cursor", 0))
    show(result)


@app.command()
def serve(
    ctx: typer.Context,
    port: Annotated[int, typer.Option(min=1024, max=65535)] = 8765,
    bundle: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Serve the packaged console and API on loopback; no frontend server required."""
    import uvicorn

    from secom_quality.api import create_app

    uvicorn.run(create_app(ctx.obj, bundle_path=bundle), host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
