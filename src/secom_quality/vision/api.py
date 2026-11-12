"""Optional vision routes; readiness is independent of the released sensor study."""

import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from secom_quality.common import read_json
from secom_quality.inference import ApplicationError

from .inference import MAX_BYTES, export_csv, load_active, process_batch
from .storage import VisionStore


def create_router(root: Path) -> APIRouter:
    router = APIRouter(prefix="/api/v2/vision", tags=["visual inspection"])

    @router.get("/progress")
    def progress():
        base = root / "vision"
        state_path = base / "runtime/run.json"
        state = read_json(state_path) if state_path.exists() else {"state": "not_started"}
        result = {key: state.get(key) for key in ("state", "phase", "updated_at", "protocol_hash")}
        result["process_alive"] = None
        if state.get("pid"):
            try:
                import psutil

                result["process_alive"] = psutil.pid_exists(state["pid"])
            except ImportError:
                pass
        experiments = base / "research/experiments"
        fits = [
            read_json(path)
            for path in experiments.rglob("result.json")
            if ".staging" not in str(path)
        ]
        result["completed_fits"] = sum(fit.get("status") == "completed" for fit in fits)
        result["failed_fits"] = sum(fit.get("status") == "failed" for fit in fits)
        active = sorted(
            experiments.rglob("fit-request.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        active = [path for path in active if path.parent.name.endswith(".staging")]
        if active:
            request = read_json(active[0])
            result["active_fit"] = {
                "config_id": request["config"]["config_id"],
                "seed": request["seed"],
                "fold": request["fold"],
            }
        latest = sorted(
            experiments.rglob("progress.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if latest and (not active or latest[0].is_relative_to(active[0].parent)):
            try:
                current = read_json(latest[0])
                result["training"] = {
                    key: current.get(key) for key in ("config_id", "epoch", "epochs", "seconds")
                }
            except (ValueError, OSError):
                pass  # An epoch writer may be replacing this optional progress snapshot.
        result["final_evaluated"] = (base / "research/final/report.json").exists()
        return result

    async def validate_form(request: Request, allowed: set[str]) -> None:
        form = await request.form()
        names = [name for name, _ in form.multi_items()]
        unknown = set(names) - allowed
        if unknown or len(names) != len(set(names)):
            raise ApplicationError(
                "request_fields",
                "Unknown or duplicate form fields are not accepted. Do not include outcome fields.",
            )

    @router.get("/model")
    @router.get("/status")
    def model():
        try:
            bundle = load_active(root)
            return {
                "ready": True,
                "manifest": bundle["manifest"],
                "report": bundle.get("report", {}),
                "score_kind": "uncalibrated",
                "calibration_status": "not_validated",
            }
        except ApplicationError:
            return JSONResponse(
                status_code=503,
                content={
                    "ready": False,
                    "message": "Vision research is not released. A verified trained bundle is required.",
                    "score_kind": "uncalibrated",
                    "calibration_status": "not_validated",
                },
            )

    @router.post("/predictions")
    async def prediction(
        request: Request,
        file: Annotated[UploadFile, File()],
        record_id: Annotated[str | None, Form()] = None,
    ):
        await validate_form(request, {"file", "record_id"})
        filename = file.filename or "image.png"
        if Path(filename).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ApplicationError("image_format", "Single predictions accept PNG/JPEG images.")
        if record_id is not None:
            if Path(record_id).name != record_id or any(c in record_id for c in "\\/:"):
                raise ApplicationError(
                    "record_id", "Use a record identity without path separators."
                )
            filename = record_id + Path(filename).suffix
        result = process_batch(
            root, await file.read(MAX_BYTES + 1), filename, "single-" + uuid.uuid4().hex, 0.1
        )
        return result["predictions"][0]

    @router.post("/batches")
    async def batch(
        request: Request,
        file: Annotated[UploadFile, File()],
        client_request_id: Annotated[str, Form()],
        capacity: Annotated[float, Form()] = 0.1,
    ):
        await validate_form(request, {"file", "client_request_id", "capacity"})
        return process_batch(
            root,
            await file.read(MAX_BYTES + 1),
            file.filename or "batch.zip",
            client_request_id,
            capacity,
        )

    @router.get("/batches")
    def batches():
        return {"batches": VisionStore(root).list()}

    @router.get("/batches/{batch_id}")
    def get_batch(batch_id: str):
        return VisionStore(root).get(batch_id)

    @router.get("/batches/{batch_id}/predictions")
    def predictions(batch_id: str, offset: int = 0, limit: int = 25):
        if offset < 0 or not 1 <= limit <= 100:
            raise ApplicationError(
                "pagination", "Use nonnegative offset and a limit from 1 to 100."
            )
        result = VisionStore(root).get(batch_id)
        return {
            "predictions": result["predictions"][offset : offset + limit],
            "total": result["row_count"],
        }

    @router.get("/batches/{batch_id}/export")
    def export(batch_id: str):
        return Response(
            export_csv(VisionStore(root).get(batch_id)),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="visual_inspection_queue.csv"'},
        )

    @router.get("/batches/{batch_id}/assets/{asset_id}/{kind}")
    def asset(batch_id: str, asset_id: str, kind: Literal["image", "overlay"]):
        result = VisionStore(root).get(batch_id)
        if not asset_id.isascii() or not asset_id.isdigit() or len(asset_id) != 3:
            raise ApplicationError("not_found", "Image asset not found.", 404)
        index = int(asset_id)
        if index >= result["row_count"]:
            raise ApplicationError("not_found", "Image asset not found.", 404)
        suffix = "-overlay" if kind == "overlay" else ""
        path = root / "vision" / "assets" / batch_id / f"{asset_id}{suffix}.png"
        if not path.is_file():
            raise ApplicationError("not_found", "Image asset not found.", 404)
        return FileResponse(
            path, media_type="image/png", headers={"X-Content-Type-Options": "nosniff"}
        )

    return router
