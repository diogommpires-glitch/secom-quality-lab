"""Local HTTP adapters for the shared inference and replay services."""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException

from .contracts import FEATURE_NAMES, LIMITATION, SCHEMA_VERSION, LoadedBundle
from .inference import (
    MAX_UPLOAD_BYTES,
    ApplicationError,
    csv_template,
    model_metadata,
    parse_csv,
    predictions_csv,
    process_batch,
    record_from_values,
    records_csv,
    score_records,
    validate_record,
)
from .monitoring import ReplayService, public_stream
from .storage import Store

logger = logging.getLogger(__name__)


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    scenario: str = "baseline"


class AdvanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_cursor: int = Field(ge=0)
    steps: int = 20


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    record_id: str
    observed_at: str | None = None
    sensors: dict[str, float | None]


class PredictionResponse(BaseModel):
    prediction_id: str
    record_id: str
    risk_score: float | None
    calibrated_probability: None = None
    score_kind: Literal["uncalibrated"]
    calibration_status: Literal["not_validated"] = "not_validated"
    feature_schema_version: str = SCHEMA_VERSION
    quality_status: Literal["valid", "warning", "abstain"]
    quality_reason_codes: list[str]
    missingness: float
    range_violation_count: int
    input_fingerprint: str
    bundle_id: str
    model_id: str
    rank: int | None = None
    review_recommended: bool = False


class BatchResponse(BaseModel):
    batch_id: str
    row_count: int
    bundle_id: str
    model_id: str
    validation_issues: list[dict[str, Any]]
    eligible_count: int
    review_capacity: int
    abstain_count: int
    predictions: list[PredictionResponse]
    queue_policy_id: str
    replayed: bool = False


def create_app(root: Path, bundle_path: Path | None = None) -> FastAPI:
    root = Path(root).resolve()
    store = Store(root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from .artifacts import load_bundle

        app.state.bundle = None
        app.state.bundle_error = None
        app.state.replay = None
        try:
            app.state.bundle = load_bundle(bundle_path or root / "bundles" / "main")
            if app.state.bundle.manifest["purpose"] not in {"main", "scientific_release"}:
                raise ValueError("The prediction API requires the scientific main bundle.")
        except Exception:
            logger.exception("Could not load the verified scientific bundle")
            app.state.bundle = None
            app.state.bundle_error = "The verified scientific bundle is unavailable."
        try:
            app.state.replay = ReplayService(root)
        except Exception:
            logger.exception("Could not load the isolated development-only replay")
        yield

    app = FastAPI(title="SECOM Quality Lab", version="1.0.0", lifespan=lifespan)
    app.state.root = root

    def error_response(request: Request, status: int, detail: dict[str, Any]) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status,
            content={"detail": {**detail, "request_id": request_id}},
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(ApplicationError)
    async def expected_error(request: Request, exc: ApplicationError):
        return error_response(request, exc.status, exc.detail())

    @app.exception_handler(RequestValidationError)
    async def request_error(request: Request, exc: RequestValidationError):
        issues = [
            {
                "field": ".".join(map(str, item["loc"])),
                "reason_code": item["type"],
                "message": item["msg"],
            }
            for item in exc.errors()
        ]
        return error_response(
            request,
            422,
            {
                "code": "request_validation",
                "message": "Request validation failed.",
                "issues": issues,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return error_response(
            request,
            exc.status_code,
            {"code": "http_error", "message": str(exc.detail), "issues": []},
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.error(
            "Request %s failed",
            getattr(request.state, "request_id", "unknown"),
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return error_response(
            request,
            500,
            {
                "code": "internal_error",
                "message": "The request could not be completed. Inspect the local server log using its request ID.",
                "issues": [],
            },
        )

    @app.middleware("http")
    async def body_limit(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        # Multipart overhead has its own small allowance; the file itself has a strict 20 MiB cap.
        length = request.headers.get("content-length")
        if length is not None:
            try:
                too_large = int(length) > MAX_UPLOAD_BYTES + 1024 * 1024
            except ValueError:
                too_large = True
            if too_large:
                return error_response(
                    request,
                    413,
                    {
                        "code": "upload_too_large",
                        "message": "Request exceeds the upload limit.",
                        "issues": [],
                    },
                )
        if request.method in {"POST", "PUT", "PATCH"}:
            chunks = []
            received = 0
            async for chunk in request.stream():
                received += len(chunk)
                if received > MAX_UPLOAD_BYTES + 1024 * 1024:
                    return error_response(
                        request,
                        413,
                        {
                            "code": "upload_too_large",
                            "message": "Request exceeds the upload limit.",
                            "issues": [],
                        },
                    )
                chunks.append(chunk)
            request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def bundle() -> LoadedBundle:
        current = getattr(app.state, "bundle", None)
        if current is None:
            raise ApplicationError(
                "model_unavailable", "The verified scientific bundle is unavailable.", 503
            )
        return current

    def replay() -> ReplayService:
        current = getattr(app.state, "replay", None)
        if current is None:
            raise ApplicationError(
                "demo_unavailable",
                "The development-only demo bundle or inputs are unavailable.",
                503,
            )
        return current

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        current = bundle()
        with store.connection() as connection:
            connection.execute("SELECT 1")
        return {
            "status": "ready",
            "bundle_id": current.manifest["bundle_id"],
            "monitoring_ready": getattr(app.state, "replay", None) is not None,
        }

    @app.get("/api/v1/model")
    def model():
        return model_metadata(bundle())

    @app.get("/api/v1/feature-schema")
    def schema():
        return {
            "schema_version": SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "n_features": len(FEATURE_NAMES),
            "missing_value": "null in JSON; empty or NaN in CSV",
            "optional_metadata": ["observed_at"],
            "required_metadata": ["record_id"],
        }

    @app.get("/api/v1/feature-schema/template")
    def template():
        return Response(
            csv_template(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="secom_input_template.csv"'},
        )

    @app.get("/api/v1/data-audit")
    def data_audit():
        return bundle().report.get("audit", {})

    @app.get("/api/v1/experiments")
    def experiments():
        report = bundle().report
        return {
            key: report.get(key)
            for key in ("fold_results", "selection", "final_metrics", "final_status")
        } | {"limitation": LIMITATION}

    @app.get("/api/v1/study-report")
    def study_report():
        return bundle().report

    @app.post("/api/v1/predictions", response_model=PredictionResponse)
    def predict(record: Annotated[PredictionRequest, Body()]):
        return score_records([validate_record(record.model_dump())], bundle())[0]

    @app.post("/api/v1/batches", response_model=BatchResponse)
    async def create_batch(
        file: Annotated[UploadFile, File()], client_request_id: Annotated[str, Form()]
    ):
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        await file.close()
        records = parse_csv(content)
        return process_batch(root, records, client_request_id, bundle())

    @app.get("/api/v1/batches")
    def batches():
        return {"batches": store.list_batches()}

    @app.get("/api/v1/batches/{batch_id}", response_model=BatchResponse)
    def get_batch(batch_id: str):
        return store.get_batch(batch_id)

    @app.get("/api/v1/batches/{batch_id}/predictions")
    def batch_predictions(batch_id: str, offset: int = 0, limit: int = 100):
        if offset < 0 or not 1 <= limit <= 2000:
            raise ApplicationError(
                "pagination", "Use a nonnegative offset and a limit from 1 to 2,000."
            )
        result = store.get_batch(batch_id)
        return {
            "predictions": result["predictions"][offset : offset + limit],
            "total": result["row_count"],
        }

    @app.get("/api/v1/batches/{batch_id}/export")
    def export(batch_id: str):
        return Response(
            predictions_csv(store.get_batch(batch_id)),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="inspection_queue.csv"'},
        )

    @app.get("/api/v1/demo/inputs")
    def demo_inputs():
        service = replay()
        records = [
            record_from_values(service.dataset.ids[i], service.dataset.X[i])
            for i in range(538, 558)
        ]
        return Response(
            records_csv(records),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="approved_development_demo.csv"'},
        )

    @app.post("/api/v1/demo/replays")
    def create_replay(request: ReplayRequest):
        return replay().create(request.scenario)

    @app.post("/api/v1/demo/replays/{stream_id}/advance")
    def advance_replay(stream_id: str, request: AdvanceRequest):
        return replay().advance(stream_id, request.expected_cursor, request.steps)

    @app.post("/api/v1/demo/replays/{stream_id}/pause")
    def pause_replay(stream_id: str):
        return replay().pause(stream_id)

    @app.get("/api/v1/monitoring/streams")
    def streams():
        return {"streams": [public_stream(state) for state in store.list_streams()]}

    @app.get("/api/v1/monitoring/streams/{stream_id}")
    def stream(stream_id: str):
        return public_stream(store.get_stream(stream_id))

    @app.get("/api/v1/monitoring/streams/{stream_id}/windows")
    def windows(stream_id: str):
        return {"windows": store.get_stream(stream_id)["windows"]}

    from .vision.api import create_router as create_vision_router

    app.include_router(create_vision_router(root))

    # API routes are registered first. Unknown API URLs must never turn into SPA HTML.
    @app.api_route(
        "/api/{unknown:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    def unknown_api(unknown: str):
        raise ApplicationError("not_found", "API route not found.", 404)

    packaged = Path(__file__).parent / "web"
    source_frontend = Path(__file__).parents[2] / "frontend" / "dist"
    web = packaged if (packaged / "index.html").is_file() else source_frontend
    if (web / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=web / "assets"), name="assets")

    @app.get("/{route:path}", include_in_schema=False)
    def frontend(route: str):
        if not (web / "index.html").is_file():
            return JSONResponse(
                status_code=503,
                content={
                    "detail": {
                        "code": "frontend_unbuilt",
                        "message": "Build the local React frontend first.",
                        "issues": [],
                    }
                },
            )
        return FileResponse(web / "index.html")

    return app
