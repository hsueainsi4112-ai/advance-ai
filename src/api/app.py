import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pythonjsonlogger.json import JsonFormatter

from src.api.errors import APIError, api_error_handler, validation_error_handler
from src.api.routes.admin import router as admin_router
from src.api.routes.forecasting import router as forecasting_router
from src.api.routes.grading import router as grading_router
from src.api.routes.recommendations import router as recommendations_router
from src.model_management import service as model_management


def _configure_access_logger() -> logging.Logger:
    logger = logging.getLogger("brfn.access")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    logging.getLogger("uvicorn.access").disabled = True
    return logger


_access_logger = _configure_access_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.quality_assessment import service as quality_service

    quality_service.load_artefacts()
    yield


app = FastAPI(
    title="Bristol Regional Food Network AI Service",
    description="AI-powered features for the digital marketplace",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)


@app.middleware("http")
async def _access_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    _access_logger.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": duration_ms,
        },
    )
    response.headers["x-request-id"] = request_id
    return response

app.include_router(grading_router)
app.include_router(recommendations_router)
app.include_router(forecasting_router)
app.include_router(admin_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {
        "status": "ready",
        "models_loaded": model_management.active_versions(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
