import io
import time
import uuid
from collections import defaultdict, deque

from fastapi import APIRouter, File, Header, UploadFile
from PIL import Image, UnidentifiedImageError

from src.api.errors import APIError
from src.api.schemas import EnsembleWeights, GradCamResponse, GradeImageResponse
from src.quality_assessment import service

router = APIRouter(prefix="/api/v1")

_ALLOWED_MIME = {"image/jpeg", "image/png"}
_MAX_BYTES = 10 * 1024 * 1024
_MAX_DIM = 4096
_CACHE_TTL_SECONDS = 15 * 60

_RATE_LIMIT_WINDOW = 60
_RATE_LIMIT_MAX = 10

_cache: dict[str, tuple[float, bytes, str]] = {}
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


def _evict_expired() -> None:
    now = time.time()
    for rid in [k for k, (t, _, _) in _cache.items() if now - t > _CACHE_TTL_SECONDS]:
        _cache.pop(rid, None)


def _enforce_rate_limit(user_key: str) -> None:
    now = time.time()
    bucket = _rate_buckets[user_key]
    while bucket and now - bucket[0] > _RATE_LIMIT_WINDOW:
        bucket.popleft()
    if len(bucket) >= _RATE_LIMIT_MAX:
        raise APIError(
            status_code=429,
            detail=f"rate limit exceeded ({_RATE_LIMIT_MAX}/{_RATE_LIMIT_WINDOW}s per user)",
            code="rate_limited",
        )
    bucket.append(now)


@router.post("/grade-image", response_model=GradeImageResponse)
async def grade_image(
    file: UploadFile = File(...),
    x_user_id: str | None = Header(None),
) -> GradeImageResponse:
    if not x_user_id:
        raise APIError(400, "X-User-Id header required", code="missing_user_id")
    _enforce_rate_limit(x_user_id)
    if file.content_type not in _ALLOWED_MIME:
        raise APIError(400, "unsupported media type", code="unsupported_media_type")
    contents = await file.read()
    if len(contents) > _MAX_BYTES:
        raise APIError(413, "payload too large", code="payload_too_large")
    try:
        with Image.open(io.BytesIO(contents)) as im:
            width, height = im.size
            image_format = im.format
    except UnidentifiedImageError as err:
        raise APIError(400, "invalid image", code="invalid_image") from err
    if image_format not in {"JPEG", "PNG"}:
        raise APIError(400, f"unsupported image format: {image_format}", code="invalid_image_format")
    if width > _MAX_DIM or height > _MAX_DIM:
        raise APIError(400, "image dimensions exceed 4096x4096", code="image_too_large")

    result = service.grade(contents)
    request_id = str(uuid.uuid4())
    _evict_expired()
    _cache[request_id] = (time.time(), contents, result["predicted_class"])

    return GradeImageResponse(
        predicted_class=result["predicted_class"],
        produce_type=result["produce_type"],
        quality=result["quality"],
        confidence=result["confidence"],
        colour_pct=result["colour_pct"],
        size_pct=result["size_pct"],
        ripeness_pct=result["ripeness_pct"],
        grade=result["grade"],
        ensemble_weights=EnsembleWeights(**result["ensemble_weights"]),
        recommended_action=result["recommended_action"],
        grad_cam_url=f"/api/v1/grad-cam/{request_id}",
        model_version=service.model_version(),
    )


@router.get("/grad-cam/{request_id}", response_model=GradCamResponse)
async def get_grad_cam(request_id: str) -> GradCamResponse:
    _evict_expired()
    entry = _cache.get(request_id)
    if entry is None:
        raise APIError(404, "request_id not found", code="request_id_not_found")
    _, image_bytes, predicted_class = entry
    heatmap = service.compute_heatmap(image_bytes, predicted_class)
    return GradCamResponse(
        heatmap_base64=heatmap["heatmap_base64"],
        predicted_class=heatmap["predicted_class"],
        explanation=heatmap["explanation"],
        model_version=service.model_version(),
    )
