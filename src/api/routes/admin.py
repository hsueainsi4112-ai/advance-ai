import hmac
import os

from fastapi import APIRouter, Depends, File, Form, Header, UploadFile

from src.api.errors import APIError
from src.api.schemas import UploadModelResponse
from src.model_management import service

_EXPECTED_ADMIN_KEY = os.environ["AI_ADMIN_KEY"]

router = APIRouter(prefix="/api/v1/admin")


def _verify_admin_key(x_admin_key: str | None = Header(None)) -> None:
    if x_admin_key is None or not hmac.compare_digest(x_admin_key, _EXPECTED_ADMIN_KEY):
        raise APIError(401, "invalid admin key", code="invalid_admin_key")


@router.post(
    "/upload-model",
    response_model=UploadModelResponse,
    dependencies=[Depends(_verify_admin_key)],
)
async def upload_model(
    task: str = Form(...),
    version: str = Form(...),
    file: UploadFile = File(...),
) -> UploadModelResponse:
    contents = await file.read()
    try:
        result = service.hot_swap(task, version, contents, file.filename or "")
    except ValueError as err:
        raise APIError(400, str(err), code="unknown_task") from err
    return UploadModelResponse(**result)
