import io
import os

import pytest
from PIL import Image


@pytest.fixture(scope="session", autouse=True)
def _set_env():
    os.environ.setdefault(
        "AI_DATA_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    )
    os.environ.setdefault("AI_ADMIN_KEY", "test-admin-key")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from src.api.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    from src.api.routes.grading import _rate_buckets

    _rate_buckets.clear()
    yield


@pytest.fixture(scope="session")
def class_names() -> set[str]:
    import json

    from src.utils.paths import artefact_path

    with open(artefact_path("data", "Task 2", "processed", "class_info.json")) as fh:
        return set(json.load(fh)["class_names"])


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (300, 300), color=(200, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def oversized_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (5000, 5000), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()
