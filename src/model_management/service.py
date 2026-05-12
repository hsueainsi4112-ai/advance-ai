import logging
import threading
import time
from pathlib import Path
from typing import Literal

from src.utils.paths import artefact_path

_logger = logging.getLogger(__name__)

TaskName = Literal["task1a_knn", "task1b_lstm", "task2_resnet", "task2_mobilenet"]

_VALID_TASKS: tuple[str, ...] = (
    "task1a_knn",
    "task1b_lstm",
    "task2_resnet",
    "task2_mobilenet",
)

_SWAP_LOCK = threading.Lock()

_active_versions: dict[str, str] = {
    "task1a_knn": "task1a_v1",
    "task1b_lstm": "task1b_v1",
    "task2_resnet": "task2_v1",
    "task2_mobilenet": "task2_v1",
}


def version_for(task: str) -> str:
    return _active_versions.get(task, "unknown")


def active_versions() -> dict[str, str]:
    return dict(_active_versions)


def hot_swap(task: str, version: str, file_bytes: bytes, filename: str) -> dict:
    if task not in _VALID_TASKS:
        raise ValueError(
            f"unknown task: {task!r}; allowed: {list(_VALID_TASKS)}"
        )

    versions_dir = artefact_path("models", "versions")
    versions_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix
    target = versions_dir / f"{task}_{version}{ext}"

    with _SWAP_LOCK:
        if target.exists():
            timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            backup = target.with_suffix(f"{target.suffix}.{timestamp}.bak")
            target.rename(backup)
            _logger.info(
                "model_management: backed up %s -> %s", target.name, backup.name
            )
        target.write_bytes(file_bytes)
        _logger.info(
            "model_management: saved %s (%d bytes)", target, len(file_bytes)
        )
        message = _reload(task, target)
        _active_versions[task] = version

    return {
        "status": "success",
        "task": task,
        "version": version,
        "message": message,
    }


def _reload(task: str, path: Path) -> str:
    if task == "task2_resnet":
        from src.quality_assessment import service as q

        q.reload_resnet(str(path))
        return f"ResNet reloaded from {path.name}"
    if task == "task2_mobilenet":
        from src.quality_assessment import service as q

        q.reload_mobilenet(str(path))
        return f"MobileNet reloaded from {path.name}"
    if task == "task1b_lstm":
        from src.forecasting import service as f

        f.reload(path)
        return f"LSTM reloaded from {path.name}"
    return (
        f"Saved to {path.name}; task1a recommendations are precomputed, "
        "no live reload path"
    )
