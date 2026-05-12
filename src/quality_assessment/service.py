import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.utils.paths import ai_data_root, artefact_path

_logger = logging.getLogger(__name__)

_grading: Any = None
_loaded: bool = False

_GRADE_TO_ACTION = {
    "A": "sell_full_price",
    "B": "sell_discounted",
    "C": "flag_for_surplus",
}


def load_artefacts() -> None:
    global _grading, _loaded
    if _loaded:
        return
    ensemble_dir = str(artefact_path("data", "Task 2", "ensemble"))
    if ensemble_dir not in sys.path:
        sys.path.insert(0, ensemble_dir)
    import grading  # type: ignore[import-not-found]

    grading.configure(drive_root=str(ai_data_root()))
    grading._initialise()
    _grading = grading
    _loaded = True
    _logger.info("quality_assessment: grading module loaded")


def reload_resnet(path: str) -> None:
    _grading.configure(drive_root=str(ai_data_root()), resnet_path=path)
    _grading._initialise()


def reload_mobilenet(path: str) -> None:
    _grading.configure(drive_root=str(ai_data_root()), mobilenet_path=path)
    _grading._initialise()


def model_version() -> str:
    from src.model_management import service as mm

    return mm.version_for("task2_resnet")


def grade(image_bytes: bytes) -> dict:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        tmp.write(image_bytes)
        tmp.close()
        result = _grading.grade_image(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    result["recommended_action"] = _GRADE_TO_ACTION[result["grade"]]
    return result


def compute_heatmap(image_bytes: bytes, predicted_class: str) -> dict:
    from src.quality_assessment import gradcam

    return gradcam.compute_heatmap(
        image_bytes,
        predicted_class,
        resnet=_grading._resnet_model,
        class_index=_grading._class_to_index[predicted_class],
    )
