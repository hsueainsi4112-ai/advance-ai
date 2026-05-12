import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.utils.paths import artefact_path

_logger = logging.getLogger(__name__)

_model: Any = None
_scalers: dict[int, Any] = {}
_qualifying: set[int] = set()
_loaded: bool = False
_INPUT_WINDOW = 90
_OUTPUT_HORIZON = 7
_N_FEATURES = 3
_TOP_N_SHAP = 10


def load_artefacts() -> None:
    global _model, _scalers, _qualifying, _loaded
    if _loaded:
        return

    from tensorflow import keras

    model_path = artefact_path("data", "Task 1B", "models", "lstm_forecast.keras")
    scalers_path = artefact_path("data", "Task 1B", "processed", "scalers.joblib")
    qualifying_path = artefact_path("data", "Task 1B", "processed", "qualifying_products.json")

    _logger.info("forecasting: loading lstm_forecast.keras")
    _model = keras.models.load_model(str(model_path), compile=False)
    _logger.info("forecasting: loading scalers.joblib")
    _scalers = joblib.load(scalers_path)
    _logger.info("forecasting: loading qualifying_products.json")
    with open(qualifying_path, "r") as fh:
        _qualifying = set(json.load(fh))

    _loaded = True
    _logger.info(
        "forecasting: ready (products=%d, scalers=%d)",
        len(_qualifying),
        len(_scalers),
    )


def reload(new_path: Path) -> None:
    global _model
    from tensorflow import keras

    _model = keras.models.load_model(str(new_path), compile=False)


def model_version() -> str:
    from src.model_management import service as mm

    return mm.version_for("task1b_lstm")


def _build_features(daily_counts: list[int], window_end: date, scaler) -> np.ndarray:
    counts = np.array(daily_counts, dtype=np.float32).reshape(-1, 1)
    counts_norm = scaler.transform(counts).flatten()
    days = [window_end - timedelta(days=_INPUT_WINDOW - 1 - i) for i in range(_INPUT_WINDOW)]
    dow = np.array([d.weekday() for d in days], dtype=np.float32)
    radians = 2.0 * np.pi * dow / 7.0
    sin_dow = np.sin(radians)
    cos_dow = np.cos(radians)
    return np.stack([counts_norm, sin_dow, cos_dow], axis=1)


def _shap_top_days(
    perturbed_preds: np.ndarray,
    base_pred_norm: np.ndarray,
    window_end: date,
) -> list[dict]:
    attributions = np.abs(perturbed_preds.mean(axis=1) - float(base_pred_norm.mean()))
    top = np.argsort(-attributions)[:_TOP_N_SHAP]
    return [
        {
            "day_index": int(idx),
            "date": (window_end - timedelta(days=_INPUT_WINDOW - 1 - int(idx))).isoformat(),
            "attribution": float(attributions[int(idx)]),
        }
        for idx in top
    ]


def get_forecast(
    instacart_product_id: int,
    daily_counts: list[int],
    window_end_date: str,
) -> dict:
    if instacart_product_id not in _qualifying:
        raise LookupError(f"product_id {instacart_product_id} not in qualifying products")
    if len(daily_counts) != _INPUT_WINDOW:
        raise ValueError(f"daily_counts must have exactly {_INPUT_WINDOW} entries")
    try:
        window_end = date.fromisoformat(window_end_date)
    except ValueError as err:
        raise ValueError(f"window_end_date must be ISO-8601 (YYYY-MM-DD): {err}") from err

    scaler = _scalers[instacart_product_id]
    features = _build_features(daily_counts, window_end, scaler)

    perturbed = np.tile(features, (_INPUT_WINDOW, 1, 1))
    diag = np.arange(_INPUT_WINDOW)
    perturbed[diag, diag, 0] = 0.0
    batch = np.concatenate([features[np.newaxis, ...], perturbed], axis=0)
    all_preds = _model.predict(batch, verbose=0)
    base_pred_norm = all_preds[0]
    perturbed_preds = all_preds[1:]

    predicted_counts = (
        scaler.inverse_transform(base_pred_norm.reshape(-1, 1)).flatten().round().astype(int)
    )
    forecast_dates = [
        (window_end + timedelta(days=i + 1)).isoformat() for i in range(_OUTPUT_HORIZON)
    ]

    top_days = _shap_top_days(perturbed_preds, base_pred_norm, window_end)
    explanation = (
        f"Forecast is most sensitive to count perturbations on {len(top_days)} "
        f"input days within the 90-day window. Larger attribution magnitudes "
        f"indicate days whose historical counts most strongly influence the "
        f"{_OUTPUT_HORIZON}-day prediction."
    )

    return {
        "product_id": instacart_product_id,
        "forecast": {
            "dates": forecast_dates,
            "predicted_counts": [int(c) for c in predicted_counts],
        },
        "shap": {
            "top_days": top_days,
            "explanation": explanation,
        },
        "input_window_days": _INPUT_WINDOW,
        "output_horizon_days": _OUTPUT_HORIZON,
    }
