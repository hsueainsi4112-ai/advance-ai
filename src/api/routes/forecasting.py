from fastapi import APIRouter

from src.api.errors import APIError
from src.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    ForecastWindow,
    ShapAttribution,
    ShapDay,
)
from src.forecasting import service

router = APIRouter(prefix="/api/v1")


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(body: ForecastRequest) -> ForecastResponse:
    try:
        result = service.get_forecast(
            body.instacart_product_id,
            body.daily_counts,
            body.window_end_date,
        )
    except LookupError as err:
        raise APIError(404, str(err), code="product_not_qualifying") from err
    except ValueError as err:
        raise APIError(400, str(err), code="invalid_forecast_input") from err

    return ForecastResponse(
        product_id=result["product_id"],
        forecast=ForecastWindow(**result["forecast"]),
        shap=ShapAttribution(
            top_days=[ShapDay(**d) for d in result["shap"]["top_days"]],
            explanation=result["shap"]["explanation"],
        ),
        input_window_days=result["input_window_days"],
        output_horizon_days=result["output_horizon_days"],
        model_version=service.model_version(),
    )
