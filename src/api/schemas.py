from typing import Literal

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str
    code: str


class RecommendationItem(BaseModel):
    product_id: int
    name: str
    score: float
    explanation: str
    bucket: Literal["reorder", "new_for_you"]


class RecommendationsResponse(BaseModel):
    user_id: int
    reorder: list[RecommendationItem]
    new_for_you: list[RecommendationItem]
    model_version: str


class ForecastRequest(BaseModel):
    instacart_product_id: int
    daily_counts: list[int]
    window_end_date: str


class ForecastWindow(BaseModel):
    dates: list[str]
    predicted_counts: list[int]


class ShapDay(BaseModel):
    day_index: int
    date: str
    attribution: float


class ShapAttribution(BaseModel):
    top_days: list[ShapDay]
    explanation: str


class ForecastResponse(BaseModel):
    product_id: int
    forecast: ForecastWindow
    shap: ShapAttribution
    input_window_days: Literal[90]
    output_horizon_days: Literal[7]
    model_version: str


class EnsembleWeights(BaseModel):
    resnet50: float
    mobilenetv2: float


class GradeImageResponse(BaseModel):
    predicted_class: str
    produce_type: str
    quality: str
    confidence: float
    colour_pct: float
    size_pct: float
    ripeness_pct: float
    grade: Literal["A", "B", "C"]
    ensemble_weights: EnsembleWeights
    recommended_action: str
    grad_cam_url: str
    model_version: str


class GradCamResponse(BaseModel):
    heatmap_base64: str
    predicted_class: str
    explanation: str
    model_version: str


class UploadModelResponse(BaseModel):
    status: Literal["success", "error"]
    task: Literal["task1a_knn", "task1b_lstm", "task2_resnet", "task2_mobilenet"]
    version: str
    message: str
