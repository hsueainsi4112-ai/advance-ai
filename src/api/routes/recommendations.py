from fastapi import APIRouter

from src.api.schemas import RecommendationItem, RecommendationsResponse
from src.recommendation import service

router = APIRouter(prefix="/api/v1")


@router.get(
    "/recommendations/{instacart_user_id}",
    response_model=RecommendationsResponse,
)
async def get_recommendations(instacart_user_id: int) -> RecommendationsResponse:
    result = service.get_recommendations(instacart_user_id)
    return RecommendationsResponse(
        user_id=result["user_id"],
        reorder=[RecommendationItem(**r) for r in result["reorder"]],
        new_for_you=[RecommendationItem(**r) for r in result["new_for_you"]],
        model_version=service.model_version(),
    )
