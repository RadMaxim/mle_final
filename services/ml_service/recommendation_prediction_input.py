from pydantic import BaseModel, Field


class RecommendationCandidate(BaseModel):
    """Один товар-кандидат для ранжирования."""

    item_id: int

    als_score: float | None = None
    als_rank: int | None = None

    similarity_score: float | None = None
    similarity_rank: int | None = None

    popular_score: float | None = None
    popular_rank: int | None = None


class RecommendationPredictionInput(BaseModel):
    """Запрос на получение рекомендаций."""

    user_id: str

    items: list[RecommendationCandidate]

    top_k: int = Field(
        default=10,
        ge=1,
        le=100
    )