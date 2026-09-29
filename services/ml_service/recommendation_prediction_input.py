from pydantic import BaseModel, Field


class RecommendationPredictionInput(BaseModel):
    """Запрос на получение рекомендаций."""

    user_id: int = Field(
        description="Идентификатор пользователя"
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Количество рекомендаций"
    )