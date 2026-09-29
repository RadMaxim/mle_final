from pydantic import BaseModel, Field, ConfigDict


class RankerModelParams(BaseModel):
    """Признаки кандидата для CatBoostRanker."""

    als_score: float = Field(
        default=0.5,
        description="Score кандидата из ALS"
    )

    als_rank: int = Field(
        default=10,
        description="Позиция кандидата в ALS",
        ge=1
    )

    similarity_score: float = Field(
        default=0.5,
        description="Score похожести кандидата"
    )

    similarity_rank: int = Field(
        default=10,
        description="Позиция кандидата среди Similar-рекомендаций",
        ge=1
    )

    popular_score: float = Field(
        default=10.0,
        description="Score популярности кандидата",
        ge=0.0
    )

    popular_rank: int = Field(
        default=10,
        description="Позиция кандидата в Top Popular",
        ge=1
    )


class RecommendationPredictionInput(BaseModel):
    """Запрос на получение ranking score."""

    user_id: str = Field(
        default="12345",
        description="Идентификатор пользователя"
    )

    model_params: RankerModelParams

    model_config = ConfigDict(
        from_attributes=True
    )