from fastapi import FastAPI

from ml_service.fast_api_handler import FastApiHandler
from ml_service.recommendation_prediction_input import (
    RecommendationPredictionInput
)

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram


app = FastAPI()


# --------------------------------------------------
# Prometheus
# --------------------------------------------------

instrumentator = Instrumentator()

instrumentator.instrument(
    app
).expose(
    app
)


ranker_predictions = Histogram(
    "ranker_prediction_score",
    "Histogram of CatBoost ranker scores",
    buckets=(
        -2.0,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        2.0,
    )
)


# --------------------------------------------------
# Handler
# --------------------------------------------------

app.handler = FastApiHandler()


# --------------------------------------------------
# Recommendation endpoint
# --------------------------------------------------

@app.post("/api/recommendation/")
def get_recommendations(
    request: RecommendationPredictionInput
):

    params = {
        "user_id": request.user_id,

        "items": [
            item.model_dump()
            for item in request.items
        ],

        "top_k": request.top_k
    }

    result = app.handler.handle(
        params
    )

    # ----------------------------------------------
    # Prometheus
    # ----------------------------------------------

    for recommendation in result.get(
        "recommendations",
        []
    ):
        ranker_predictions.observe(
            recommendation["score"]
        )

    return result