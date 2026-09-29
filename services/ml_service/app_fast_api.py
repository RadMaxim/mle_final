from fastapi import FastAPI

from ml_service.fast_api_handler import FastApiHandler
from ml_service.RecommendationPredictionInput import (
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
# Prediction endpoint
# --------------------------------------------------

@app.post("/api/recommendation/")
def get_prediction_for_item(
    request: RecommendationPredictionInput
):

    params = {
        "user_id": request.user_id,
        "model_params": (
            request.model_params.model_dump()
        )
    }

    pred = app.handler.handle(
        params
    )

    prediction_value = pred.get(
        "score",
        0.0
    )

    ranker_predictions.observe(
        prediction_value
    )

    return pred