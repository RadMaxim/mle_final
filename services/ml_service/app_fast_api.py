from fastapi import FastAPI

from ml_service.fast_api_handler import FastApiHandler

from prometheus_fastapi_instrumentator import (
    Instrumentator
)

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

@app.get(
    "/api/recommendations/{user_id}"
)
def get_recommendations(
    user_id: int,
    top_k: int = 10
):

    result = (
        app.handler.get_recommendations(
            user_id=user_id,
            top_k=top_k
        )
    )

    for recommendation in result.get(
        "recommendations",
        []
    ):
        ranker_predictions.observe(
            recommendation["score"]
        )

    return result