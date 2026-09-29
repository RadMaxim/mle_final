from catboost import CatBoostRanker
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class FastApiHandler:

    def __init__(self):

        self.model_path = (
            BASE_DIR
            / "models"
            / "catboost_ranker.cbm"
        )

        self.model = CatBoostRanker()

        self.model.load_model(
            str(self.model_path)
        )

        print(
            f"Model loaded: {self.model_path}"
        )

        self.required_model_params = [
            "als_score",
            "als_rank",
            "similarity_score",
            "similarity_rank",
            "popular_score",
            "popular_rank",
        ]


    def validate_params(
        self,
        params: dict
    ) -> bool:

        if "user_id" not in params:
            return False

        if "model_params" not in params:
            return False

        model_params = params[
            "model_params"
        ]

        if not isinstance(
            model_params,
            dict
        ):
            return False

        if set(model_params.keys()) != set(
            self.required_model_params
        ):
            return False

        return True


    def handle(
        self,
        params: dict
    ) -> dict:

        try:

            if not self.validate_params(
                params
            ):
                return {
                    "Error":
                    "Problem with parameters"
                }

            user_id = params["user_id"]

            model_params = params[
                "model_params"
            ]

            X = pd.DataFrame(
                [model_params],
                columns=self.required_model_params
            )

            prediction = float(
                self.model.predict(X)[0]
            )

            return {
                "user_id": user_id,
                "score": prediction
            }

        except Exception as e:

            print(
                f"Prediction error: {e}"
            )

            return {
                "Error":
                "Problem with request"
            }