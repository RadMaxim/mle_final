from pathlib import Path

import pandas as pd
from catboost import CatBoostRanker


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
            f"Model successfully loaded: "
            f"{self.model_path}"
        )

        self.features = [
            "als_score",
            "als_rank",
            "similarity_score",
            "similarity_rank",
            "popular_score",
            "popular_rank",
        ]


    def handle(
        self,
        params: dict
    ) -> dict:

        try:

            user_id = params["user_id"]
            items = params["items"]
            top_k = params["top_k"]

            if not items:
                return {
                    "user_id": user_id,
                    "recommendations": []
                }

            # ------------------------------------------
            # Создаём DataFrame кандидатов
            # ------------------------------------------

            candidates = pd.DataFrame(
                items
            )

            # ------------------------------------------
            # Предсказание CatBoostRanker
            # ------------------------------------------

            candidates["score"] = (
                self.model.predict(
                    candidates[self.features]
                )
            )

            # ------------------------------------------
            # Сортируем кандидатов
            # ------------------------------------------

            candidates = (
                candidates
                .sort_values(
                    "score",
                    ascending=False
                )
                .head(top_k)
                .reset_index(drop=True)
            )

            # ------------------------------------------
            # Добавляем rank
            # ------------------------------------------

            candidates["rank"] = (
                candidates.index + 1
            )

            # ------------------------------------------
            # Формируем ответ
            # ------------------------------------------

            recommendations = (
                candidates[
                    [
                        "item_id",
                        "score",
                        "rank"
                    ]
                ]
                .to_dict(
                    orient="records"
                )
            )

            return {
                "user_id": user_id,
                "recommendations": recommendations
            }

        except Exception as e:

            print(
                f"Prediction error: {e}"
            )

            return {
                "Error":
                "Problem with request"
            }