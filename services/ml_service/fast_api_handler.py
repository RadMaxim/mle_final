from pathlib import Path
import os

import duckdb
import pandas as pd

from catboost import CatBoostRanker


BASE_DIR = Path(__file__).resolve().parent.parent


class FastApiHandler:

    def __init__(self):

        # --------------------------------------------------
        # Модель
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Пути к данным
        # --------------------------------------------------

        self.candidates_path = os.getenv(
            "CANDIDATES_PATH",
            "/app/candidates_result.parquet"
        )

        self.events_path = os.getenv(
            "EVENTS_PATH",
            "/app/events.parquet"
        )

        print(
            f"Candidates path: "
            f"{self.candidates_path}"
        )

        print(
            f"Events path: "
            f"{self.events_path}"
        )

        # --------------------------------------------------
        # Признаки модели
        # --------------------------------------------------

        self.features = [
            "als_score",
            "als_rank",
            "similarity_score",
            "similarity_rank",
            "popular_score",
            "popular_rank",
        ]


    def get_recommendations(
        self,
        user_id: int,
        top_k: int = 10
    ) -> dict:

        try:

            # --------------------------------------------------
            # Получаем кандидатов пользователя
            #
            # При этом исключаем товары, которые пользователь
            # уже купил (event = transaction)
            # --------------------------------------------------

            candidates = duckdb.sql(
                """
                SELECT
                    c.item_id,
                    c.als_score,
                    c.als_rank,
                    c.similarity_score,
                    c.similarity_rank,
                    c.popular_score,
                    c.popular_rank
                FROM read_parquet(?) AS c
                WHERE c.user_id = ?
                  AND c.item_id NOT IN (
                      SELECT DISTINCT
                          e.itemid
                      FROM read_parquet(?) AS e
                      WHERE e.visitorid = ?
                        AND e.event = 'transaction'
                  )
                """,
                params=[
                    self.candidates_path,
                    user_id,
                    self.events_path,
                    user_id
                ]
            ).df()

            # --------------------------------------------------
            # Если кандидатов нет
            # --------------------------------------------------

            if candidates.empty:

                return {
                    "user_id": user_id,
                    "recommendations": []
                }

            # --------------------------------------------------
            # Предсказание CatBoostRanker
            # --------------------------------------------------

            candidates["score"] = (
                self.model.predict(
                    candidates[
                        self.features
                    ]
                )
            )

            # --------------------------------------------------
            # Сортировка и Top-K
            # --------------------------------------------------

            candidates = (
                candidates
                .sort_values(
                    "score",
                    ascending=False
                )
                .head(top_k)
                .reset_index(
                    drop=True
                )
            )

            # --------------------------------------------------
            # Rank
            # --------------------------------------------------

            candidates["rank"] = (
                candidates.index + 1
            )

            # --------------------------------------------------
            # Формируем ответ
            # --------------------------------------------------

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
                "Error": str(e)
            }