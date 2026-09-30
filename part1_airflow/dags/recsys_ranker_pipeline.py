from datetime import datetime, timedelta
from pathlib import Path

import gc
import os

import boto3
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from airflow.decorators import dag, task


# ============================================================
# PARAMS
# ============================================================

PARAMS_PATH = Path(
    "/opt/airflow/params.yaml"
)

with PARAMS_PATH.open(
    "r",
    encoding="utf-8",
) as file:

    params = yaml.safe_load(file)


# ============================================================
# PATHS
# ============================================================

WORK_DIR = Path(
    params["ranker"]["work_dir"]
)

INPUT_DIR = (
    WORK_DIR
    / "input"
)

INTERMEDIATE_DIR = (
    WORK_DIR
    / "intermediate"
)

OUTPUT_DIR = (
    WORK_DIR
    / "output"
)

MODEL_DIR = (
    WORK_DIR
    / "models"
)


EVENTS_PATH = (
    INPUT_DIR
    / "events.parquet"
)

CANDIDATES_PATH = (
    INPUT_DIR
    / "candidates_result.parquet"
)

RANKER_TRAIN_PATH = (
    INTERMEDIATE_DIR
    / "ranker_train.parquet"
)

SCORED_PATH = (
    INTERMEDIATE_DIR
    / "recommendations_scored.parquet"
)

RECOMMENDATIONS_PATH = (
    OUTPUT_DIR
    / "recommendations.parquet"
)

MODEL_PATH = (
    MODEL_DIR
    / "catboost_ranker.cbm"
)


# ============================================================
# SPLIT
# ============================================================

EVALUATION_DAYS = params[
    "split"
]["evaluation_days"]

RANKER_DAYS = params[
    "split"
]["ranker_days"]


# ============================================================
# RANKER PARAMS
# ============================================================

FEATURES = params[
    "ranker"
]["features"]

POSITIVE_EVENTS = params[
    "ranker"
]["positive_events"]

POSITIVE_EVENTS_SQL = ", ".join(
    f"'{event}'"
    for event in POSITIVE_EVENTS
)


RANKER_ITERATIONS = params[
    "ranker"
]["iterations"]

RANKER_DEPTH = params[
    "ranker"
]["depth"]

RANKER_LEARNING_RATE = params[
    "ranker"
]["learning_rate"]

RANKER_LOSS_FUNCTION = params[
    "ranker"
]["loss_function"]

RANKER_RANDOM_STATE = params[
    "ranker"
]["random_seed"]


INFERENCE_BATCH_SIZE = params[
    "ranker"
]["inference_batch_size"]

ROW_GROUP_SIZE = params[
    "ranker"
]["row_group_size"]


# ============================================================
# S3
# ============================================================

S3_ENDPOINT = params[
    "s3"
]["endpoint_url"]

S3_DATA_PREFIX = params[
    "s3"
]["data_prefix"]

S3_CANDIDATES_PREFIX = params[
    "s3"
]["candidates_prefix"]

S3_RECOMMENDATIONS_PREFIX = params[
    "s3"
]["recommendations_prefix"]


# ============================================================
# MLFLOW
# ============================================================

MLFLOW_TRACKING_URI = params[
    "mlflow"
]["tracking_uri"]

MLFLOW_EXPERIMENT = params[
    "mlflow"
]["ranker_experiment_name"]


# ============================================================
# AIRFLOW
# ============================================================

default_args = {

    "owner":
        params["airflow"]["owner"],

    "retries":
        params["airflow"]["retries"],

    "retry_delay":
        timedelta(
            minutes=params[
                "airflow"
            ]["retry_delay_minutes"]
        ),
}


# ============================================================
# HELPERS
# ============================================================

def get_s3_client():

    bucket_name = os.getenv(
        "S3_BUCKET_NAME"
    )

    access_key = os.getenv(
        "AWS_ACCESS_KEY_ID"
    )

    secret_key = os.getenv(
        "AWS_SECRET_ACCESS_KEY"
    )

    if not all([
        bucket_name,
        access_key,
        secret_key,
    ]):

        raise RuntimeError(
            "Не заданы S3 credentials"
        )

    client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    return client, bucket_name


def remove_if_exists(
    path: Path,
):

    if path.exists():

        path.unlink()


def check_file(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Файл не создан: {path}"
        )

    if path.stat().st_size == 0:

        raise RuntimeError(
            f"Файл пустой: {path}"
        )


# ============================================================
# DAG
# ============================================================

@dag(
    dag_id="recsys_ranker_pipeline",
    description=(
        "Обучение CatBoostRanker "
        "и генерация итоговых рекомендаций"
    ),
    default_args=default_args,
    start_date=datetime(
        2026,
        9,
        1,
    ),
    schedule=None,
    catchup=False,
    tags=[
        "recsys",
        "ranker",
        "catboost",
    ],
)
def recsys_ranker_pipeline():


    # ========================================================
    # 1. DOWNLOAD INPUTS
    # ========================================================

    @task
    def download_inputs():

        INPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        INTERMEDIATE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        s3, bucket_name = (
            get_s3_client()
        )

        files = {

            (
                f"{S3_DATA_PREFIX}/"
                "events.parquet"
            ):
                EVENTS_PATH,

            (
                f"{S3_CANDIDATES_PREFIX}/"
                "candidates_result.parquet"
            ):
                CANDIDATES_PATH,
        }

        for s3_key, local_path in (
            files.items()
        ):

            print(
                f"Скачиваю: "
                f"s3://{bucket_name}/"
                f"{s3_key}"
            )

            s3.download_file(
                bucket_name,
                s3_key,
                str(local_path),
            )

            check_file(
                local_path
            )

            print(
                f"Скачан: {local_path}"
            )


    # ========================================================
    # 2. BUILD RANKER TRAIN
    # ========================================================

    @task
    def build_ranker_train():

        remove_if_exists(
            RANKER_TRAIN_PATH
        )

        con = duckdb.connect()

        try:

            # ------------------------------------------------
            # Time split
            # ------------------------------------------------

            data_end = con.execute(
                f"""
                SELECT
                    MAX(timestamp)
                    + INTERVAL 1 MILLISECOND

                FROM read_parquet(
                    '{EVENTS_PATH}'
                )
                """
            ).fetchone()[0]

            eval_start = (
                data_end
                - timedelta(
                    days=EVALUATION_DAYS
                )
            )

            ranker_start = (
                eval_start
                - timedelta(
                    days=RANKER_DAYS
                )
            )

            print(
                "Data end:",
                data_end
            )

            print(
                "Ranker train:",
                ranker_start,
                "->",
                eval_start
            )

            print(
                "Evaluation:",
                eval_start,
                "->",
                data_end
            )


            # ------------------------------------------------
            # Positive pairs count
            # ------------------------------------------------

            positive_pairs_count = (
                con.execute(
                    f"""
                    SELECT COUNT(*)

                    FROM (

                        SELECT DISTINCT
                            visitorid,
                            itemid

                        FROM read_parquet(
                            '{EVENTS_PATH}'
                        )

                        WHERE
                            timestamp >= ?
                            AND timestamp < ?

                            AND event IN (
                                {POSITIVE_EVENTS_SQL}
                            )
                    )
                    """,
                    [
                        ranker_start,
                        eval_start,
                    ],
                )
                .fetchone()[0]
            )


            # ------------------------------------------------
            # Positive pairs found in candidates
            # ------------------------------------------------

            positive_found = (
                con.execute(
                    f"""
                    SELECT COUNT(*)

                    FROM (

                        SELECT DISTINCT
                            e.visitorid,
                            e.itemid

                        FROM read_parquet(
                            '{EVENTS_PATH}'
                        ) e

                        INNER JOIN read_parquet(
                            '{CANDIDATES_PATH}'
                        ) c

                            ON
                                e.visitorid
                                = c.user_id

                            AND
                                e.itemid
                                = c.item_id

                        WHERE
                            e.timestamp >= ?
                            AND e.timestamp < ?

                            AND e.event IN (
                                {POSITIVE_EVENTS_SQL}
                            )
                    )
                    """,
                    [
                        ranker_start,
                        eval_start,
                    ],
                )
                .fetchone()[0]
            )

            coverage = (
                positive_found
                / positive_pairs_count

                if positive_pairs_count > 0

                else 0.0
            )

            print(
                "Positive pairs:",
                positive_pairs_count
            )

            print(
                "Positive found:",
                positive_found
            )

            print(
                "Coverage:",
                coverage
            )


            # ------------------------------------------------
            # Build train dataset
            # ------------------------------------------------

            con.execute(
                f"""
                COPY (

                    WITH positive_pairs AS (

                        SELECT DISTINCT

                            visitorid
                                AS user_id,

                            itemid
                                AS item_id

                        FROM read_parquet(
                            '{EVENTS_PATH}'
                        )

                        WHERE
                            timestamp >= ?
                            AND timestamp < ?

                            AND event IN (
                                {POSITIVE_EVENTS_SQL}
                            )
                    ),

                    positive_users AS (

                        SELECT DISTINCT
                            user_id

                        FROM positive_pairs
                    ),

                    train_raw AS (

                        SELECT

                            c.user_id,
                            c.item_id,

                            c.als_score,
                            c.als_rank,

                            c.similarity_score,
                            c.similarity_rank,

                            c.popular_score,
                            c.popular_rank,

                            CASE
                                WHEN
                                    p.item_id IS NOT NULL
                                THEN 1
                                ELSE 0
                            END::TINYINT
                                AS target

                        FROM read_parquet(
                            '{CANDIDATES_PATH}'
                        ) c

                        INNER JOIN positive_users u

                            ON
                                c.user_id
                                = u.user_id

                        LEFT JOIN positive_pairs p

                            ON
                                c.user_id
                                = p.user_id

                            AND
                                c.item_id
                                = p.item_id
                    ),

                    valid_users AS (

                        SELECT
                            user_id

                        FROM train_raw

                        GROUP BY
                            user_id

                        HAVING
                            SUM(target) > 0

                            AND
                            COUNT(*)
                            - SUM(target) > 0
                    )

                    SELECT
                        t.*

                    FROM train_raw t

                    INNER JOIN valid_users v

                        ON
                            t.user_id
                            = v.user_id

                    ORDER BY
                        t.user_id,
                        t.als_rank
                )

                TO '{RANKER_TRAIN_PATH}'

                (
                    FORMAT PARQUET,
                    COMPRESSION SNAPPY,
                    ROW_GROUP_SIZE
                        {ROW_GROUP_SIZE}
                )
                """,
                [
                    ranker_start,
                    eval_start,
                ],
            )

            check_file(
                RANKER_TRAIN_PATH
            )


            # ------------------------------------------------
            # Stats
            # ------------------------------------------------

            stats = con.execute(
                f"""
                SELECT

                    COUNT(*),

                    COUNT(
                        DISTINCT user_id
                    ),

                    SUM(target),

                    AVG(target)

                FROM read_parquet(
                    '{RANKER_TRAIN_PATH}'
                )
                """
            ).fetchone()

            return {

                "positive_pairs_count":
                    int(
                        positive_pairs_count
                    ),

                "positive_found":
                    int(
                        positive_found
                    ),

                "candidate_positive_coverage":
                    float(
                        coverage
                    ),

                "ranker_rows":
                    int(
                        stats[0]
                    ),

                "ranker_users":
                    int(
                        stats[1]
                    ),

                "positive_count":
                    int(
                        stats[2]
                    ),

                "positive_share":
                    float(
                        stats[3]
                    ),
            }

        finally:

            con.close()


    # ========================================================
    # 3. TRAIN CATBOOST
    # ========================================================

    @task
    def train_ranker():

        from catboost import (
            CatBoostRanker
        )

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        check_file(
            RANKER_TRAIN_PATH
        )

        ranker_train = (
            pd.read_parquet(
                RANKER_TRAIN_PATH
            )
        )

        print(
            "Ranker rows:",
            f"{len(ranker_train):,}"
        )

        print(
            "Ranker users:",
            ranker_train[
                "user_id"
            ].nunique()
        )

        ranker = CatBoostRanker(

            iterations=
                RANKER_ITERATIONS,

            depth=
                RANKER_DEPTH,

            learning_rate=
                RANKER_LEARNING_RATE,

            loss_function=
                RANKER_LOSS_FUNCTION,

            random_seed=
                RANKER_RANDOM_STATE,

            verbose=1,
        )

        ranker.fit(

            ranker_train[
                FEATURES
            ],

            ranker_train[
                "target"
            ],

            group_id=
                ranker_train[
                    "user_id"
                ],
        )

        ranker.save_model(
            str(MODEL_PATH)
        )

        check_file(
            MODEL_PATH
        )

        print(
            f"Модель сохранена: "
            f"{MODEL_PATH}"
        )

        del ranker_train
        del ranker

        gc.collect()


    # ========================================================
    # 4. GENERATE RECOMMENDATIONS
    # ========================================================

    @task
    def generate_recommendations():

        from catboost import (
            CatBoostRanker
        )

        check_file(
            CANDIDATES_PATH
        )

        check_file(
            MODEL_PATH
        )

        remove_if_exists(
            SCORED_PATH
        )

        remove_if_exists(
            RECOMMENDATIONS_PATH
        )

        ranker = CatBoostRanker()

        ranker.load_model(
            str(MODEL_PATH)
        )

        parquet_file = pq.ParquetFile(
            CANDIDATES_PATH
        )

        total_rows = (
            parquet_file
            .metadata
            .num_rows
        )

        writer = None

        processed = 0

        try:

            for batch_arrow in (
                parquet_file.iter_batches(

                    batch_size=
                        INFERENCE_BATCH_SIZE,

                    columns=[
                        "user_id",
                        "item_id",
                        *FEATURES,
                    ],
                )
            ):

                batch = (
                    batch_arrow
                    .to_pandas()
                )

                scores = (
                    ranker.predict(
                        batch[
                            FEATURES
                        ]
                    )
                    .astype(
                        np.float32
                    )
                )

                result = batch[
                    [
                        "user_id",
                        "item_id",
                    ]
                ].copy()

                result[
                    "score"
                ] = scores

                table = (
                    pa.Table
                    .from_pandas(
                        result,
                        preserve_index=False,
                    )
                )

                if writer is None:

                    writer = (
                        pq.ParquetWriter(
                            SCORED_PATH,
                            table.schema,
                            compression="snappy",
                        )
                    )

                writer.write_table(
                    table
                )

                processed += len(
                    result
                )

                print(
                    f"Обработано: "
                    f"{processed:,} / "
                    f"{total_rows:,} "
                    f"("
                    f"{processed / total_rows * 100:.2f}"
                    f"%)"
                )

                del batch_arrow
                del batch
                del result
                del scores
                del table

                gc.collect()

        finally:

            if writer is not None:

                writer.close()

        check_file(
            SCORED_PATH
        )


        # ----------------------------------------------------
        # Final rank via DuckDB
        # ----------------------------------------------------

        con = duckdb.connect()

        try:

            con.execute(
                f"""
                COPY (

                    SELECT

                        user_id,
                        item_id,
                        score,

                        CAST(

                            ROW_NUMBER()
                            OVER (

                                PARTITION BY
                                    user_id

                                ORDER BY
                                    score DESC
                            )

                            AS SMALLINT

                        ) AS rank

                    FROM read_parquet(
                        '{SCORED_PATH}'
                    )
                )

                TO '{RECOMMENDATIONS_PATH}'

                (
                    FORMAT PARQUET,
                    COMPRESSION SNAPPY,
                    ROW_GROUP_SIZE
                        {ROW_GROUP_SIZE}
                )
                """
            )

        finally:

            con.close()

        check_file(
            RECOMMENDATIONS_PATH
        )

        remove_if_exists(
            SCORED_PATH
        )

        print(
            "Итоговые рекомендации: "
            f"{RECOMMENDATIONS_PATH}"
        )


    # ========================================================
    # 5. UPLOAD RECOMMENDATIONS TO S3
    # ========================================================

    @task
    def upload_recommendations():

        check_file(
            RECOMMENDATIONS_PATH
        )

        s3, bucket_name = (
            get_s3_client()
        )

        s3_key = (
            f"{S3_RECOMMENDATIONS_PREFIX}/"
            "recommendations.parquet"
        )

        s3.upload_file(
            str(
                RECOMMENDATIONS_PATH
            ),
            bucket_name,
            s3_key,
        )

        print(
            f"Загружено: "
            f"s3://{bucket_name}/"
            f"{s3_key}"
        )


    # ========================================================
    # 6. LOG TO MLFLOW
    # ========================================================

    @task
    def log_mlflow(
        metrics: dict,
    ):

        import mlflow
        import mlflow.catboost

        from catboost import (
            CatBoostRanker
        )

        check_file(
            MODEL_PATH
        )

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        mlflow.set_experiment(
            MLFLOW_EXPERIMENT
        )

        ranker = CatBoostRanker()

        ranker.load_model(
            str(MODEL_PATH)
        )

        with mlflow.start_run(
            run_name="catboost_ranker"
        ):

            # -----------------------------------------------
            # PARAMS
            # -----------------------------------------------

            mlflow.log_params({

                "iterations":
                    RANKER_ITERATIONS,

                "depth":
                    RANKER_DEPTH,

                "learning_rate":
                    RANKER_LEARNING_RATE,

                "loss_function":
                    RANKER_LOSS_FUNCTION,

                "random_seed":
                    RANKER_RANDOM_STATE,

                "evaluation_days":
                    EVALUATION_DAYS,

                "ranker_days":
                    RANKER_DAYS,

                "features":
                    ",".join(
                        FEATURES
                    ),

                "positive_events":
                    ",".join(
                        POSITIVE_EVENTS
                    ),

                "inference_batch_size":
                    INFERENCE_BATCH_SIZE,

                "row_group_size":
                    ROW_GROUP_SIZE,
            })


            # -----------------------------------------------
            # METRICS
            # -----------------------------------------------

            mlflow.log_metrics({

                key:
                    float(value)

                for key, value
                in metrics.items()
            })


            # -----------------------------------------------
            # MODEL
            # -----------------------------------------------

            mlflow.catboost.log_model(

                cb_model=
                    ranker,

                artifact_path=
                    "model",
            )

        print(
            "MLflow run успешно записан"
        )


    # ========================================================
    # GRAPH
    # ========================================================

    download = (
        download_inputs()
    )

    metrics = (
        build_ranker_train()
    )

    model = (
        train_ranker()
    )

    recommendations = (
        generate_recommendations()
    )

    upload = (
        upload_recommendations()
    )

    mlflow_log = (
        log_mlflow(
            metrics
        )
    )


    download >> metrics

    metrics >> model

    model >> recommendations

    recommendations >> upload

    upload >> mlflow_log


recsys_ranker_pipeline()