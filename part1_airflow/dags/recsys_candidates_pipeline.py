from datetime import datetime, timedelta
from pathlib import Path
import os
import time

import boto3
import duckdb
import requests
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
    params["candidates"]["work_dir"]
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


EVENTS_PATH = (
    INPUT_DIR
    / "events.parquet"
)

TOP_POPULAR_PATH = (
    INPUT_DIR
    / "top_popular.parquet"
)

PERSONAL_ALS_PATH = (
    INPUT_DIR
    / "personal_als.parquet"
)

SIMILAR_PATH = (
    INPUT_DIR
    / "similar.parquet"
)


USER_HISTORY_PATH = (
    INTERMEDIATE_DIR
    / "user_history.parquet"
)

SIMILAR_FEATURES_PATH = (
    INTERMEDIATE_DIR
    / "similar_features.parquet"
)

CANDIDATES_UNIQUE_PATH = (
    INTERMEDIATE_DIR
    / "candidates_unique.parquet"
)

CANDIDATES_POPULAR_PATH = (
    INTERMEDIATE_DIR
    / "candidates_with_popular.parquet"
)

CANDIDATES_SIMILAR_PATH = (
    INTERMEDIATE_DIR
    / "candidates_with_similar.parquet"
)

CANDIDATES_ALS_PATH = (
    INTERMEDIATE_DIR
    / "candidates_with_als.parquet"
)

CANDIDATES_RESULT_PATH = (
    OUTPUT_DIR
    / "candidates_result.parquet"
)


# ============================================================
# PARAMETERS
# ============================================================

HISTORY_ITEMS = params[
    "candidates"
]["history_items_per_user"]

SIMILAR_TOP_N = params[
    "candidates"
]["similar_top_n"]

POPULAR_TOP_N = params[
    "candidates"
]["popular_top_n"]

ROW_GROUP_SIZE = params[
    "candidates"
]["row_group_size"]

MIN_VALID_SIZE = params[
    "candidates"
]["min_valid_size"]


EVALUATION_DAYS = params[
    "split"
]["evaluation_days"]

RANKER_DAYS = params[
    "split"
]["ranker_days"]


# ============================================================
# S3
# ============================================================

S3_ENDPOINT = params[
    "s3"
]["endpoint_url"]

S3_DATA_PREFIX = params[
    "s3"
]["data_prefix"]

S3_RECOMMENDATIONS_PREFIX = params[
    "s3"
]["recommendations_prefix"]

S3_CANDIDATES_PREFIX = params[
    "s3"
]["candidates_prefix"]


# ============================================================
# MLFLOW
# ============================================================

MLFLOW_TRACKING_URI = params[
    "mlflow"
]["tracking_uri"].rstrip("/")

MLFLOW_EXPERIMENT = params[
    "mlflow"
]["candidates_experiment_name"]


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


def remove_if_exists(path: Path):

    if path.exists():
        path.unlink()


def check_file(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Не создан файл: {path}"
        )

    if path.stat().st_size < MIN_VALID_SIZE:
        raise RuntimeError(
            f"Файл слишком мал: {path}"
        )


# ============================================================
# DAG
# ============================================================

@dag(
    dag_id="recsys_candidates_pipeline",
    description=(
        "Построение кандидатов и признаков "
        "для ранжирующей модели"
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
        "candidates",
        "ranking",
    ],
)
def recsys_candidates_pipeline():


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
                f"{S3_RECOMMENDATIONS_PREFIX}/"
                "top_popular.parquet"
            ):
                TOP_POPULAR_PATH,

            (
                f"{S3_RECOMMENDATIONS_PREFIX}/"
                "personal_als.parquet"
            ):
                PERSONAL_ALS_PATH,

            (
                f"{S3_RECOMMENDATIONS_PREFIX}/"
                "similar.parquet"
            ):
                SIMILAR_PATH,
        }

        for s3_key, local_path in (
            files.items()
        ):

            print(
                f"Скачиваю "
                f"s3://{bucket_name}/{s3_key}"
            )

            s3.download_file(
                bucket_name,
                s3_key,
                str(local_path),
            )

            print(
                f"Скачан: {local_path}"
            )


    # ========================================================
    # 2. BUILD CANDIDATES
    # ========================================================

    @task
    def build_candidates():

        con = duckdb.connect()

        try:

            # ------------------------------------------------
            # Вычисляем временные границы
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
                "Eval start:",
                eval_start
            )

            print(
                "Ranker start:",
                ranker_start
            )


            # =================================================
            # USER HISTORY
            # =================================================

            remove_if_exists(
                USER_HISTORY_PATH
            )

            con.execute(
                f"""
                COPY (
                    SELECT
                        user_id,
                        item_id
                            AS source_item_id

                    FROM (
                        SELECT
                            visitorid
                                AS user_id,

                            itemid
                                AS item_id,

                            timestamp,

                            ROW_NUMBER() OVER (
                                PARTITION BY
                                    visitorid,
                                    itemid
                                ORDER BY
                                    timestamp DESC
                            ) AS item_rn

                        FROM read_parquet(
                            '{EVENTS_PATH}'
                        )

                        WHERE
                            event IN (
                                'view',
                                'addtocart',
                                'transaction'
                            )

                            AND timestamp < ?
                    )

                    WHERE item_rn = 1

                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY user_id
                        ORDER BY timestamp DESC
                    ) <= {HISTORY_ITEMS}
                )

                TO '{USER_HISTORY_PATH}'

                (
                    FORMAT PARQUET,
                    COMPRESSION SNAPPY
                )
                """,
                [
                    ranker_start
                ],
            )

            check_file(
                USER_HISTORY_PATH
            )


            # =================================================
            # SIMILAR FEATURES
            # =================================================

            remove_if_exists(
                SIMILAR_FEATURES_PATH
            )

            con.execute(
                f"""
                COPY (
                    SELECT
                        h.user_id,

                        s.similar_itemid
                            AS item_id,

                        MAX(
                            s.score
                        ) AS similarity_score,

                        MIN(
                            s.rank
                        ) AS similarity_rank,

                        COUNT(
                            DISTINCT
                            h.source_item_id
                        ) AS similarity_sources

                    FROM read_parquet(
                        '{USER_HISTORY_PATH}'
                    ) h

                    JOIN read_parquet(
                        '{SIMILAR_PATH}'
                    ) s

                        ON
                            h.source_item_id
                            = s.itemid

                    WHERE
                        s.rank <=
                        {SIMILAR_TOP_N}

                    GROUP BY
                        h.user_id,
                        s.similar_itemid
                )

                TO '{SIMILAR_FEATURES_PATH}'

                (
                    FORMAT PARQUET,
                    COMPRESSION SNAPPY
                )
                """
            )

            check_file(
                SIMILAR_FEATURES_PATH
            )


            # =================================================
            # UNIQUE CANDIDATES
            # =================================================

            remove_if_exists(
                CANDIDATES_UNIQUE_PATH
            )

            con.execute(
                f"""
                COPY (
                    SELECT DISTINCT
                        user_id,
                        item_id

                    FROM (

                        SELECT
                            visitorid
                                AS user_id,

                            itemid
                                AS item_id

                        FROM read_parquet(
                            '{PERSONAL_ALS_PATH}'
                        )

                        UNION ALL

                        SELECT
                            user_id,
                            item_id

                        FROM read_parquet(
                            '{SIMILAR_FEATURES_PATH}'
                        )

                        UNION ALL

                        SELECT
                            u.user_id,
                            p.item_id

                        FROM (

                            SELECT DISTINCT
                                visitorid
                                    AS user_id

                            FROM read_parquet(
                                '{EVENTS_PATH}'
                            )

                            WHERE
                                timestamp < ?
                        ) u

                        CROSS JOIN (

                            SELECT
                                itemid
                                    AS item_id

                            FROM read_parquet(
                                '{TOP_POPULAR_PATH}'
                            )

                            ORDER BY rank

                            LIMIT
                                {POPULAR_TOP_N}
                        ) p
                    )
                )

                TO '{CANDIDATES_UNIQUE_PATH}'

                (
                    FORMAT PARQUET,
                    COMPRESSION SNAPPY
                )
                """,
                [
                    ranker_start
                ],
            )

            check_file(
                CANDIDATES_UNIQUE_PATH
            )


            # =================================================
            # POPULAR FEATURES
            # =================================================

            remove_if_exists(
                CANDIDATES_POPULAR_PATH
            )

            con.execute(
                f"""
                COPY (
                    SELECT
                        CAST(
                            c.user_id
                            AS INTEGER
                        ) AS user_id,

                        CAST(
                            c.item_id
                            AS INTEGER
                        ) AS item_id,

                        CAST(
                            p.score
                            AS FLOAT
                        ) AS popular_score,

                        CAST(
                            p.rank
                            AS SMALLINT
                        ) AS popular_rank

                    FROM read_parquet(
                        '{CANDIDATES_UNIQUE_PATH}'
                    ) c

                    LEFT JOIN read_parquet(
                        '{TOP_POPULAR_PATH}'
                    ) p

                        ON
                            c.item_id
                            = p.itemid
                )

                TO '{CANDIDATES_POPULAR_PATH}'

                (
                    FORMAT PARQUET,
                    COMPRESSION SNAPPY,
                    ROW_GROUP_SIZE
                        {ROW_GROUP_SIZE}
                )
                """
            )

            check_file(
                CANDIDATES_POPULAR_PATH
            )


            # =================================================
            # SIMILAR FEATURES
            # =================================================

            remove_if_exists(
                CANDIDATES_SIMILAR_PATH
            )

            con.execute(
                f"""
                COPY (
                    SELECT
                        CAST(
                            c.user_id
                            AS INTEGER
                        ) AS user_id,

                        CAST(
                            c.item_id
                            AS INTEGER
                        ) AS item_id,

                        c.popular_score,
                        c.popular_rank,

                        CAST(
                            s.similarity_score
                            AS FLOAT
                        ) AS similarity_score,

                        CAST(
                            s.similarity_rank
                            AS SMALLINT
                        ) AS similarity_rank,

                        CAST(
                            s.similarity_sources
                            AS SMALLINT
                        ) AS similarity_sources

                    FROM read_parquet(
                        '{CANDIDATES_POPULAR_PATH}'
                    ) c

                    LEFT JOIN read_parquet(
                        '{SIMILAR_FEATURES_PATH}'
                    ) s

                        ON
                            c.user_id
                            = s.user_id

                        AND
                            c.item_id
                            = s.item_id
                )

                TO '{CANDIDATES_SIMILAR_PATH}'

                (
                    FORMAT PARQUET,
                    COMPRESSION SNAPPY,
                    ROW_GROUP_SIZE
                        {ROW_GROUP_SIZE}
                )
                """
            )

            check_file(
                CANDIDATES_SIMILAR_PATH
            )

            remove_if_exists(
                CANDIDATES_POPULAR_PATH
            )


            # =================================================
            # ALS FEATURES
            # =================================================

            remove_if_exists(
                CANDIDATES_ALS_PATH
            )

            con.execute(
                f"""
                COPY (
                    SELECT
                        c.*,

                        CAST(
                            a.score
                            AS FLOAT
                        ) AS als_score,

                        CAST(
                            a.rank
                            AS SMALLINT
                        ) AS als_rank

                    FROM read_parquet(
                        '{CANDIDATES_SIMILAR_PATH}'
                    ) c

                    LEFT JOIN read_parquet(
                        '{PERSONAL_ALS_PATH}'
                    ) a

                        ON
                            c.user_id
                            = a.visitorid

                        AND
                            c.item_id
                            = a.itemid
                )

                TO '{CANDIDATES_ALS_PATH}'

                (
                    FORMAT PARQUET,
                    COMPRESSION SNAPPY,
                    ROW_GROUP_SIZE
                        {ROW_GROUP_SIZE}
                )
                """
            )

            check_file(
                CANDIDATES_ALS_PATH
            )

            remove_if_exists(
                CANDIDATES_SIMILAR_PATH
            )


            # =================================================
            # FINAL RESULT
            # =================================================

            remove_if_exists(
                CANDIDATES_RESULT_PATH
            )

            con.execute(
                f"""
                COPY (
                    SELECT
                        user_id,
                        item_id,

                        CAST(
                            als_score
                            AS FLOAT
                        ) AS als_score,

                        als_rank,

                        CAST(
                            popular_score
                            AS FLOAT
                        ) AS popular_score,

                        popular_rank,

                        CAST(
                            similarity_score
                            AS FLOAT
                        ) AS similarity_score,

                        similarity_rank,

                        CAST(
                            similarity_sources
                            AS SMALLINT
                        ) AS similarity_sources,

                        CAST(
                            CASE
                                WHEN
                                    als_score
                                    IS NOT NULL
                                THEN 1
                                ELSE 0
                            END
                            AS TINYINT
                        ) AS from_als,

                        CAST(
                            CASE
                                WHEN
                                    popular_score
                                    IS NOT NULL
                                THEN 1
                                ELSE 0
                            END
                            AS TINYINT
                        ) AS from_popular,

                        CAST(
                            CASE
                                WHEN
                                    similarity_score
                                    IS NOT NULL
                                THEN 1
                                ELSE 0
                            END
                            AS TINYINT
                        ) AS from_similar,

                        CAST(
                            LN(
                                1
                                + popular_score
                            )
                            AS FLOAT
                        ) AS popular_score_log

                    FROM read_parquet(
                        '{CANDIDATES_ALS_PATH}'
                    )
                )

                TO '{CANDIDATES_RESULT_PATH}'

                (
                    FORMAT PARQUET,
                    COMPRESSION SNAPPY,
                    ROW_GROUP_SIZE
                        {ROW_GROUP_SIZE}
                )
                """
            )

            check_file(
                CANDIDATES_RESULT_PATH
            )

            remove_if_exists(
                CANDIDATES_ALS_PATH
            )

            print(
                "Готово:",
                CANDIDATES_RESULT_PATH
            )


            # =================================================
            # METRICS
            #
            # Считаем DuckDB, не грузим 200+ млн строк в RAM.
            # =================================================

            metrics = con.execute(
                f"""
                SELECT
                    COUNT(*)
                        AS candidates_count,

                    AVG(from_als)
                        AS from_als_share,

                    AVG(from_popular)
                        AS from_popular_share,

                    AVG(from_similar)
                        AS from_similar_share,

                    AVG(
                        CASE
                            WHEN als_score
                                IS NOT NULL
                            THEN 1.0
                            ELSE 0.0
                        END
                    ) AS als_score_notna_share,

                    AVG(
                        CASE
                            WHEN popular_score
                                IS NOT NULL
                            THEN 1.0
                            ELSE 0.0
                        END
                    ) AS popular_score_notna_share,

                    AVG(
                        CASE
                            WHEN similarity_score
                                IS NOT NULL
                            THEN 1.0
                            ELSE 0.0
                        END
                    ) AS similarity_score_notna_share

                FROM read_parquet(
                    '{CANDIDATES_RESULT_PATH}'
                )
                """
            ).fetchone()


            return {

                "candidates_count":
                    int(metrics[0]),

                "from_als_share":
                    float(metrics[1]),

                "from_popular_share":
                    float(metrics[2]),

                "from_similar_share":
                    float(metrics[3]),

                "als_score_notna_share":
                    float(metrics[4]),

                "popular_score_notna_share":
                    float(metrics[5]),

                "similarity_score_notna_share":
                    float(metrics[6]),
            }

        finally:

            con.close()


    # ========================================================
    # 3. UPLOAD FINAL RESULT TO S3
    # ========================================================

    @task
    def upload_candidates():

        check_file(
            CANDIDATES_RESULT_PATH
        )

        s3, bucket_name = (
            get_s3_client()
        )

        s3_key = (
            f"{S3_CANDIDATES_PREFIX}/"
            "candidates_result.parquet"
        )

        s3.upload_file(
            str(
                CANDIDATES_RESULT_PATH
            ),
            bucket_name,
            s3_key,
        )

        print(
            f"Загружено: "
            f"s3://{bucket_name}/{s3_key}"
        )


    # ========================================================
    # 4. LOG TO MLFLOW THROUGH REST API
    # ========================================================

    @task
    def log_mlflow(
        metrics: dict,
    ):

        # ----------------------------------------------------
        # Найти experiment
        # ----------------------------------------------------

        response = requests.get(
            (
                f"{MLFLOW_TRACKING_URI}"
                "/api/2.0/mlflow/"
                "experiments/get-by-name"
            ),
            params={
                "experiment_name":
                    MLFLOW_EXPERIMENT
            },
            timeout=30,
        )


        if response.status_code == 200:

            experiment_id = (
                response.json()[
                    "experiment"
                ][
                    "experiment_id"
                ]
            )

        else:

            # ----------------------------------------------
            # Создать experiment
            # ----------------------------------------------

            response = requests.post(
                (
                    f"{MLFLOW_TRACKING_URI}"
                    "/api/2.0/mlflow/"
                    "experiments/create"
                ),
                json={
                    "name":
                        MLFLOW_EXPERIMENT
                },
                timeout=30,
            )

            response.raise_for_status()

            experiment_id = (
                response.json()[
                    "experiment_id"
                ]
            )


        # ----------------------------------------------------
        # Создать run
        # ----------------------------------------------------

        start_time = int(
            time.time() * 1000
        )

        response = requests.post(
            (
                f"{MLFLOW_TRACKING_URI}"
                "/api/2.0/mlflow/"
                "runs/create"
            ),
            json={
                "experiment_id":
                    experiment_id,

                "start_time":
                    start_time,

                "tags": [
                    {
                        "key":
                            "mlflow.runName",

                        "value":
                            "candidate_generation",
                    }
                ],
            },
            timeout=30,
        )

        response.raise_for_status()

        run_id = (
            response.json()[
                "run"
            ][
                "info"
            ][
                "run_id"
            ]
        )


        # ----------------------------------------------------
        # Params
        # ----------------------------------------------------

        params_to_log = {

            "history_items_per_user":
                HISTORY_ITEMS,

            "similar_top_n":
                SIMILAR_TOP_N,

            "popular_top_n":
                POPULAR_TOP_N,

            "evaluation_days":
                EVALUATION_DAYS,

            "ranker_days":
                RANKER_DAYS,

            "row_group_size":
                ROW_GROUP_SIZE,
        }


        # ----------------------------------------------------
        # Log batch
        # ----------------------------------------------------

        response = requests.post(
            (
                f"{MLFLOW_TRACKING_URI}"
                "/api/2.0/mlflow/"
                "runs/log-batch"
            ),
            json={

                "run_id":
                    run_id,

                "params": [
                    {
                        "key":
                            key,

                        "value":
                            str(value),
                    }
                    for key, value
                    in params_to_log.items()
                ],

                "metrics": [
                    {
                        "key":
                            key,

                        "value":
                            float(value),

                        "timestamp":
                            int(
                                time.time()
                                * 1000
                            ),

                        "step":
                            0,
                    }
                    for key, value
                    in metrics.items()
                ],
            },
            timeout=30,
        )

        response.raise_for_status()

        print(
            f"MLflow run создан: "
            f"{run_id}"
        )


    # ========================================================
    # GRAPH
    # ========================================================

    download = (
        download_inputs()
    )

    metrics = (
        build_candidates()
    )

    upload = (
        upload_candidates()
    )

    mlflow_log = (
        log_mlflow(
            metrics
        )
    )


    download >> metrics

    metrics >> upload

    upload >> mlflow_log


recsys_candidates_pipeline()