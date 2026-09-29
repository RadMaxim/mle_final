from datetime import datetime, timedelta
from pathlib import Path
import os

import boto3
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
    params["recommendations"]["work_dir"]
)

INPUT_DIR = (
    WORK_DIR
    / "input"
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
    OUTPUT_DIR
    / "top_popular.parquet"
)

PERSONAL_ALS_PATH = (
    OUTPUT_DIR
    / "personal_als.parquet"
)

SIMILAR_PATH = (
    OUTPUT_DIR
    / "similar.parquet"
)


# ============================================================
# RECOMMENDATION PARAMS
# ============================================================

TOP_N = params[
    "recommendations"
]["top_n"]

SIMILAR_N = params[
    "recommendations"
]["similar_n"]

PERSONAL_BATCH_SIZE = params[
    "recommendations"
]["personal_batch_size"]

SIMILAR_BATCH_SIZE = params[
    "recommendations"
]["similar_batch_size"]


# ============================================================
# ALS PARAMS
# ============================================================

ALS_FACTORS = params[
    "als"
]["factors"]

ALS_REGULARIZATION = params[
    "als"
]["regularization"]

ALS_ALPHA = params[
    "als"
]["alpha"]

ALS_ITERATIONS = params[
    "als"
]["iterations"]

ALS_RANDOM_STATE = params[
    "als"
]["random_state"]

ALS_CALCULATE_TRAINING_LOSS = params[
    "als"
]["calculate_training_loss"]


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


# ============================================================
# MLFLOW
# ============================================================

MLFLOW_TRACKING_URI = params[
    "mlflow"
]["tracking_uri"]

MLFLOW_EXPERIMENT = params[
    "mlflow"
]["experiment_name"]


# ============================================================
# AIRFLOW
# ============================================================

default_args = {

    "owner": params[
        "airflow"
    ]["owner"],

    "retries": params[
        "airflow"
    ]["retries"],

    "retry_delay": timedelta(
        minutes=params[
            "airflow"
        ]["retry_delay_minutes"]
    ),
}


# ============================================================
# DAG
# ============================================================

@dag(
    dag_id="recsys_recommendation_pipeline",
    description=(
        "Генерация Top Popular, "
        "Personal ALS и Similar recommendations"
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
        "als",
        "recommendations",
    ],
)
def recsys_recommendation_pipeline():


    # ========================================================
    # 1. DOWNLOAD EVENTS
    # ========================================================

    @task
    def download_events():

        INPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

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

        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        s3_key = (
            f"{S3_DATA_PREFIX}/"
            "events.parquet"
        )

        s3.download_file(
            bucket_name,
            s3_key,
            str(EVENTS_PATH),
        )

        print(
            f"Скачан: "
            f"s3://{bucket_name}/{s3_key}"
        )


    # ========================================================
    # 2. TOP POPULAR
    # ========================================================

    @task
    def build_top_popular():

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        events = pd.read_parquet(
            EVENTS_PATH
        )

        # ----------------------------------------------
        # Time split
        # ----------------------------------------------

        data_end = (
            events["timestamp"].max()
            + pd.Timedelta(
                milliseconds=1
            )
        )

        eval_start = (
            data_end
            - pd.Timedelta(
                days=EVALUATION_DAYS
            )
        )

        ranker_start = (
            eval_start
            - pd.Timedelta(
                days=RANKER_DAYS
            )
        )

        train = events[
            events["timestamp"]
            < ranker_start
        ]

        # ----------------------------------------------
        # Positive interactions
        # ----------------------------------------------

        positive_events = (
            train[
                train["event"].isin(
                    [
                        "addtocart",
                        "transaction",
                    ]
                )
            ]
            [
                [
                    "visitorid",
                    "itemid",
                ]
            ]
            .drop_duplicates()
        )

        # ----------------------------------------------
        # Top Popular
        # ----------------------------------------------

        top_popular = (
            positive_events
            .groupby(
                "itemid",
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    "size": "score"
                }
            )
            .sort_values(
                "score",
                ascending=False,
            )
            .head(TOP_N)
            .reset_index(
                drop=True
            )
        )

        top_popular["rank"] = (
            np.arange(
                1,
                len(top_popular) + 1,
                dtype=np.int16,
            )
        )

        top_popular.to_parquet(
            TOP_POPULAR_PATH,
            index=False,
        )

        print(
            f"Top Popular: "
            f"{TOP_POPULAR_PATH}"
        )


    # ========================================================
    # 3. TRAIN ALS + GENERATE RECOMMENDATIONS
    # ========================================================

    @task
    def train_and_generate_als():
        from implicit.als import AlternatingLeastSquares
        from scipy.sparse import csr_matrix
        from sklearn.preprocessing import LabelEncoder
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        events = pd.read_parquet(
            EVENTS_PATH
        )

        # ----------------------------------------------
        # Time split
        # ----------------------------------------------

        data_end = (
            events["timestamp"].max()
            + pd.Timedelta(
                milliseconds=1
            )
        )

        eval_start = (
            data_end
            - pd.Timedelta(
                days=EVALUATION_DAYS
            )
        )

        ranker_start = (
            eval_start
            - pd.Timedelta(
                days=RANKER_DAYS
            )
        )

        train = events[
            events["timestamp"]
            < ranker_start
        ].copy()

        # ----------------------------------------------
        # Encoders
        # ----------------------------------------------

        user_encoder = (
            LabelEncoder()
        )

        item_encoder = (
            LabelEncoder()
        )

        user_encoder.fit(
            train["visitorid"]
        )

        item_encoder.fit(
            train["itemid"]
        )

        train["user_idx"] = (
            user_encoder.transform(
                train["visitorid"]
            )
        )

        train["item_idx"] = (
            item_encoder.transform(
                train["itemid"]
            )
        )

        # ----------------------------------------------
        # Event weights
        # ----------------------------------------------

        user_events = (
            train
            .groupby(
                "visitorid"
            )["event"]
            .apply(set)
        )

        def conversion_to(
            target,
            source,
        ):

            has_source = (
                user_events.apply(
                    lambda values:
                    source in values
                )
            )

            has_target = (
                user_events.apply(
                    lambda values:
                    target in values
                )
            )

            return (
                (has_source & has_target)
                .sum()
                / max(
                    has_source.sum(),
                    1,
                )
            )

        p_cart_given_view = max(
            conversion_to(
                "addtocart",
                "view",
            ),
            1e-3,
        )

        event_weights = {

            "view":
                1.0,

            "addtocart":
                float(
                    np.sqrt(
                        1.0
                        / p_cart_given_view
                    )
                ),

            "transaction":
                float(
                    np.sqrt(
                        1.0
                        / p_cart_given_view
                    )
                    * 1.3
                ),
        }

        # ----------------------------------------------
        # ALS training dataframe
        # ----------------------------------------------

        train_als = train[
            [
                "user_idx",
                "item_idx",
                "event",
            ]
        ].copy()

        train_als["weight"] = (
            train_als[
                "event"
            ]
            .map(
                event_weights
            )
            .astype(
                "float32"
            )
        )

        # ----------------------------------------------
        # Sparse matrix
        # ----------------------------------------------

        user_item_matrix = csr_matrix(
            (
                train_als[
                    "weight"
                ].to_numpy(
                    dtype=np.float32
                ),

                (
                    train_als[
                        "user_idx"
                    ].to_numpy(
                        dtype=np.int32
                    ),

                    train_als[
                        "item_idx"
                    ].to_numpy(
                        dtype=np.int32
                    ),
                ),
            ),

            shape=(
                len(
                    user_encoder.classes_
                ),
                len(
                    item_encoder.classes_
                ),
            ),
        )

        del train_als

        # ----------------------------------------------
        # Train ALS
        # ----------------------------------------------

        model = (
            AlternatingLeastSquares(
                factors=ALS_FACTORS,
                regularization=(
                    ALS_REGULARIZATION
                ),
                alpha=ALS_ALPHA,
                iterations=(
                    ALS_ITERATIONS
                ),
                random_state=(
                    ALS_RANDOM_STATE
                ),
                calculate_training_loss=(
                    ALS_CALCULATE_TRAINING_LOSS
                ),
            )
        )

        model.fit(
            user_item_matrix
        )

        # ====================================================
        # PERSONAL ALS
        # ====================================================

        users = np.arange(
            user_item_matrix.shape[0],
            dtype=np.int32,
        )

        writer = None

        try:

            for start in range(
                0,
                len(users),
                PERSONAL_BATCH_SIZE,
            ):

                end = min(
                    start
                    + PERSONAL_BATCH_SIZE,

                    len(users),
                )

                batch_users = (
                    users[
                        start:end
                    ]
                )

                item_ids_enc, scores = (
                    model.recommend(
                        userid=batch_users,

                        user_items=(
                            user_item_matrix[
                                batch_users
                            ]
                        ),

                        N=TOP_N,

                        filter_already_liked_items=False,
                    )
                )

                batch_size = len(
                    batch_users
                )

                original_user_ids = (
                    user_encoder
                    .classes_[
                        batch_users
                    ]
                )

                original_item_ids = (
                    item_encoder
                    .classes_[
                        item_ids_enc.reshape(
                            -1
                        )
                    ]
                )

                batch = pd.DataFrame({

                    "visitorid":
                        np.repeat(
                            original_user_ids,
                            TOP_N,
                        ),

                    "itemid":
                        original_item_ids,

                    "score":
                        scores
                        .reshape(-1)
                        .astype(
                            np.float32
                        ),

                    "rank":
                        np.tile(
                            np.arange(
                                1,
                                TOP_N + 1,
                                dtype=np.int16,
                            ),
                            batch_size,
                        ),
                })

                table = (
                    pa.Table
                    .from_pandas(
                        batch,
                        preserve_index=False,
                    )
                )

                if writer is None:

                    writer = (
                        pq.ParquetWriter(
                            PERSONAL_ALS_PATH,
                            table.schema,
                            compression="snappy",
                        )
                    )

                writer.write_table(
                    table
                )

                del batch
                del table
                del item_ids_enc
                del scores

                print(
                    f"Personal ALS: "
                    f"{end:,}/{len(users):,}"
                )

        finally:

            if writer is not None:
                writer.close()

        # ====================================================
        # SIMILAR ITEMS
        # ====================================================

        items = np.arange(
            len(
                item_encoder.classes_
            ),
            dtype=np.int32,
        )

        writer = None

        try:

            for start in range(
                0,
                len(items),
                SIMILAR_BATCH_SIZE,
            ):

                end = min(
                    start
                    + SIMILAR_BATCH_SIZE,

                    len(items),
                )

                batch_items = (
                    items[
                        start:end
                    ]
                )

                similar_ids, scores = (
                    model.similar_items(
                        itemid=batch_items,
                        N=SIMILAR_N + 1,
                    )
                )

                source_ids = []
                target_ids = []
                batch_scores = []
                ranks = []

                for (
                    row_idx,
                    source_item_idx,
                ) in enumerate(
                    batch_items
                ):

                    row_ids = (
                        similar_ids[
                            row_idx
                        ]
                    )

                    row_scores = (
                        scores[
                            row_idx
                        ]
                    )

                    mask = (
                        row_ids
                        != source_item_idx
                    )

                    row_ids = (
                        row_ids[
                            mask
                        ][:SIMILAR_N]
                    )

                    row_scores = (
                        row_scores[
                            mask
                        ][:SIMILAR_N]
                    )

                    count = len(
                        row_ids
                    )

                    source_ids.extend(
                        [source_item_idx]
                        * count
                    )

                    target_ids.extend(
                        row_ids
                    )

                    batch_scores.extend(
                        row_scores
                    )

                    ranks.extend(
                        range(
                            1,
                            count + 1,
                        )
                    )

                source_ids = (
                    np.asarray(
                        source_ids,
                        dtype=np.int32,
                    )
                )

                target_ids = (
                    np.asarray(
                        target_ids,
                        dtype=np.int32,
                    )
                )

                batch = pd.DataFrame({

                    "itemid":
                        item_encoder
                        .classes_[
                            source_ids
                        ],

                    "similar_itemid":
                        item_encoder
                        .classes_[
                            target_ids
                        ],

                    "score":
                        np.asarray(
                            batch_scores,
                            dtype=np.float32,
                        ),

                    "rank":
                        np.asarray(
                            ranks,
                            dtype=np.int16,
                        ),
                })

                table = (
                    pa.Table
                    .from_pandas(
                        batch,
                        preserve_index=False,
                    )
                )

                if writer is None:

                    writer = (
                        pq.ParquetWriter(
                            SIMILAR_PATH,
                            table.schema,
                            compression="snappy",
                        )
                    )

                writer.write_table(
                    table
                )

                del batch
                del table
                del similar_ids
                del scores

                print(
                    f"Similar: "
                    f"{end:,}/{len(items):,}"
                )

        finally:

            if writer is not None:
                writer.close()


        # ----------------------------------------------
        # Small values -> XCom
        # ----------------------------------------------

        return {

            "p_cart_given_view":
                float(
                    p_cart_given_view
                ),

            "weight_view":
                float(
                    event_weights[
                        "view"
                    ]
                ),

            "weight_addtocart":
                float(
                    event_weights[
                        "addtocart"
                    ]
                ),

            "weight_transaction":
                float(
                    event_weights[
                        "transaction"
                    ]
                ),

            "train_users":
                int(
                    len(
                        user_encoder.classes_
                    )
                ),

            "train_items":
                int(
                    len(
                        item_encoder.classes_
                    )
                ),

            "interactions":
                int(
                    user_item_matrix.nnz
                ),
        }


    # ========================================================
    # 4. UPLOAD RESULTS TO S3
    # ========================================================

    @task
    def upload_recommendations():

        bucket_name = os.getenv(
            "S3_BUCKET_NAME"
        )

        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=os.getenv(
                "AWS_ACCESS_KEY_ID"
            ),
            aws_secret_access_key=os.getenv(
                "AWS_SECRET_ACCESS_KEY"
            ),
        )

        files = {

            TOP_POPULAR_PATH:
                (
                    f"{S3_RECOMMENDATIONS_PREFIX}/"
                    "top_popular.parquet"
                ),

            PERSONAL_ALS_PATH:
                (
                    f"{S3_RECOMMENDATIONS_PREFIX}/"
                    "personal_als.parquet"
                ),

            SIMILAR_PATH:
                (
                    f"{S3_RECOMMENDATIONS_PREFIX}/"
                    "similar.parquet"
                ),
        }

        for local_path, s3_key in (
            files.items()
        ):

            if not local_path.exists():

                raise FileNotFoundError(
                    f"Не найден: "
                    f"{local_path}"
                )

            s3.upload_file(
                str(local_path),
                bucket_name,
                s3_key,
            )

            print(
                f"Загружено: "
                f"s3://{bucket_name}/"
                f"{s3_key}"
            )


    # ========================================================
    # 5. LOG MLFLOW
    # ========================================================

    @task
    def log_mlflow(
        als_stats: dict,
    ):
        import mlflow


        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        mlflow.set_experiment(
            MLFLOW_EXPERIMENT
        )

        with mlflow.start_run(
            run_name=(
                "als_recommendation_generation"
            )
        ):

            # ------------------------------------------
            # ALS params
            # ------------------------------------------

            mlflow.log_params({

                "als_factors":
                    ALS_FACTORS,

                "als_regularization":
                    ALS_REGULARIZATION,

                "als_alpha":
                    ALS_ALPHA,

                "als_iterations":
                    ALS_ITERATIONS,

                "als_random_state":
                    ALS_RANDOM_STATE,

                "top_n":
                    TOP_N,

                "similar_n":
                    SIMILAR_N,
            })

            # ------------------------------------------
            # Event weights
            # ------------------------------------------

            mlflow.log_params({

                "weight_view":
                    als_stats[
                        "weight_view"
                    ],

                "weight_addtocart":
                    als_stats[
                        "weight_addtocart"
                    ],

                "weight_transaction":
                    als_stats[
                        "weight_transaction"
                    ],
            })

            # ------------------------------------------
            # Metrics
            # ------------------------------------------

            mlflow.log_metrics({

                "p_cart_given_view":
                    als_stats[
                        "p_cart_given_view"
                    ],

                "train_users":
                    als_stats[
                        "train_users"
                    ],

                "train_items":
                    als_stats[
                        "train_items"
                    ],

                "interactions":
                    als_stats[
                        "interactions"
                    ],
            })


    # ========================================================
    # GRAPH
    # ========================================================

    download = (
        download_events()
    )

    popular = (
        build_top_popular()
    )

    als_stats = (
        train_and_generate_als()
    )

    upload = (
        upload_recommendations()
    )

    mlflow_log = (
        log_mlflow(
            als_stats
        )
    )


    download >> [
        popular,
        als_stats,
    ]

    [
        popular,
        als_stats,
    ] >> upload

    upload >> mlflow_log


recsys_recommendation_pipeline()