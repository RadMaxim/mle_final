from datetime import datetime, timedelta
from pathlib import Path
import os

import boto3
import duckdb
import yaml

from airflow.decorators import dag, task


# --------------------------------------------------
# Загружаем параметры pipeline
#
# На хосте:
#   ./params.yaml
#
# В Airflow-контейнере:
#   /opt/airflow/params.yaml
# --------------------------------------------------

PARAMS_PATH = Path("/opt/airflow/params.yaml")


if not PARAMS_PATH.exists():
    raise FileNotFoundError(
        f"Файл параметров не найден: {PARAMS_PATH}"
    )


with PARAMS_PATH.open(
    "r",
    encoding="utf-8"
) as file:

    params = yaml.safe_load(file)


# --------------------------------------------------
# Пути внутри Airflow-контейнера
# --------------------------------------------------

RAW_DIR = Path(
    params["paths"]["raw_dir"]
)

PARQUET_DIR = Path(
    params["paths"]["parquet_dir"]
)

CLEANED_DIR = Path(
    params["paths"]["cleaned_dir"]
)


# --------------------------------------------------
# Параметры preprocessing
# --------------------------------------------------

CONSTANT_PROPERTIES = (
    params["preprocessing"][
        "constant_properties"
    ]
)

PARQUET_COMPRESSION = (
    params["preprocessing"][
        "parquet_compression"
    ]
)


# --------------------------------------------------
# S3
# --------------------------------------------------

S3_ENDPOINT_URL = (
    params["s3"]["endpoint_url"]
)

S3_DATA_PREFIX = (
    params["s3"]["data_prefix"]
)

S3_FILES = (
    params["s3"]["files"]
)


# --------------------------------------------------
# Airflow
# --------------------------------------------------

default_args = {

    "owner": (
        params["airflow"]["owner"]
    ),

    "retries": (
        params["airflow"]["retries"]
    ),

    "retry_delay": timedelta(
        minutes=params["airflow"][
            "retry_delay_minutes"
        ]
    ),
}


# --------------------------------------------------
# DAG
# --------------------------------------------------

@dag(
    dag_id="recsys_data_pipeline",
    description=(
        "Подготовка исходных данных "
        "рекомендательной системы"
    ),
    default_args=default_args,
    start_date=datetime(2026, 9, 1),
    schedule=None,
    catchup=False,
    tags=[
        "recsys",
        "preprocessing"
    ],
)
def recsys_data_pipeline():


    # --------------------------------------------------
    # 1. Проверяем наличие исходных данных
    # --------------------------------------------------

    @task
    def check_raw_data():

        required_files = [
            "category_tree.csv",
            "events.csv",
            "item_properties_part1.csv",
            "item_properties_part2.csv",
        ]

        for file_name in required_files:

            path = (
                RAW_DIR
                / file_name
            )

            if not path.exists():

                raise FileNotFoundError(
                    f"Не найден входной файл: {path}"
                )

            print(
                f"{file_name}: "
                f"{path.stat().st_size / 1024**2:.2f} MB"
            )

        print(
            "Все исходные файлы найдены."
        )


    # --------------------------------------------------
    # 2. CSV -> Parquet
    # --------------------------------------------------

    @task
    def convert_to_parquet():

        PARQUET_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        files = [
            "category_tree",
            "events",
            "item_properties_part1",
            "item_properties_part2",
        ]

        con = duckdb.connect()

        try:

            for name in files:

                input_path = (
                    RAW_DIR
                    / f"{name}.csv"
                )

                output_path = (
                    PARQUET_DIR
                    / f"{name}.parquet"
                )

                con.execute(
                    f"""
                    COPY (
                        SELECT *
                        FROM read_csv_auto(
                            '{input_path}',
                            header=true
                        )
                    )
                    TO '{output_path}'
                    (
                        FORMAT PARQUET,
                        COMPRESSION {PARQUET_COMPRESSION}
                    )
                    """
                )

                print(
                    f"Создан: {output_path}"
                )

        finally:

            con.close()


    # --------------------------------------------------
    # 3. Очистка events
    # --------------------------------------------------

    @task
    def clean_events():

        CLEANED_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        source = (
            PARQUET_DIR
            / "events.parquet"
        )

        destination = (
            CLEANED_DIR
            / S3_FILES["events"]
        )

        con = duckdb.connect()

        try:

            con.execute(
                f"""
                COPY (
                    SELECT DISTINCT

                        to_timestamp(
                            timestamp / 1000.0
                        ) AS timestamp,

                        CAST(
                            visitorid AS INTEGER
                        ) AS visitorid,

                        event,

                        CAST(
                            itemid AS INTEGER
                        ) AS itemid,

                        CAST(
                            transactionid AS INTEGER
                        ) AS transactionid

                    FROM read_parquet(
                        '{source}'
                    )
                )
                TO '{destination}'
                (
                    FORMAT PARQUET,
                    COMPRESSION {PARQUET_COMPRESSION}
                )
                """
            )

        finally:

            con.close()

        print(
            f"events сохранён: "
            f"{destination}"
        )


    # --------------------------------------------------
    # 4. Очистка category_tree
    # --------------------------------------------------

    @task
    def clean_category_tree():

        CLEANED_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        source = (
            PARQUET_DIR
            / "category_tree.parquet"
        )

        destination = (
            CLEANED_DIR
            / S3_FILES["category_tree"]
        )

        con = duckdb.connect()

        try:

            con.execute(
                f"""
                COPY (
                    SELECT

                        CAST(
                            categoryid AS SMALLINT
                        ) AS categoryid,

                        CAST(
                            parentid AS SMALLINT
                        ) AS parentid

                    FROM read_parquet(
                        '{source}'
                    )
                )
                TO '{destination}'
                (
                    FORMAT PARQUET,
                    COMPRESSION {PARQUET_COMPRESSION}
                )
                """
            )

        finally:

            con.close()

        print(
            f"category_tree сохранён: "
            f"{destination}"
        )


    # --------------------------------------------------
    # 5. Очистка item_properties
    # --------------------------------------------------

    @task
    def clean_item_properties():

        CLEANED_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        part1 = (
            PARQUET_DIR
            / "item_properties_part1.parquet"
        )

        part2 = (
            PARQUET_DIR
            / "item_properties_part2.parquet"
        )

        destination = (
            CLEANED_DIR
            / S3_FILES["item_properties"]
        )

        constants = ", ".join(
            f"'{value}'"
            for value in CONSTANT_PROPERTIES
        )

        con = duckdb.connect()

        try:

            con.execute(
                f"""
                COPY (
                    SELECT

                        to_timestamp(
                            timestamp / 1000.0
                        ) AS timestamp,

                        CAST(
                            itemid AS INTEGER
                        ) AS itemid,

                        property,
                        value

                    FROM (

                        SELECT *
                        FROM read_parquet(
                            '{part1}'
                        )

                        UNION ALL

                        SELECT *
                        FROM read_parquet(
                            '{part2}'
                        )
                    )

                    WHERE property NOT IN (
                        {constants}
                    )
                )
                TO '{destination}'
                (
                    FORMAT PARQUET,
                    COMPRESSION {PARQUET_COMPRESSION}
                )
                """
            )

        finally:

            con.close()

        print(
            f"item_properties сохранён: "
            f"{destination}"
        )


    # --------------------------------------------------
    # 6. Проверка итоговых данных
    # --------------------------------------------------

    @task
    def validate_cleaned_data():

        files = {

            "events": (
                CLEANED_DIR
                / S3_FILES["events"]
            ),

            "item_properties": (
                CLEANED_DIR
                / S3_FILES["item_properties"]
            ),

            "category_tree": (
                CLEANED_DIR
                / S3_FILES["category_tree"]
            ),
        }

        con = duckdb.connect()

        try:

            for name, path in files.items():

                if not path.exists():

                    raise FileNotFoundError(
                        f"Не создан файл: {path}"
                    )

                rows = con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM read_parquet(
                        '{path}'
                    )
                    """
                ).fetchone()[0]

                if rows == 0:

                    raise ValueError(
                        f"{name} пустой"
                    )

                print(
                    f"{name}: "
                    f"{rows:,} строк"
                )

        finally:

            con.close()


    # --------------------------------------------------
    # 7. Загрузка в S3
    # --------------------------------------------------

    @task
    def upload_to_s3():

        # Секреты НЕ находятся в params.yaml.
        # Они приходят через .env -> docker compose.

        bucket_name = os.getenv(
            "S3_BUCKET_NAME"
        )

        access_key = os.getenv(
            "AWS_ACCESS_KEY_ID"
        )

        secret_key = os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )


        # ----------------------------------------------
        # Проверяем переменные окружения
        # ----------------------------------------------

        required_env = {

            "S3_BUCKET_NAME":
                bucket_name,

            "AWS_ACCESS_KEY_ID":
                access_key,

            "AWS_SECRET_ACCESS_KEY":
                secret_key,
        }

        missing = [
            name
            for name, value
            in required_env.items()
            if not value
        ]

        if missing:

            raise RuntimeError(
                "Не заданы переменные окружения: "
                + ", ".join(missing)
            )


        # ----------------------------------------------
        # S3 client
        # ----------------------------------------------

        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )


        # ----------------------------------------------
        # Файлы для загрузки
        # ----------------------------------------------

        files = {

            S3_FILES["events"]:
                f"{S3_DATA_PREFIX}/"
                f"{S3_FILES['events']}",

            S3_FILES["item_properties"]:
                f"{S3_DATA_PREFIX}/"
                f"{S3_FILES['item_properties']}",

            S3_FILES["category_tree"]:
                f"{S3_DATA_PREFIX}/"
                f"{S3_FILES['category_tree']}",
        }


        # ----------------------------------------------
        # Upload
        # ----------------------------------------------

        for local_name, s3_key in files.items():

            local_path = (
                CLEANED_DIR
                / local_name
            )

            s3.upload_file(
                str(local_path),
                bucket_name,
                s3_key
            )

            print(
                f"Загружено: "
                f"s3://{bucket_name}/{s3_key}"
            )


    # --------------------------------------------------
    # Граф зависимостей
    # --------------------------------------------------

    raw_check = check_raw_data()

    parquet = convert_to_parquet()

    events = clean_events()

    categories = (
        clean_category_tree()
    )

    properties = (
        clean_item_properties()
    )

    validation = (
        validate_cleaned_data()
    )

    upload = upload_to_s3()


    raw_check >> parquet

    parquet >> [
        events,
        categories,
        properties,
    ]

    [
        events,
        categories,
        properties,
    ] >> validation

    validation >> upload


recsys_data_pipeline()