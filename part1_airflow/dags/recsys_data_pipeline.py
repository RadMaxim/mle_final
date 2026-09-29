from __future__ import annotations

import os
import shutil
from datetime import datetime

from airflow.decorators import dag, task

BASE_DIR = "/opt/airflow/data"
ARCHIVE_DIR = f"{BASE_DIR}/archive"
PARQUET_DIR = f"{BASE_DIR}/parquet"
OPTIMIZED_DIR = f"{BASE_DIR}/parquet_optimized"
EDA_DIR = f"{BASE_DIR}/eda"
CLEANED_DIR = f"{BASE_DIR}/parquet_cleaned"


@dag(
    dag_id="recsys_data_pipeline",
    description="CSV -> Parquet -> optimized -> EDA tables -> cleaned",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["recsys", "preprocessing", "eda"],
)
def recsys_data_pipeline():

    @task()
    def csv_to_parquet():
        import gc
        import pandas as pd

        os.makedirs(PARQUET_DIR, exist_ok=True)

        for name in [
            "category_tree",
            "events",
            "item_properties_part1",
            "item_properties_part2",
        ]:
            src = f"{ARCHIVE_DIR}/{name}.csv"
            dst = f"{PARQUET_DIR}/{name}.parquet"

            if not os.path.exists(src):
                raise FileNotFoundError(src)

            df = pd.read_csv(src)
            df.to_parquet(dst, index=False)

            del df
            gc.collect()

    @task()
    def optimize_types():
        import gc
        import pandas as pd

        os.makedirs(OPTIMIZED_DIR, exist_ok=True)

        df = pd.read_parquet(f"{PARQUET_DIR}/category_tree.parquet")
        df["categoryid"] = df["categoryid"].astype("int16")
        df["parentid"] = df["parentid"].astype("Int16")
        df.to_parquet(f"{OPTIMIZED_DIR}/category_tree.parquet", index=False)
        del df
        gc.collect()

        df = pd.read_parquet(f"{PARQUET_DIR}/events.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["visitorid"] = df["visitorid"].astype("int32")
        df["itemid"] = df["itemid"].astype("int32")
        df["transactionid"] = df["transactionid"].astype("Int32")
        df["event"] = df["event"].astype("category")
        df.to_parquet(f"{OPTIMIZED_DIR}/events.parquet", index=False)
        del df
        gc.collect()

        for name in ["item_properties_part1", "item_properties_part2"]:
            df = pd.read_parquet(f"{PARQUET_DIR}/{name}.parquet")
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["itemid"] = df["itemid"].astype("int32")
            df["property"] = df["property"].astype("category")
            df.to_parquet(f"{OPTIMIZED_DIR}/{name}.parquet", index=False)
            del df
            gc.collect()

    @task()
    def combine_item_properties():
        import duckdb

        part1 = f"{OPTIMIZED_DIR}/item_properties_part1.parquet"
        part2 = f"{OPTIMIZED_DIR}/item_properties_part2.parquet"
        output = f"{OPTIMIZED_DIR}/item_properties.parquet"

        con = duckdb.connect()
        con.execute(f"""
            COPY (
                SELECT *
                FROM read_parquet(['{part1}', '{part2}'])
            )
            TO '{output}'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        con.close()

        os.remove(part1)
        os.remove(part2)

    @task()
    def build_eda_tables():
        import duckdb

        os.makedirs(EDA_DIR, exist_ok=True)

        events_path = f"{OPTIMIZED_DIR}/events.parquet"
        props_path = f"{OPTIMIZED_DIR}/item_properties.parquet"

        con = duckdb.connect()

        con.execute(f"""
            COPY (
                WITH stats AS (
                    SELECT
                        property,
                        COUNT(DISTINCT itemid) AS items_count,
                        COUNT(DISTINCT value) AS unique_values
                    FROM read_parquet('{props_path}')
                    GROUP BY property
                ),
                totals AS (
                    SELECT COUNT(DISTINCT itemid) AS total_items
                    FROM read_parquet('{props_path}')
                )
                SELECT
                    s.property,
                    s.items_count,
                    s.unique_values,
                    s.items_count * 100.0 / t.total_items AS coverage_percent
                FROM stats s
                CROSS JOIN totals t
                ORDER BY coverage_percent DESC
            )
            TO '{EDA_DIR}/property_stats.parquet'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)

        con.execute(f"""
            COPY (
                SELECT
                    date_trunc('month', timestamp) AS month,
                    COUNT(*) AS interactions,
                    COUNT(DISTINCT visitorid) AS users,
                    COUNT(DISTINCT itemid) AS items,
                    COUNT(*) * 1.0
                        / NULLIF(COUNT(DISTINCT visitorid), 0)
                        AS interactions_per_user
                FROM read_parquet('{events_path}')
                WHERE timestamp >= TIMESTAMP '2015-06-01'
                  AND timestamp < TIMESTAMP '2015-09-01'
                GROUP BY 1
                ORDER BY 1
            )
            TO '{EDA_DIR}/monthly_stats.parquet'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)

        con.close()

    @task()
    def build_cleaned_data():
        import duckdb

        os.makedirs(CLEANED_DIR, exist_ok=True)

        events_path = f"{OPTIMIZED_DIR}/events.parquet"
        props_path = f"{OPTIMIZED_DIR}/item_properties.parquet"
        category_tree_path = f"{OPTIMIZED_DIR}/category_tree.parquet"
        property_stats_path = f"{EDA_DIR}/property_stats.parquet"

        con = duckdb.connect()

        con.execute(f"""
            COPY (
                SELECT DISTINCT *
                FROM read_parquet('{events_path}')
            )
            TO '{CLEANED_DIR}/events.parquet'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)

        con.execute(f"""
            COPY (
                SELECT p.*
                FROM read_parquet('{props_path}') p
                LEFT JOIN (
                    SELECT property
                    FROM read_parquet('{property_stats_path}')
                    WHERE coverage_percent = 100
                      AND unique_values = 1
                ) c
                USING (property)
                WHERE c.property IS NULL
            )
            TO '{CLEANED_DIR}/item_properties.parquet'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)

        con.execute(f"""
            COPY (
                SELECT *
                FROM read_parquet('{category_tree_path}')
            )
            TO '{CLEANED_DIR}/category_tree.parquet'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)

        con.close()

    @task()
    def cleanup_intermediate():
        required_files = [
            f"{CLEANED_DIR}/events.parquet",
            f"{CLEANED_DIR}/item_properties.parquet",
            f"{CLEANED_DIR}/category_tree.parquet",
        ]

        missing = [
            p for p in required_files
            if not os.path.exists(p) or os.path.getsize(p) == 0
        ]

        if missing:
            raise RuntimeError(f"Не удаляю промежуточные данные: {missing}")

        shutil.rmtree(PARQUET_DIR, ignore_errors=True)
        shutil.rmtree(OPTIMIZED_DIR, ignore_errors=True)

    t1 = csv_to_parquet()
    t2 = optimize_types()
    t3 = combine_item_properties()
    t4 = build_eda_tables()
    t5 = build_cleaned_data()
    t6 = cleanup_intermediate()

    t1 >> t2 >> t3 >> t4 >> t5 >> t6


recsys_data_pipeline()
