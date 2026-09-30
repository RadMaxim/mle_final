from datetime import datetime, timedelta
from pathlib import Path

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


MLFLOW_TRACKING_URI = params[
    "mlflow"
]["tracking_uri"]


# ============================================================
# AIRFLOW
# ============================================================

default_args = {
    "owner":
        params["airflow"]["owner"],

    "retries": 0,
}


# ============================================================
# DAG
# ============================================================

@dag(
    dag_id="test_mlflow_pipeline",
    description="Проверка подключения Airflow к MLflow",
    default_args=default_args,
    start_date=datetime(
        2026,
        9,
        1,
    ),
    schedule=None,
    catchup=False,
    tags=[
        "test",
        "mlflow",
    ],
)
def test_mlflow_pipeline():


    # ========================================================
    # TEST MLFLOW
    # ========================================================

    @task
    def send_test_run():

        import mlflow

        print(
            "MLflow tracking URI:",
            MLFLOW_TRACKING_URI
        )

        # ----------------------------------------------------
        # Подключаемся к MLflow
        # ----------------------------------------------------

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        # ----------------------------------------------------
        # Experiment
        # ----------------------------------------------------

        experiment_name = (
            "airflow_mlflow_test"
        )

        print(
            "Experiment:",
            experiment_name
        )

        mlflow.set_experiment(
            experiment_name
        )


        # ----------------------------------------------------
        # Run
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name="airflow_test_run"
        ) as run:

            print(
                "Run ID:",
                run.info.run_id
            )

            # -----------------------------------------------
            # PARAMS
            # -----------------------------------------------

            mlflow.log_params({
                "source":
                    "airflow",

                "test_type":
                    "connection_test",

                "model":
                    "none",
            })


            # -----------------------------------------------
            # METRICS
            # -----------------------------------------------

            mlflow.log_metrics({
                "test_metric":
                    1.0,

                "rows":
                    100.0,

                "accuracy":
                    0.95,
            })


            # -----------------------------------------------
            # TAGS
            # -----------------------------------------------

            mlflow.set_tags({
                "dag":
                    "test_mlflow_pipeline",

                "environment":
                    "airflow",

                "purpose":
                    "mlflow_connection_test",
            })


        print(
            "Тестовый MLflow run "
            "успешно отправлен"
        )


    send_test_run()


test_mlflow_pipeline()