from datetime import datetime

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


# ============================================================
# MASTER DAG
# ============================================================

with DAG(
    dag_id="recsys_full_pipeline",
    description=(
        "Полный pipeline рекомендательной системы: "
        "data -> recommendations -> candidates -> ranker"
    ),
    start_date=datetime(
        2026,
        9,
        1,
    ),
    schedule=None,
    catchup=False,
    tags=[
        "recsys",
        "orchestration",
        "full_pipeline",
    ],
) as dag:

    # --------------------------------------------------------
    # 1. DATA PREPROCESSING
    # --------------------------------------------------------

    run_data_pipeline = TriggerDagRunOperator(
        task_id="run_data_pipeline",

        trigger_dag_id=(
            "recsys_data_pipeline"
        ),

        wait_for_completion=True,

        poke_interval=30,

        allowed_states=[
            "success",
        ],

        failed_states=[
            "failed",
        ],

        reset_dag_run=False,

        retries=0,
    )


    # --------------------------------------------------------
    # 2. RECOMMENDATIONS
    # --------------------------------------------------------

    run_recommendation_pipeline = (
        TriggerDagRunOperator(
            task_id=(
                "run_recommendation_pipeline"
            ),

            trigger_dag_id=(
                "recsys_recommendation_pipeline"
            ),

            wait_for_completion=True,

            poke_interval=30,

            allowed_states=[
                "success",
            ],

            failed_states=[
                "failed",
            ],

            reset_dag_run=False,

            retries=0,
        )
    )


    # --------------------------------------------------------
    # 3. CANDIDATES
    # --------------------------------------------------------

    run_candidates_pipeline = (
        TriggerDagRunOperator(
            task_id=(
                "run_candidates_pipeline"
            ),

            trigger_dag_id=(
                "recsys_candidates_pipeline"
            ),

            wait_for_completion=True,

            poke_interval=30,

            allowed_states=[
                "success",
            ],

            failed_states=[
                "failed",
            ],

            reset_dag_run=False,

            retries=0,
        )
    )


    # --------------------------------------------------------
    # 4. RANKER
    # --------------------------------------------------------

    run_ranker_pipeline = (
        TriggerDagRunOperator(
            task_id=(
                "run_ranker_pipeline"
            ),

            trigger_dag_id=(
                "recsys_ranker_pipeline"
            ),

            wait_for_completion=True,

            poke_interval=30,

            allowed_states=[
                "success",
            ],

            failed_states=[
                "failed",
            ],

            reset_dag_run=False,

            retries=0,
        )
    )


    # ========================================================
    # DEPENDENCIES
    # ========================================================

    (
        run_data_pipeline
        >> run_recommendation_pipeline
        >> run_candidates_pipeline
        >> run_ranker_pipeline
    )