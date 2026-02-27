from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from src.wine_knn_functions import (
    load_data,
    data_preprocessing,
    build_save_model,
    load_model_elbow
)

default_args = {
    "owner": "Sai Pranav Krovvidi",
    "start_date": datetime(2026, 2, 10),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="wine_knn_pipeline",
    default_args=default_args,
    description="Wine dataset KNN elbow Airflow pipeline",
    schedule=None,
    catchup=False,
    tags=["mlops", "knn"],
) as dag:

    load_data_task = PythonOperator(
        task_id="load_data_task",
        python_callable=load_data,
    )

    data_preprocessing_task = PythonOperator(
        task_id="data_preprocessing_task",
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    build_save_model_task = PythonOperator(
        task_id="build_save_model_task",
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "wine_knn_model.pkl"],
    )

    load_model_task = PythonOperator(
        task_id="load_model_task",
        python_callable=load_model_elbow,
        op_args=["wine_knn_model.pkl", build_save_model_task.output],
    )

    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task


if __name__ == "__main__":
    dag.test()
