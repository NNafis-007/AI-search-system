from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow


def log_to_mlflow():
    tracking_uri = mlflow.get_tracking_uri()
    print(f"Current tracking uri: {tracking_uri}")

    mlflow.set_experiment("Airflow_MLflow_Experiment")
    with mlflow.start_run():
        mlflow.log_param("param1", 5)
        mlflow.log_metric("metric1", 0.85)
        mlflow.set_tag("tag1", "Airflow_MLflow")
        print("Logged parameters and metrics to MLflow.")

with DAG(
    dag_id='mlflow_integration_dag',
    start_date=datetime(2025, 4, 26),
    # schedule_interval='@once',
    catchup=False,
) as dag:
    log_task = PythonOperator(
        task_id='log_to_mlflow',
        python_callable=log_to_mlflow,
    )