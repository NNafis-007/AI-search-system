from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import os
import pickle

# Set your MLflow tracking URI
MLFLOW_TRACKING_URI = "http://host.docker.internal:5000"  # Adjust if needed

def start_experiment():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("airflow_mlflow_experiment")
    print("Experiment set.")

def train_model():
    # Dummy training: fit a linear model y = 2x
    X = [i for i in range(10)]
    y = [2 * i for i in X]

    model = {"coef": 2, "intercept": 0}  # Dummy model
    os.makedirs("/tmp/model", exist_ok=True)
    model_path = "/tmp/model/dummy_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved.")
    return model_path

def log_to_mlflow(ti):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("airflow_mlflow_experiment")
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "linear_regression")
        mlflow.log_metric("mse", 0.01)

        model_path = ti.xcom_pull(task_ids="train_model")
        mlflow.log_artifact(model_path, artifact_path="models")
        
        print(f"Logged to MLflow Run ID: {run.info.run_id}")

default_args = {
    'start_date': datetime(2025, 4, 26),
}

with DAG(
    dag_id="airflow_mlflow_interaction",
    default_args=default_args,
    catchup=False,
    tags=["mlflow", "airflow"],
) as dag:

    t1 = PythonOperator(
        task_id="start_experiment",
        python_callable=start_experiment
    )

    t2 = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    t3 = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=log_to_mlflow
    )

    t1 >> t2 >> t3  # Set task dependencies