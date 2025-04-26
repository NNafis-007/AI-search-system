import subprocess
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

# Define the path to your scripts
monitoring_script_path = '/opt/airflow/dags/monitoring.py'
finetune_script_path = '/opt/airflow/dags/finetune.py'

# Function to monitor drift
def monitor_drift():
    # Use subprocess to call the monitoring script
    result = subprocess.run(
        ['python', monitoring_script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    if result.returncode != 0:
        print(f"Monitoring failed with error: {result.stderr.decode('utf-8')}")
        return False  # Drift detected or error occurred
    else:
        print(f"Monitoring completed successfully: {result.stdout.decode('utf-8')}")
        drift_result = "Drift detected" in result.stdout.decode('utf-8')
        return drift_result

# Function to fine-tune the model
def fine_tune_model():
    # If drift is detected, fine-tune the model using subprocess
    print("Drift detected. Starting model fine-tuning...")
    result = subprocess.run(
        ['python', finetune_script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        print(f"Fine-tuning failed with error: {result.stderr.decode('utf-8')}")
    else:
        print(f"Fine-tuning completed successfully: {result.stdout.decode('utf-8')}")

# Define default_args dictionary (for common arguments)
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
with DAG(
    'model_drift_monitoring_and_finetune',
    default_args=default_args,
    description='Monitor data drift and fine-tune model if drift is detected',
    schedule_interval=timedelta(days=1),  # Run daily (adjust as needed)
    start_date=datetime(2025, 4, 26),
    catchup=False,
) as dag:

    # Task to monitor drift
    monitor_drift_task = PythonOperator(
        task_id='monitor_drift',
        python_callable=monitor_drift,
    )

    # Task to fine-tune the model
    fine_tune_task = PythonOperator(
        task_id='fine_tune_model',
        python_callable=fine_tune_model,
    )

    # Set task dependencies
    monitor_drift_task >> fine_tune_task
