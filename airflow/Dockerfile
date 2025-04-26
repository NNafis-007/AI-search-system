FROM astrocrpublic.azurecr.io/runtime:3.0-1

RUN pip install --no-cache-dir requests apache-airflow[http] mlflow

RUN export MLFLOW_TRACKING_URI=http://localhost:5000