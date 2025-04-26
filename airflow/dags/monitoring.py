import os
import numpy as np
import psycopg2
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from datetime import datetime, timedelta
from embedding_models import EmbeddingModels
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from evidently import ColumnMapping
from evidently.test_suite import TestSuite
# from evidently.test_preset import EmbeddingsDriftPreset
from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset
from evidently.metrics import EmbeddingsDriftMetric
from evidently.metrics.data_drift.embedding_drift_methods import model, distance, ratio, mmd
from evidently.tests import TestEmbeddingsDrift
from evidently.test_preset import DataDriftTestPreset, NoTargetPerformanceTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.workspace.cloud import CloudWorkspace
import boto3
from botocore.config import Config

load_dotenv()

# ---- Configuration ---- #
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "queries"
DATABASE_URL = os.getenv("DATABASE_URL")

EVIDENTLY_KEY = os.getenv("EVIDENTLY_KEY")
EVIDENTLY_PROJECT_ID = os.getenv("EVIDENTLY_PROJECT_ID")

PG_CONN_PARAMS = {
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
    "dbname": os.getenv("PG_DBNAME"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
}

QUERY_LOOKBACK_DAYS = 1  # Compare last 30 days with earlier
DRIFT_THRESHOLD = 0.05  # 5% increase in distance means drift

# ---- Embedding Setup ---- #
embedding_model = EmbeddingModels()
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

s3 = boto3.client(
            "s3",
            aws_access_key_id="AKIAWVZBUMBFCLDHCBHP",
            aws_secret_access_key="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",
            config=Config(signature_version="s3v4")
        )

def describe_product(row):
    category = row.get('category', '') or ''
    brand = row.get('brand', '') or ''
    title = row.get('title', '') or ''
    description = row.get('description', '') or ''
    price = row.get('price', '') or ''
    specs = row.get('specTableContent', '') or ''

    parts = []
    if title:
        parts.append(f"The product is '{title}'")
    else:
        parts.append("This product")

    if brand:
        parts.append(f"from {brand}")

    if category:
        parts.append(f"in the {category} category")

    sentence = " ".join(parts) + "."

    if description:
        sentence += f" Description: {description}"

    if specs:
        sentence += f" Specifications: {specs}"

    if price:
        sentence += f" It is priced at {price}."

    return sentence.strip()

# ---- Load Embeddings ---- #
def get_query_embeddings():
    # with psycopg2.connect(**PG_CONN_PARAMS) as conn:
    #     df = pd.read_sql("""
    #         SELECT id, text, created_at
    #         FROM user_queries
    #         ORDER BY created_at ASC
    #     """, conn)
        
    #     df["embedding"] = df["text"].apply(lambda x: embedding_model.get_dense_embeddings([x])[0])
    #     return df
    response = qdrant_client.scroll(collection_name="user_queries", limit=10000, with_vectors=True)
    print(response)
    points = response[0]
    rows = []
    for p in points:
        payload = p.payload or {}
        vectors = p.vector or {}
        row = {
            "id": p.id,
            "timestamp": payload.get("created_at"),
            "dense_vector": vectors.get("text-dense")
        }
        rows.append(row)

    print(rows)
    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(by="created_at", ascending=True).reset_index(drop=True)
    return df_sorted

def get_product_embeddings():
    # with psycopg2.connect(**PG_CONN_PARAMS) as conn:
    #     df = pd.read_sql("""
    #         SELECT *
    #         FROM products
    #         ORDER BY created_at ASC
    #     """, conn)
    #     # df_text = pd.DataFrame()
    #     df["text"] = df.apply(describe_product, axis=1)
    #     df["embedding"] = df["text"].apply(lambda x: embedding_model.get_dense_embeddings([x])[0])
    #     return df
    response = qdrant_client.scroll(collection_name="products", limit=10000, with_vectors=True, with_payload=True)
    points = response[0]
    rows = []
    # i = 0
    for p in points:
        payload = p.payload or {}
        vectors = p.vector or {}
        # if i < 1:
        #     print(p.payload)
        #     i += 1
        row = {
            "id": p.id,
            "timestamp": payload.get("created_at"),
            "dense_vector": vectors.get("text-dense"),
            "sparse_vector": vectors.get("text-sparse")
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(by="timestamp", ascending=True).reset_index(drop=True)
    return df_sorted
    
def get_product_centroid():
    response = qdrant_client.scroll(collection_name="products", limit=10000, with_vectors=True)
    dense_vectors = [point.vector["text-dense"] for point in response[0]]
    return np.mean(dense_vectors, axis=0)

# ---- Cosine Distance Drift ---- #
def compute_avg_cosine_distance(vectors, reference):
    return np.mean([cosine(vec, reference) for vec in vectors])

# ---- Drift Calculation ---- #
def calculate_and_log_drift():
    # query_df = get_query_embeddings()
    product_df = get_product_embeddings()
    product_centroid = get_product_centroid()

    # cutoff = datetime.utcnow() - timedelta(days=QUERY_LOOKBACK_DAYS)
    # historical = df[df["created_at"] < cutoff]
    # recent = df[df["created_at"] >= cutoff]

    # if len(historical) < 10 or len(recent) < 10:
    #     print("Not enough data to compute drift.")
    #     return
    # historical_query = query_df[:60]
    # recent_query = query_df[60:]
    # hist_query_vectors = np.stack(historical_query["embedding"].values)
    # recent_query_vectors = np.stack(recent_query["embedding"].values)

    # hist_mean = np.mean(hist_query_vectors, axis=0)
    # recent_mean = np.mean(recent_query_vectors, axis=0)

    # l2_drift = np.linalg.norm(hist_mean - recent_mean)
    # print(f"L2 drift between historical and recent mean embeddings: {l2_drift:.4f}")

    # hist_cos = compute_avg_cosine_distance(hist_query_vectors, product_centroid)
    # recent_cos = compute_avg_cosine_distance(recent_query_vectors, product_centroid)
    # cos_drift = recent_cos - hist_cos
    # print(f"Mean cosine distance drift (recent - historical): {cos_drift:.4f}")

    # if cos_drift > DRIFT_THRESHOLD:
    #     print("⚠️ Drift detected. Consider triggering model retraining.")
    # else:
    #     print("✅ No significant drift detected.")

    # hist_query_embed_df = pd.DataFrame(hist_query_vectors, columns=[f"dim_{i}" for i in range(hist_query_vectors.shape[1])])
    # recent_query_embed_df = pd.DataFrame(recent_query_vectors, columns=[f"dim_{i}" for i in range(recent_query_vectors.shape[1])])
    
    # column_mapping = ColumnMapping(
    #     embeddings={"full_embedding": hist_query_embed_df.columns.tolist()}
    # )

    # report = Report(metrics=[
    #     EmbeddingsDriftMetric("full_embedding")
    # ])
    # report.run(reference_data=hist_query_embed_df, current_data=recent_query_embed_df, column_mapping=column_mapping)
    # report.save_html("query_embedding_drift_report.html")

    hist_product = product_df[:8000]
    recent_product = product_df[8000:]
    hist_product_vectors = np.stack(hist_product["dense_vector"].values)
    recent_product_vectors = np.stack(recent_product["dense_vector"].values)
    hist_product_embed_df = pd.DataFrame(hist_product_vectors, columns=[f"dim_{i}" for i in range(hist_product_vectors.shape[1])])
    recent_product_embed_df = pd.DataFrame(recent_product_vectors, columns=[f"dim_{i}" for i in range(recent_product_vectors.shape[1])])
    
    column_mapping = ColumnMapping(
        embeddings={"full_embedding": hist_product_embed_df.columns.tolist()}
    )

    report = Report(metrics=[
        EmbeddingsDriftMetric("full_embedding")
    ])
    report.run(reference_data=hist_product_embed_df, current_data=recent_product_embed_df, column_mapping=column_mapping)
    result = report.as_dict()
    result = result["metrics"][0]["result"]["drift_detected"]
    print(result)
    report.save_html("product_embedding_drift_report.html")

    # with open("query_embedding_drift_report.html", 'rb') as file_data:
    #     html_content = file_data.read()
    # bucket_name = 'poridhi-manat-bucket'
    # s3_key = 'reports/query_embedding_drift_report.html'

    # try:
    #     s3.put_object(Bucket=bucket_name, Key=s3_key, Body=html_content, ContentType='text/html')
    # except Exception as e:
    #     print(f"Failed to upload file : {e}")

    with open("product_embedding_drift_report.html", 'rb') as file_data:
        html_content = file_data.read()
    bucket_name = 'poridhi-manat-bucket'
    s3_key = 'reports/product_embedding_drift_report.html'

    try:
        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=html_content, ContentType='text/html')
    except Exception as e:
        print(f"Failed to upload file : {e}")



if __name__ == "__main__":
    calculate_and_log_drift()
