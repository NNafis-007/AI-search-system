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

# ---- Load Embeddings ---- #
def get_query_embeddings():
    with psycopg2.connect(**PG_CONN_PARAMS) as conn:
        df = pd.read_sql("""
            SELECT id, text, created_at
            FROM user_queries
        """, conn)
        df["embedding"] = df["text"].apply(lambda x: embedding_model.get_dense_embeddings([x])[0])
        return df
    
def get_product_centroid():
    response = qdrant_client.scroll(collection_name="products", limit=10000, with_vectors=True)
    dense_vectors = [point.vector["text-dense"] for point in response[0]]
    return np.mean(dense_vectors, axis=0)

# ---- Cosine Distance Drift ---- #
def compute_avg_cosine_distance(vectors, reference):
    return np.mean([cosine(vec, reference) for vec in vectors])

# ---- Drift Calculation ---- #
def calculate_and_log_drift():
    df = get_query_embeddings()
    product_centroid = get_product_centroid()

    # cutoff = datetime.utcnow() - timedelta(days=QUERY_LOOKBACK_DAYS)
    # historical = df[df["created_at"] < cutoff]
    # recent = df[df["created_at"] >= cutoff]

    # if len(historical) < 10 or len(recent) < 10:
    #     print("Not enough data to compute drift.")
    #     return
    historical = df[:60]
    recent = df[60:]

    hist_vectors = np.stack(historical["embedding"].values)
    recent_vectors = np.stack(recent["embedding"].values)

    hist_mean = np.mean(hist_vectors, axis=0)
    recent_mean = np.mean(recent_vectors, axis=0)

    # L2 drift
    l2_drift = np.linalg.norm(hist_mean - recent_mean)
    print(f"L2 drift between historical and recent mean embeddings: {l2_drift:.4f}")

    # Cosine centroid drift
    hist_cos = compute_avg_cosine_distance(hist_vectors, product_centroid)
    recent_cos = compute_avg_cosine_distance(recent_vectors, product_centroid)
    cos_drift = recent_cos - hist_cos
    print(f"Mean cosine distance drift (recent - historical): {cos_drift:.4f}")

    if cos_drift > DRIFT_THRESHOLD:
        print("⚠️ Drift detected. Consider triggering model retraining.")
    else:
        print("✅ No significant drift detected.")

    # suite = TestSuite(tests=[EmbeddingDriftPreset()])
    # suite.run(reference_data=historical, current_data=recent)
    # suite.save_html("embedding_drift_report.html")

    hist_embed_df = pd.DataFrame(hist_vectors, columns=[f"dim_{i}" for i in range(hist_vectors.shape[1])])
    recent_embed_df = pd.DataFrame(recent_vectors, columns=[f"dim_{i}" for i in range(recent_vectors.shape[1])])
    
    column_mapping = ColumnMapping(
        embeddings={"full_embedding": hist_embed_df.columns.tolist()}
    )

    report = Report(metrics=[
        EmbeddingsDriftMetric("full_embedding")
    ])
    report.run(reference_data=hist_embed_df, current_data=recent_embed_df, column_mapping=column_mapping)
    report.save_html("embedding_drift_report.html")

if __name__ == "__main__":
    calculate_and_log_drift()
