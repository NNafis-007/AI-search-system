# search_utils.py

import os
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
import psycopg2

from notebooks.data_processing import fetch_product_data
from embedding_models import EmbeddingModels
from QdrantSearchModel import QdrantSearchEngine
from utils import rank_list, rrf

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = "products"
TABLE_NAME = "all_products"

PG_CONN_PARAMS = {
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
    "dbname": os.getenv("PG_DBNAME"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
}

embedding_models = None

def initialize_search_engine() -> QdrantSearchEngine:
    global embedding_models
    if embedding_models is None:
        embedding_models = EmbeddingModels()

    search_engine = QdrantSearchEngine(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        embedding_models=embedding_models
    )
    return search_engine

def load_and_embed_data() -> pd.DataFrame:
    print("Fetching car data from database...")
    data = fetch_product_data(DATABASE_URL, TABLE_NAME)

    if not data:
        print("No data fetched from the database.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    print(f"Created DataFrame with {len(df)} rows.")

    global embedding_models
    if embedding_models is None:
        embedding_models = EmbeddingModels()

    df = embedding_models.add_embeddings_to_df(df)
    return df

def index_data(df: pd.DataFrame):
    if df.empty:
        print("No data to index.")
        return

    search_engine = initialize_search_engine()
    search_engine.upsert_data(df)

def hybrid_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    search_engine = initialize_search_engine()
    search_results = search_engine.search(query, limit)

    dense_results, sparse_results = search_results[0], search_results[1]
    dense_rank_list = rank_list(dense_results)
    sparse_rank_list = rank_list(sparse_results)
    rrf_rank_list = rrf([dense_rank_list, sparse_rank_list])

    final_results = search_engine.get_points_by_ids_from_rank_list(rrf_rank_list)
    return final_results[:limit]

def process_results_for_api(results):
    processed = []
    for result in results:
        processed.append({
            "id": result.id,
            "text": result.payload.get("data").get("title"),
            "product_id": result.payload.get("data").get("category"),
        })
    return processed

def load_and_index_data():
    df = load_and_embed_data()
    if not df.empty:
        index_data(df)

def run_cli():
    from utils import format_search_results

    print("Hybrid Search System")
    print("===================")

    answer = input("Do you want to load and index data? This may take some time. (y/n): ")
    if answer.lower() == 'y':
        df = load_and_embed_data()
        if not df.empty:
            print(f"Loaded {len(df)} rows of data.")
            index_data(df)

    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ")
        if query.lower() in ('exit', 'quit'):
            break

        results = hybrid_search(query)
        print("\nSearch Results:")
        print(format_search_results(results))

def store_user_query(user_query: str) -> None:
    """Store a user query string into the user_query table."""
    with psycopg2.connect(**PG_CONN_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_queries (text)
                VALUES (%s)
            """, (user_query,))
        conn.commit()

def store_query_positive(user_query: str, positive: str) -> None:
    """Store a query and positive string into the query_positive table."""
    with psycopg2.connect(**PG_CONN_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO finetune_data (query, positive)
                VALUES (%s, %s)
            """, (user_query, positive))
        conn.commit()
