# search_utils.py

import os
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

from notebooks.data_processing import fetch_product_data
from embedding_models import EmbeddingModels
from QdrantSearchModel import QdrantSearchEngine
from utils import rank_list, rrf

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = "products"
TABLE_NAME = "mock_data"

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
            "text": result.payload.get("text", ""),
            "product_id": result.payload.get("product_id", ""),
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
