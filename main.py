#asif check
import os
import pandas as pd
from typing import List, Dict, Any

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from data_processing import fetch_product_data
from embedding_models import EmbeddingModels
from QdrantSearchModel import QdrantSearchEngine
from utils import rank_list, rrf, format_search_results


# Configuration constants
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = "products"
TABLE_NAME = "mock_data"

app = FastAPI(title="Hybrid Search API")

# CORS for frontend JS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic models for API requests/responses
class SearchQuery(BaseModel):
    query: str
    limit: int = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]

# Global embedding models to avoid reloading for every request
embedding_models = None

def initialize_search_engine() -> QdrantSearchEngine:
    """Initialize the search engine with embedding models."""
    # Use global embedding models if available
    global embedding_models
    if embedding_models is None:
        embedding_models = EmbeddingModels()
    
    # Initialize search engine
    search_engine = QdrantSearchEngine(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        embedding_models=embedding_models
    )
    
    return search_engine

def load_and_embed_data() -> pd.DataFrame:
    """Load data from the database and compute embeddings."""
    # Fetch data from database
    print("Fetching car data from database...")
    data = fetch_product_data(DATABASE_URL, TABLE_NAME)
    
    if not data:
        print("No data fetched from the database.")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(data)
    print(f"Created DataFrame with {len(df)} rows.")
    
    # Compute embeddings
    global embedding_models
    if embedding_models is None:
        embedding_models = EmbeddingModels()
    df = embedding_models.add_embeddings_to_df(df)
    
    return df

def index_data(df: pd.DataFrame):
    """Index the DataFrame data into Qdrant."""
    if df.empty:
        print("No data to index.")
        return
    
    search_engine = initialize_search_engine()
    search_engine.upsert_data(df)

def hybrid_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Perform hybrid search combining dense and sparse results.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        
    Returns:
        List of search results
    """
    # Initialize search engine
    search_engine = initialize_search_engine()
    
    # Perform search
    search_results = search_engine.search(query, limit)
    
    # Extract dense and sparse results
    dense_results, sparse_results = search_results[0], search_results[1]
    
    # Rank results using RRF
    dense_rank_list = rank_list(dense_results)
    sparse_rank_list = rank_list(sparse_results)
    rrf_rank_list = rrf([dense_rank_list, sparse_rank_list])
    
    # Get final results by IDs
    
    final_results = search_engine.get_points_by_ids_from_rank_list(rrf_rank_list)
    
    # Return top results
    return final_results[:limit]

def process_results_for_api(results):
    """Convert Qdrant results to API-friendly format."""
    processed = []
    for result in results:
        processed.append({
            "id": result.id,
            "text": result.payload.get("text", ""),
            "product_id": result.payload.get("product_id", ""),
            # Add any other fields you need
        })
    return processed

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search", response_model=SearchResponse)
async def api_search(query: SearchQuery):
    """API endpoint for hybrid search."""
    results = hybrid_search(query.query, query.limit)
    processed_results = process_results_for_api(results)
    return {"results": processed_results}

@app.post("/api/index")
async def api_index_data(background_tasks: BackgroundTasks):
    """API endpoint to trigger data indexing in the background."""
    background_tasks.add_task(load_and_index_data)
    return {"status": "Indexing started in background"}

def load_and_index_data():
    """Load data from database and index it."""
    df = load_and_embed_data()
    if not df.empty:
        index_data(df)

@app.get("/api/status")
async def api_status():
    """API endpoint to check server status."""
    search_engine = initialize_search_engine()
    collection_exists = search_engine.client.collection_exists(COLLECTION_NAME)
    return {
        "status": "online",
        "collection_exists": collection_exists
    }

def run_cli():
    """Run the command line interface for the hybrid search system."""
    print("Hybrid Search System")
    print("===================")
    
    # Check if we need to load and index data
    answer = input("Do you want to load and index data? This may take some time. (y/n): ")
    if answer.lower() == 'y':
        df = load_and_embed_data()
        if not df.empty:
            print(f"Loaded {len(df)} rows of data.")
            index_data(df)
    
    # Interactive search loop
    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ")
        if query.lower() in ('exit', 'quit'):
            break
        
        results = hybrid_search(query)
        print("\nSearch Results:")
        print(format_search_results(results))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Hybrid Search System')
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode instead of API server')
    args = parser.parse_args()
    
    if args.cli:
        run_cli()
    else:
        # API mode is default - the server will be started by uvicorn
        # Run with: uvicorn main:app --reload
        print("To start the API server, run: uvicorn main:app --reload")