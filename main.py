# main.py

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any

from search_utils import (
    hybrid_search,
    process_results_for_api,
    load_and_index_data,
    run_cli,
    initialize_search_engine
)

app = FastAPI(title="Hybrid Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class SearchQuery(BaseModel):
    query: str
    limit: int = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search", response_model=SearchResponse)
async def api_search(query: SearchQuery):
    results = hybrid_search(query.query, query.limit)
    processed_results = process_results_for_api(results)
    return {"results": processed_results}

@app.post("/api/index")
async def api_index_data(background_tasks: BackgroundTasks):
    background_tasks.add_task(load_and_index_data)
    return {"status": "Indexing started in background"}

@app.get("/api/status")
async def api_status():
    search_engine = initialize_search_engine()
    collection_exists = search_engine.client.collection_exists("products")
    return {
        "status": "online",
        "collection_exists": collection_exists
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Hybrid Search System')
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode instead of API server')
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        print("To start the API server, run: uvicorn main:app --reload")
