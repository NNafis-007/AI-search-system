import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    SparseVector,
    PointStruct,
    SearchRequest,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    ScoredPoint,
)
from typing import List, Tuple, Dict, Any
from fastembed import SparseEmbedding # type: ignore

from embedding_models import EmbeddingModels

class QdrantSearchEngine:
    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str = "products",
        embedding_models: EmbeddingModels = None
    ):
        """Initialize the Qdrant search engine.
        
        Args:
            url: Qdrant server URL
            api_key: API key for authentication
            collection_name: Name of the collection to use
            embedding_models: EmbeddingModels instance, will create a new one if None
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.embedding_models = embedding_models or EmbeddingModels()
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
        
    def _ensure_collection_exists(self):
        """Ensure that the collection exists in Qdrant."""
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                self.collection_name,
                vectors_config={
                    "text-dense": VectorParams(
                        size=384,  # MiniLM-L6-v2 size
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "text-sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
            )
            print(f"Collection '{self.collection_name}' created successfully.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")
            
    def upsert_data(self, df: pd.DataFrame):
        """Insert or update data in the Qdrant collection.
        
        Args:
            df: DataFrame containing id, text, sparse_embedding, and dense_embedding columns
        """
        points = self._make_points(df)
        self.client.upsert(self.collection_name, points)
        print(f"Upserted {len(points)} points into collection '{self.collection_name}'.")
        
    def _make_points(self, df: pd.DataFrame) -> List[PointStruct]:
        """Convert DataFrame rows to Qdrant points."""
        sparse_vectors = df["sparse_embedding"].tolist()
        texts = df["text"].tolist()
        dense_vectors = df["dense_embedding"].tolist()
        ids = df["id"].tolist()
        
        points = []
        for idx, (row_id, text, sparse_vector, dense_vector) in enumerate(
            zip(ids, texts, sparse_vectors, dense_vectors)
        ):
            sparse_vector = SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=sparse_vector.values.tolist()
            )
            point = PointStruct(
                id=row_id,
                payload={
                    "text": text,
                    "product_id": row_id,
                },
                vector={
                    "text-sparse": sparse_vector,
                    "text-dense": dense_vector.tolist(),
                },
            )
            points.append(point)
        return points
    
    def search(self, query_text: str, limit: int = 5):
        """Search for similar items using both dense and sparse vectors.
        
        Args:
            query_text: Text query to search for
            limit: Maximum number of results to return
            
        Returns:
            List of search results from both dense and sparse searches
        """
        # Compute sparse and dense vectors
        query_sparse_vectors = self.embedding_models.get_sparse_embeddings([query_text])
        query_dense_vector = self.embedding_models.get_dense_embeddings([query_text])
        
        search_results = self.client.search_batch(
            collection_name=self.collection_name,
            requests=[
                SearchRequest(
                    vector=NamedVector(
                        name="text-dense",
                        vector=query_dense_vector[0].tolist(),
                    ),
                    limit=limit,
                    with_payload=True,
                ),
                SearchRequest(
                    vector=NamedSparseVector(
                        name="text-sparse",
                        vector=SparseVector(
                            indices=query_sparse_vectors[0].indices.tolist(),
                            values=query_sparse_vectors[0].values.tolist(),
                        ),
                    ),
                    limit=limit,
                    with_payload=True,
                ),
            ],
        )
        
        return search_results
    
    def get_points_by_ids_from_rank_list(self, rrf_rank_list: list[tuple[int, float]]):
        """Retrieve points from the collection by their IDs."""
        return self.client.retrieve(
            collection_name=self.collection_name, 
            ids=[item[0] for item in rrf_rank_list]
        )