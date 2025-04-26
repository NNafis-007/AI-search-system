# EmbeddingClient.py
import numpy as np
import requests
from typing import List
from fastembed import SparseEmbedding  # For type compatibility
from pydantic import BaseModel

class SparseEmbeddingResponse(BaseModel):
    indices: List[int]
    values: List[float]

class EmbeddingModels:
    def __init__(self, service_url: str = 'http://localhost:4000'):
        """Client for BentoML embedding service"""
        self.service_url = service_url.rstrip('/')
        self.batch_size = 32  # Optional: Can be passed to API calls if needed
        
    def get_sparse_embeddings(self, texts: List[str]) -> List[SparseEmbedding]:
        """Get sparse embeddings from service (matches original interface)"""
        response = requests.post(
            f"{self.service_url}/embed",
            json={"texts": texts}
        )
        response.raise_for_status()
        return [
            SparseEmbedding(
                indices=np.array(item["indices"], dtype=np.int64),
                values=np.array(item["values"], dtype=np.float32)
            ) for item in response.json()["sparse"]
        ]
    
    def get_dense_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get dense embeddings from service (matches original interface)"""
        response = requests.post(
            f"{self.service_url}/embed",
            json={"texts": texts}
        )
        response.raise_for_status()
        return [np.array(vec, dtype=np.float32) for vec in response.json()["dense"]]

    def add_embeddings_to_df(self, df, text_column: str = 'text'):
        """Same interface as before, now using service"""
        texts = df[text_column].tolist()
        
        print("Fetching sparse embeddings from service...")
        df["sparse_embedding"] = self.get_sparse_embeddings(texts)
        
        print("Fetching dense embeddings from service...")
        df["dense_embedding"] = self.get_dense_embeddings(texts)
        
        return df
# import numpy as np
# from fastembed import TextEmbedding, SparseTextEmbedding, SparseEmbedding # type: ignore

# # Default model names
# SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
# DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# class EmbeddingModels:
#     def __init__(self, sparse_model_name=SPARSE_MODEL_NAME, dense_model_name=DENSE_MODEL_NAME, batch_size=32):
#         """Initialize embedding models."""
#         self.batch_size = batch_size
#         print(f"Loading sparse model: {sparse_model_name}")
#         self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name, batch_size=batch_size)
        
#         print(f"Loading dense model: {dense_model_name}")
#         self.dense_model = TextEmbedding(model_name=dense_model_name, batch_size=batch_size)
        
#     def get_sparse_embeddings(self, texts: list[str]) -> list[SparseEmbedding]:
#         """Generate sparse embeddings for a list of texts."""
#         return list(self.sparse_model.embed(texts, batch_size=self.batch_size))
    
#     def get_dense_embeddings(self, texts: list[str]) -> list[np.ndarray]:
#         """Generate dense embeddings for a list of texts."""
#         return list(self.dense_model.embed(texts, batch_size=self.batch_size))
        
#     def add_embeddings_to_df(self, df, text_column='text'):
#         """Process a DataFrame to add sparse and dense embeddings."""
#         texts = df[text_column].tolist()
        
#         print("Generating sparse embeddings...")
#         df["sparse_embedding"] = self.get_sparse_embeddings(texts)
        
#         print("Generating dense embeddings...")
#         df["dense_embedding"] = self.get_dense_embeddings(texts)
        
#         return df