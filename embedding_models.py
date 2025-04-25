import numpy as np
from fastembed import TextEmbedding, SparseTextEmbedding, SparseEmbedding

# Default model names
SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class EmbeddingModels:
    def __init__(self, sparse_model_name=SPARSE_MODEL_NAME, dense_model_name=DENSE_MODEL_NAME, batch_size=32):
        """Initialize embedding models."""
        self.batch_size = batch_size
        print(f"Loading sparse model: {sparse_model_name}")
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name, batch_size=batch_size)
        
        print(f"Loading dense model: {dense_model_name}")
        self.dense_model = TextEmbedding(model_name=dense_model_name, batch_size=batch_size)
        
    def get_sparse_embeddings(self, texts: list[str]) -> list[SparseEmbedding]:
        """Generate sparse embeddings for a list of texts."""
        return list(self.sparse_model.embed(texts, batch_size=self.batch_size))
    
    def get_dense_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Generate dense embeddings for a list of texts."""
        return list(self.dense_model.embed(texts, batch_size=self.batch_size))
        
    def add_embeddings_to_df(self, df, text_column='text'):
        """Process a DataFrame to add sparse and dense embeddings."""
        texts = df[text_column].tolist()
        
        print("Generating sparse embeddings...")
        df["sparse_embedding"] = self.get_sparse_embeddings(texts)
        
        print("Generating dense embeddings...")
        df["dense_embedding"] = self.get_dense_embeddings(texts)
        
        return df