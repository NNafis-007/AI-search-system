# embedding_models.py
from fastembed import TextEmbedding, SparseTextEmbedding

class EmbeddingModels:
    def __init__(self, sparse_name: str, dense_name: str, batch_size: int):
        self.batch_size = batch_size
        self.sparse = SparseTextEmbedding(model_name=sparse_name, batch_size=batch_size)
        self.dense = TextEmbedding(model_name=dense_name, batch_size=batch_size)

    def get_sparse(self, texts: list[str]) -> list:
        return list(self.sparse.embed(texts, batch_size=self.batch_size))

    def get_dense(self, texts: list[str]) -> list:
        return list(self.dense.embed(texts, batch_size=self.batch_size))