# model/bento/service.py
from pathlib import Path
import bentoml
from fastembed import TextEmbedding, SparseTextEmbedding

_DEFAULT_SPARSE = "Qdrant/bm42-all-minilm-l6-v2-attentions"
_DEFAULT_DENSE = "sentence-transformers/all-MiniLM-L6-v2"
_BATCH_SIZE = 32

@bentoml.service(  # <-- This decorator is CRUCIAL
    traffic={"timeout": 60},
    workers=1
)
class EmbeddingService:
    def __init__(self):
        self.sparse_model = SparseTextEmbedding(
            model_name=_DEFAULT_SPARSE,
            batch_size=_BATCH_SIZE
        )
        self.dense_model = TextEmbedding(
            model_name=_DEFAULT_DENSE,
            batch_size=_BATCH_SIZE
        )

    @bentoml.api
    def embed(self, texts: list[str]) -> dict[str, list]:
        """POST /embed with {"texts": ["doc1", ...]}"""
        sparse = list(self.sparse_model.embed(texts))
        dense = list(self.dense_model.embed(texts))
        return {
            "sparse": sparse,
            "dense": [vec.tolist() for vec in dense],
        }

if __name__ == "__main__":
    svc = EmbeddingService()
    print("Test:", svc.embed(["sample text"]))