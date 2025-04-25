# model/bento/service.py
from pathlib import Path
import bentoml
from fastembed import TextEmbedding, SparseTextEmbedding

_DEFAULT_SPARSE = "Qdrant/bm42-all-minilm-l6-v2-attentions"
_DEFAULT_DENSE = "sentence-transformers/all-MiniLM-L6-v2"
_BATCH_SIZE = 32

# service.py
@bentoml.service(
    traffic={"timeout": 60},
    workers=1
)
class EmbeddingService:
    def __init__(self):
        self.sparse_model = SparseTextEmbedding(
            model_name="Qdrant/bm42-all-minilm-l6-v2-attentions",
            batch_size=32
        )
        self.dense_model = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32
        )

    @bentoml.api  # Make sure this decorator is present
    def embed(self, texts: list[str]) -> dict[str, list]:
        """Handle POST requests to /embed"""
        sparse = list(self.sparse_model.embed(texts))
        dense = list(self.dense_model.embed(texts))
        
        return {
            "sparse": [{"indices": s.indices.tolist(), 
                       "values": s.values.tolist()} for s in sparse],
            "dense": [d.tolist() for d in dense]
        }

if __name__ == "__main__":
    svc = EmbeddingService()
    print("Test:", svc.embed(["sample text"]))