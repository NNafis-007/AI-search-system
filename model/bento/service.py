import os
import re
import tempfile
import boto3     
import bentoml                           # AWS SDK for Python :contentReference[oaicite:4]{index=4}
import joblib                               # For loading serialized models :contentReference[oaicite:5]{index=5}
from pathlib import Path
from fastembed import TextEmbedding, SparseTextEmbedding
from botocore.config import Config

_DEFAULT_SPARSE = "Qdrant/bm42-all-minilm-l6-v2-attentions"
_DEFAULT_DENSE  = "sentence-transformers/all-MiniLM-L6-v2"
_BATCH_SIZE     = 32

def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key paths."""
    match = re.match(r"s3://([^/]+)/(.+)", uri)
    if not match:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return match.group(1), match.group(2)



@bentoml.service(
    traffic={"timeout": 60},
    workers=1
)
class EmbeddingService:
    def __init__(self):
        # Initialize S3 client
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id="AKIAWVZBUMBFCLDHCBHP",
            aws_secret_access_key="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",
            config=Config(signature_version="s3v4")
        )

        # === Sparse Model ===
        sparse_uri = "s3://poridhi-manat-bucket/sparse.onnx"
        self.is_sparse_loaded = False

        if sparse_uri:
            try:
                bucket, key = parse_s3_uri(sparse_uri)
                tmp_path = tempfile.mktemp(suffix=Path(key).suffix)  # :contentReference[oaicite:6]{index=6}
                print(tmp_path)
                self.s3.download_file(bucket, key, tmp_path)        # :contentReference[oaicite:7]{index=7}
                self.sparse_model = joblib.load(tmp_path)           # :contentReference[oaicite:8]{index=8}
                self.is_sparse_loaded = True
            except Exception as e:
                print(f"🔶 Failed to load sparse model from S3: {e}")

        if not self.is_sparse_loaded:
            # Fallback embedding
            self.sparse_model = SparseTextEmbedding(
                model_name=_DEFAULT_SPARSE,
                batch_size=_BATCH_SIZE
            )
            print("loaded from url")
            self.is_sparse_loaded = False

        # === Dense Model ===
        dense_uri = "s3://poridhi-manat-bucket/dense.onnx"
        self.is_dense_loaded = False

        if dense_uri:
            try:
                bucket, key = parse_s3_uri(dense_uri)
                tmp_path = tempfile.mktemp(suffix=Path(key).suffix)  # :contentReference[oaicite:9]{index=9}
                print(tmp_path)
                self.s3.download_file(bucket, key, tmp_path)        # :contentReference[oaicite:10]{index=10}
                self.dense_model = joblib.load(tmp_path)            # :contentReference[oaicite:11]{index=11}
                self.is_dense_loaded = True
            except Exception as e:
                print(f"🔶 Failed to load dense model from S3: {e}")

        if not self.is_dense_loaded:
            # Fallback embedding
            self.dense_model = TextEmbedding(
                model_name=_DEFAULT_DENSE,
                batch_size=_BATCH_SIZE
            )
            print("loaded from url")
            self.is_dense_loaded = False

    
    @bentoml.api
    def embed(self, texts: list[str]) -> dict[str, list]:
        """Compute sparse and dense embeddings."""
        # For raw classes, use .embed; for deserialized objects, assume .predict
        if hasattr(self.sparse_model, "predict"):
            sparse_raw = self.sparse_model.predict(texts)
        else:
            sparse_raw = list(self.sparse_model.embed(texts))

        if hasattr(self.dense_model, "predict"):
            dense_raw = self.dense_model.predict(texts)
        else:
            dense_raw = list(self.dense_model.embed(texts))

        # Post-process
        sparse = [
            {"indices": s.indices.tolist(), "values": s.values.tolist()}
            for s in sparse_raw
        ]
        dense = [d.tolist() for d in dense_raw]

        print(dense)
        return {"sparse": sparse, "dense": dense}


