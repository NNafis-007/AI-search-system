# mlflow_register_models.py
import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Configuration
_DEFAULT_SPARSE = "Qdrant/bm42-all-minilm-l6-v2-attentions"
_DEFAULT_DENSE = "sentence-transformers/all-MiniLM-L6-v2"
_BATCH_SIZE = 32

class SparseEmbeddingWrapper(PythonModel):
    def __init__(self):
        """Don't store model instances here - they can't be pickled"""
        self.model_name = _DEFAULT_SPARSE
        self.batch_size = _BATCH_SIZE

    def load_context(self, context):
        """Initialize model during loading"""
        from fastembed import SparseTextEmbedding  # Local import
        self.model = SparseTextEmbedding(
            model_name=self.model_name,
            batch_size=self.batch_size
        )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        texts = model_input["texts"].tolist()
        embeddings = list(self.model.embed(texts))
        return pd.DataFrame([{
            "indices": emb.indices.tolist(),
            "values": emb.values.tolist()
        } for emb in embeddings])

class DenseEmbeddingWrapper(PythonModel):
    def __init__(self):
        self.model_name = _DEFAULT_DENSE
        self.batch_size = _BATCH_SIZE

    def load_context(self, context):
        """Initialize model during loading"""
        from fastembed import TextEmbedding  # Local import
        self.model = TextEmbedding(
            model_name=self.model_name,
            batch_size=self.batch_size
        )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        texts = model_input["texts"].tolist()
        embeddings = list(self.model.embed(texts))
        return pd.DataFrame({"embedding": [emb.tolist() for emb in embeddings]})

def register_models():
    # Register sparse model
    with mlflow.start_run(run_name="sparse_embedding"):
        input_schema = Schema([ColSpec(DataType.string, "texts")])
        output_schema = Schema([
            ColSpec(DataType.integer, "indices", [None]),
            ColSpec(DataType.double, "values", [None])
        ])
        
        mlflow.pyfunc.log_model(
            python_model=SparseEmbeddingWrapper(),
            artifact_path="sparse_model",
            registered_model_name="sparse_model",
            signature=ModelSignature(inputs=input_schema, outputs=output_schema),
            input_example=pd.DataFrame({"texts": ["sample text"]}),
            code_paths=[__file__],  # Use code_paths instead of code_path
            extra_pip_requirements=["fastembed", "numpy", "nltk"]
        )

    # Register dense model
    with mlflow.start_run(run_name="dense_embedding"):
        output_schema = Schema([
            ColSpec(DataType.double, "embedding", [None])
        ])
        
        mlflow.pyfunc.log_model(
            python_model=DenseEmbeddingWrapper(),
            artifact_path="dense_model",
            registered_model_name="dense_model",
            signature=ModelSignature(
                inputs=Schema([ColSpec(DataType.string, "texts")]),
                outputs=output_schema
            ),
            input_example=pd.DataFrame({"texts": ["sample text"]}),
            code_paths=[__file__],
            extra_pip_requirements=["fastembed", "numpy"]
        )

if __name__ == "__main__":
    register_models()