from math import ceil
from sentence_transformers import CrossEncoder
import logging
import torch
from threading import Lock

logger = logging.getLogger(__name__)


class BGEReranker:
    """Thread-safe singleton cross-encoder reranker using BAAI/bge-reranker-large.
    
    Singleton pattern prevents OOM while allowing agents to be parallelized during search.
    Uses fp16 for better performance over quantization.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.model = CrossEncoder('BAAI/bge-reranker-large', model_kwargs={"torch_dtype": "float16"})
            self.mutex = Lock()
            self._initialized = True

    def rerank_documents(self, query: str, documents: list[str], k: int):
        """Rerank documents by relevance to query. Returns top-k results."""
        with self.mutex:
            document_ranks = self.model.rank(query, documents, return_documents=True, top_k=k, show_progress_bar=False)
        return document_ranks

    def split_document_for_reranker(self, document, query_size, document_size, reranker_max_size):
        """Split oversized documents into overlapping chunks for reranking."""
        chunk_size = ceil(len(document) / ceil(document_size / reranker_max_size))
        context_overlap = 50
        return [document[i:i + chunk_size] for i in range(0, document_size, chunk_size - context_overlap)]
