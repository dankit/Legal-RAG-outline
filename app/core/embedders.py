from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from threading import Lock, Semaphore


class GeminiEmbeddings:
    """Google Vertex AI embeddings (kept for reference; BGE is the primary embedder)."""

    def __init__(self):
        self.embeddings = VertexAIEmbeddings(model="gemini-embedding-001")

    async def create_document_embeddings_async(self, documents) -> list[list[float]]:
        embedded_documents = await self.embeddings.aembed_documents(documents)
        return embedded_documents

    async def create_query_embeddings_async(self, query) -> list[float]:
        embedded_query = await self.embeddings.aembed_query(query)
        return embedded_query


class BGEEmbeddings:
    """Thread-safe singleton wrapper around BAAI/bge-m3 embeddings.
    
    Uses a semaphore to limit concurrent embedding calls.
    """
    _instance = None
    _lock = Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if BGEEmbeddings._initialized:
            return
        
        with self._lock:
            if BGEEmbeddings._initialized:
                return
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": device})
            self.semaphore = Semaphore(10)
            BGEEmbeddings._initialized = True

    def create_document_embeddings(self, documents) -> list[list[float]]:
        with self.semaphore:
            return self.embeddings.embed_documents(documents)

    def create_query_embeddings(self, query) -> list[float]:
        with self.semaphore:
            return self.embeddings.embed_query(query)
