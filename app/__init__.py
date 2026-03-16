"""Legal RAG: AI-powered Legal Document Search and Analysis System

A retrieval-augmented generation system for legal document search, analysis,
and conversational Q&A using hybrid search, reranking, and LLM synthesis.
"""

from .agents import ChatAgent
from .search import SearchAgent
from .core import VectorDatabaseClient, BGEEmbeddings, BGEReranker
from .config import SearchConfig

__version__ = "1.0.0"

__all__ = [
    'ChatAgent',
    'SearchAgent',
    'VectorDatabaseClient',
    'BGEEmbeddings',
    'BGEReranker',
    'SearchConfig'
]
