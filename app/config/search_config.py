"""Configuration classes for search components."""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class SearchMethod(Enum):
    """Available search methods."""
    HYBRID = "hybrid"
    WEB = "web"

@dataclass
class SearchConfig:
    """Configuration for search operations."""
    max_tool_calls_attempts: int = 5
    max_requery_attempts: int = 3
    hybrid_search_top_k: int = 25
    rerank_top_k: int = 10
    excluded_collections: List[str] = None
    
    def __post_init__(self):
        if self.excluded_collections is None:
            self.excluded_collections = ["testing"]

@dataclass
class QueryHistory:
    """Tracks query execution history."""
    prev_search_methods: List[str]
    prev_collection_names: List[str]
    prev_tool_calls: List[dict]
    total_token_usage: int
    parent_query: Optional[str] = None
    
    def __init__(self, parent_query: Optional[str] = None):
        self.prev_search_methods = []
        self.prev_collection_names = []
        self.prev_tool_calls = []
        self.total_token_usage = 0
        self.parent_query = parent_query
