"""Query processing and expansion logic."""

import logging
from typing import List, Dict, Any
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..config.search_config import QueryHistory
from ..prompts.search_prompts import get_query_expansion_prompt

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Handles query expansion and preprocessing."""
    
    def __init__(self, llm: ChatVertexAI):
        self.llm = llm
    
    def expand_query(self, query: str, query_history: QueryHistory, context: str = "") -> List[str]:
        """Expands or rephrases the query to maximize search relevance."""
        try:
            is_web_search = "WebSearchTool" in query_history.prev_search_methods
            system_prompt = get_query_expansion_prompt(is_web_search)
            query_context = f"Past conversation context: {context}\n\nUser's query: {query}" if context else f"User's query: {query}"
            response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query_context)])
            
            query_history.total_token_usage += response.usage_metadata['total_tokens']
            query_expansions = [q.strip() for q in response.content.split("\n") if q.strip()]
            return query_expansions
        except Exception as e:
            logger.error(f"Error expanding query '{query}': {e}")
            return [query]
