"""Search method selection logic."""

import logging
from typing import Optional, Dict, Any, List
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage
from .hybrid_search import HybridSearchTool
from ..webscrapers.web_search import WebSearchTool
from ..config.search_config import SearchConfig, QueryHistory
from ..prompts.search_prompts import get_search_method_selection_prompt

logger = logging.getLogger(__name__)


class SearchMethodSelector:
    """Determines the best search method for a query using LLM-based routing."""
    
    def __init__(self, llm: ChatVertexAI, db, config: SearchConfig):
        self.llm = llm
        self.db = db
        self.config = config
    
    def select_search_method(self, query: str, query_history: QueryHistory, iteration: int) -> Optional[Dict[str, Any]]:
        """Determines the best search method (hybrid search or web search) for the query."""
        try:
            if iteration >= self.config.max_requery_attempts:
                logger.warning(f"Max requery attempts reached for query '{query}', defaulting to WebSearchTool")
                query_history.prev_search_methods.append('WebSearchTool')
                return {'name': 'WebSearchTool', 'args': {'query': query}}
            
            collections = self._get_available_collections(query_history)
            system_prompt = get_search_method_selection_prompt(collections)
            
            llm_tool_response = self.llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=query)], 
                tools=[HybridSearchTool, WebSearchTool])
            
            query_history.total_token_usage += llm_tool_response.usage_metadata['total_tokens']

            if not llm_tool_response.tool_calls:
                logger.warning(f"No tool calls returned for query '{query}', defaulting to WebSearchTool")
                query_history.prev_search_methods.append('WebSearchTool')
                return {'name': 'WebSearchTool', 'args': {'query': query}}
                       
            tool_call = llm_tool_response.tool_calls[0]
            
            if tool_call['name'] == 'HybridSearchTool' and 'collection_name' in tool_call['args']:
                if tool_call['args']['collection_name'] not in collections:
                    logger.warning(f"Collection {tool_call['args']['collection_name']} not found, defaulting to WebSearchTool")
                    query_history.prev_search_methods.append('WebSearchTool')
                    return {'name': 'WebSearchTool', 'args': {'query': query}}
                else:
                    query_history.prev_collection_names.append(tool_call['args']['collection_name'])

            query_history.prev_search_methods.append(tool_call['name'])
            query_history.prev_tool_calls.append(tool_call)
            return tool_call
            
        except Exception as e:
            logger.error(f"Error determining search method for query '{query}': {e}")
            return {'name': 'WebSearchTool', 'args': {'query': query}}
    
    def _get_available_collections(self, query_history: QueryHistory) -> List:
        """Get available collections excluding previously tried ones."""
        collections = self.db.list_collections()
        prev_collections = query_history.prev_collection_names
        return [collection.name for collection in collections 
                if collection.name not in (self.config.excluded_collections + prev_collections)]
