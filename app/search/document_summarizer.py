"""Document summarization logic."""

import logging
from typing import Dict, Any, List
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..core.vector_database import VectorDatabaseCollection
from ..config.search_config import SearchConfig, QueryHistory
from ..prompts.search_prompts import get_document_summarization_prompt, get_web_results_summarization_prompt
from ..tools.vector_db_tools import GetChunkNeighborsTool, GetChunkByIDTool

logger = logging.getLogger(__name__)


class DocumentSummarizer:
    """Handles document summarization for different search methods.
    
    Can iteratively fetch more context via tool calls (chunk neighbors, chunk by ID)
    to produce better summaries.
    """
    
    def __init__(self, llm: ChatVertexAI, config: SearchConfig):
        self.llm = llm
        self.config = config
    
    def summarize_documents(self, query: str, collection: VectorDatabaseCollection, 
                          top_documents: Dict[str, str], metadata: Dict[str, Dict[str, Any]], 
                          query_history: QueryHistory, iteration: int = 0) -> str:
        """Iteratively summarizes documents with option to fetch more context via tool calls."""
        try:
            if iteration >= self.config.max_tool_calls_attempts:
                logger.warning(f"Max summarization attempts reached for query '{query}'")
                return f"Could not find complete answer within max_attempts ({self.config.max_tool_calls_attempts}). Partial information: {list(top_documents.keys())}"
            
            system_prompt = get_document_summarization_prompt(query, query_history.prev_tool_calls)
            
            query_context = f"""Top Documents: {top_documents}
Metadata: {metadata}"""
            
            answer = self.llm.invoke([SystemMessage(content=system_prompt),
                HumanMessage(content=query_context)], 
                tools=[GetChunkNeighborsTool, GetChunkByIDTool])
            
            query_history.total_token_usage += answer.usage_metadata['total_tokens']
            
            if not answer.tool_calls:
                return answer.content
            
            for tool_call in answer.tool_calls:
                query_history.prev_tool_calls.append(tool_call)
                logger.info(f"Document Summarizer Tool call: {tool_call['name']} with args: {tool_call['args']} for query: {query}")
                
                if tool_call['name'] == "GetChunkNeighborsTool":
                    self._process_neighbors_tool_call(tool_call, collection, top_documents, metadata)
                elif tool_call['name'] == "GetChunkByIDTool":
                    self._process_chunk_by_id_tool_call(tool_call, collection, top_documents, metadata)
            
            return self.summarize_documents(query, collection, top_documents, metadata, query_history, iteration + 1)
            
        except Exception as e:
            logger.error(f"Error summarizing documents for query '{query}': {e}", exc_info=True)
            return f"Error summarizing documents: {str(e)}"
    
    def _process_neighbors_tool_call(self, tool_call: Dict[str, Any], collection: VectorDatabaseCollection, 
                                     top_documents: Dict[str, str], metadata: Dict[str, Dict[str, Any]]) -> None:
        """Process GetChunkNeighborsTool call."""
        filename = tool_call['args']['filename']
        chunkid = tool_call['args']['chunkid']
        k = tool_call['args']['k']
        neighbors = collection.get_chunk_neighbors(filename, chunkid, k)
        for i in range(len(neighbors['documents'])):
            top_documents[neighbors['ids'][i]] = neighbors['documents'][i]
            metadata[neighbors['ids'][i]] = {
                "page": neighbors['metadatas'][i].get('page', 'unknown'),
                "section": neighbors['metadatas'][i].get('section', 'unknown'),
                "filename": neighbors['metadatas'][i].get('filename', 'unknown')
            }
    
    def _process_chunk_by_id_tool_call(self, tool_call: Dict[str, Any], collection: VectorDatabaseCollection, 
                                       top_documents: Dict[str, str], metadata: Dict[str, Dict[str, Any]]) -> None:
        """Process GetChunkByIDTool call."""
        id = tool_call['args']['id']
        chunk = collection.get_chunk_by_id(id)
        top_documents[id] = chunk['documents'][0]
        metadata[id] = {
            "page": chunk['metadatas'][0].get('page', 'unknown'),
            "section": chunk['metadatas'][0].get('section', 'unknown'),
            "filename": chunk['metadatas'][0].get('filename', 'unknown')
        }
    
    def summarize_web_results(self, query: str, top_web_results: Dict[str, Any], query_history: QueryHistory) -> str:
        """Summarizes web search results for the query."""
        try:
            system_prompt = get_web_results_summarization_prompt(query)
            query_context = f"""Web Results: {top_web_results}"""
            answer = self.llm.invoke([SystemMessage(content=system_prompt),
                HumanMessage(content=query_context)])
            query_history.total_token_usage += answer.usage_metadata['total_tokens']
            return answer.content
        except Exception as e:
            logger.error(f"Error summarizing web results for query '{query}': {e}")
            return f"Error summarizing web results: {str(e)}"
