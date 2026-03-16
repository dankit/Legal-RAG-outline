"""Result synthesis and answer generation logic."""

import logging
from typing import Dict, Any, Callable
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..config.search_config import SearchConfig, QueryHistory
from .query_processor import QueryProcessor
from ..tools.search_tools import SearchTool
from ..prompts.search_prompts import get_result_synthesis_prompt

logger = logging.getLogger(__name__)


class ResultSynthesizer:
    """Handles result synthesis and answer generation.
    
    Can re-search if initial results are insufficient, up to max_requery_attempts.
    """
    
    def __init__(self, llm: ChatVertexAI, config: SearchConfig, search_executor: Callable[[str], str] = None):
        self.llm = llm
        self.config = config
        self.search_executor = search_executor
    
    def synthesize_answer(self, query: str, queries_and_summaries: Dict[str, str], 
                         query_history: QueryHistory, iteration: int = 0) -> str:
        """Generates final answer from subquery summaries, with option to re-search if needed."""
        try:
            can_requery = iteration < self.config.max_requery_attempts
            system_prompt = get_result_synthesis_prompt(query, can_requery)
            tools = [SearchTool] if can_requery else []
            answer = self.llm.invoke([SystemMessage(content=system_prompt),
                HumanMessage(content=f"Summaries: {queries_and_summaries}")], tools=tools)
            
            query_history.total_token_usage += answer.usage_metadata['total_tokens']
            
            if not answer.tool_calls:
                return answer.content

            iteration += 1
            for tool_call in answer.tool_calls:
                if tool_call['name'] == "SearchTool":
                    self._handle_research_tool_call(tool_call, query, queries_and_summaries, query_history, iteration)
            return self.synthesize_answer(query, queries_and_summaries, query_history, iteration)
            
        except Exception as e:
            logger.error(f"Error generating answer for query '{query}': {e}")
            return f"Error generating answer: {str(e)}"
    
    def _handle_research_tool_call(self, tool_call: Dict[str, Any], original_query: str, 
                                 queries_and_summaries: Dict[str, str], query_history: QueryHistory, iteration: int) -> str:
        logger.info(f"Result Synthesizer Tool call: {tool_call['name']} with args: {tool_call['args']} for query: {original_query}")
        if "WebSearchTool" in query_history.prev_tool_calls:
            query_processor = QueryProcessor(self.llm)
            tool_call_query = query_processor.expand_query(tool_call['args']['query'], query_history)
            logger.info(f"Expanded query to: {tool_call_query} for query: {original_query} using WebSearchTool")
        else:
            tool_call_query = tool_call['args']['query']
        search_result = self.search_executor(tool_call_query, original_query, iteration)
        queries_and_summaries[tool_call_query] = search_result
        query_history.prev_tool_calls.append(tool_call)
        return search_result
