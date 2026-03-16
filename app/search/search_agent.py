"""SearchAgent: the main orchestrator for query processing, retrieval, and synthesis."""

import logging
from typing import Dict, Any, Optional
from langchain_google_vertexai import ChatVertexAI
from ..core.vector_database import VectorDatabaseClient
from .hybrid_search import hybrid_search
from ..core.embedders import BGEEmbeddings
from ..core.reranker import BGEReranker
from ..webscrapers.web_search import web_search

from ..config.search_config import SearchConfig, QueryHistory
from .query_processor import QueryProcessor
from .search_method_selector import SearchMethodSelector
from .document_summarizer import DocumentSummarizer
from .result_synthesizer import ResultSynthesizer
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class SearchAgent:
    """Orchestrates the full search pipeline: query expansion, method selection,
    hybrid/web search, reranking, summarization, and result synthesis.
    
    Pipeline:
    1. Expand query into subqueries
    2. For each subquery: select search method -> retrieve -> rerank -> summarize
    3. Synthesize all subquery summaries into a final answer
    """

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.db = VectorDatabaseClient()
        self.llm = ChatVertexAI(model="gemini-2.5-flash", temperature=0, thinking_budget=1000)
        self.embedder = BGEEmbeddings()
        self.reranker = BGEReranker()
        self.query_processor = QueryProcessor(self.llm)
        self.method_selector = SearchMethodSelector(self.llm, self.db, self.config)
        self.document_summarizer = DocumentSummarizer(self.llm, self.config)
        self.result_synthesizer = ResultSynthesizer(self.llm, self.config, self._execute_search)
        self.query_history: Dict[str, QueryHistory] = {}
    
    def run(self, query: str, context: str = "") -> str:
        """Main entry point for the search agent."""
        try:
            logger.info(f"Running search agent with query: {query} and context: {context[:100]}...")            
            self._add_query_to_history(query)

            query_expansions = self.query_processor.expand_query(query, self.query_history[query], context)
            logger.info(f"query '{query}' expanded into {len(query_expansions)} subqueries: {query_expansions}")
            
            summaries_per_subquery = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self._execute_search, subquery, query, 0): subquery for subquery in query_expansions}
                for future in as_completed(futures):
                    subquery = futures[future]
                    try:
                        result = future.result()
                        summaries_per_subquery[subquery] = result
                    except Exception as e:
                        logger.error(f"Error executing search for subquery '{subquery}': {e}")
                        summaries_per_subquery[subquery] = f"Error: {str(e)}"
            
            answer = self.result_synthesizer.synthesize_answer(
                query, summaries_per_subquery, self.query_history[query])            
            
            return answer
            
        except Exception as e:
            logger.error(f"Error running search agent for query '{query}': {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    def _execute_search(self, query: str, parent_query: Optional[str] = None, iteration: int = 0) -> str:
        """Execute the appropriate search method for a single query."""
        self._add_query_to_history(query, parent_query)
        llm_tool_response = self.method_selector.select_search_method(query, self.query_history[query], iteration)
        logger.info(f"Tool call made to {llm_tool_response['name']} with args: {llm_tool_response['args']} for query: {query}")
            
        if llm_tool_response['name'] == "HybridSearchTool":
            return self._execute_hybrid_search(query, llm_tool_response)
        elif llm_tool_response['name'] == "WebSearchTool":
            return self._execute_web_search(llm_tool_response['args']['query'])
    
    def _execute_hybrid_search(self, query: str, tool_response: Dict[str, Any]) -> str:
        """Execute hybrid search with reranking."""
        try:
            collection = self.db.get_collection(tool_response['args']['collection_name'])
            hybrid_results = hybrid_search(
                tool_response['args']['collection_name'], query, 
                self.embedder, collection, self.config.hybrid_search_top_k
            )
            
            document_texts = []
            document_ids = []
            for doc_id, doc_text in hybrid_results.items():
                document_ids.append(doc_id)
                document_texts.append(doc_text)
            
            reranked_results = self.reranker.rerank_documents(query, document_texts, self.config.rerank_top_k)
            
            top_results_metadata = {}
            top_results_texts = {}
            for doc in reranked_results:
                chunk_id = document_ids[doc['corpus_id']]
                top_results_texts[chunk_id] = doc['text']
                chunk_metadata = collection.get_chunk_by_id(chunk_id)['metadatas'][0]
                top_results_metadata[chunk_id] = {
                    "page": chunk_metadata.get('page', 'unknown'),
                    "section": chunk_metadata.get('section', 'unknown'),
                    "filename": chunk_metadata.get('filename', 'unknown')
                }
            
            return self.document_summarizer.summarize_documents(
                query, collection, top_results_texts, top_results_metadata, self.query_history[query])

        except Exception as e:
            logger.error(f"Error in hybrid search for query '{query}': {type(e).__name__}: {e}", exc_info=True)
            return f"Error in hybrid search: {str(e)}"
    
    def _execute_web_search(self, query: str) -> str:
        """Execute web search."""
        try:
            web_results = web_search(query)
            return self.document_summarizer.summarize_web_results(
                query, web_results, self.query_history[query])
        except Exception as e:
            logger.error(f"Error in web search for query '{query}': {type(e).__name__}: {e}", exc_info=True)
            return f"Error in web search: {str(e)}"
    
    def _add_query_to_history(self, query: str, parent_query: Optional[str] = None) -> None:
        """Adds a query to the history with tracking for parent relationships."""
        if query not in self.query_history:
            self.query_history[query] = QueryHistory(parent_query)
        elif parent_query and not self.query_history[query].parent_query:
            self.query_history[query].parent_query = parent_query
    
    def _calculate_total_tokens(self) -> int:
        total_tokens = 0
        for query in self.query_history:
            total_tokens += self.query_history[query].total_token_usage
        return total_tokens
    
    def get_query_history(self, query: str) -> Optional[QueryHistory]:
        return self.query_history.get(query)
    
    def clear_history(self) -> None:
        self.query_history.clear()
        logger.info("Search agent history cleared")
