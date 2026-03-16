import json
import time
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ..core.vector_database import VectorDatabaseClient, VectorDatabaseCollection
from ..core.embedders import BGEEmbeddings
from ..core.reranker import BGEReranker
from ..search.hybrid_search import hybrid_search


@dataclass
class RetrievalResult:
    """Container for retrieval results with metadata."""
    chunk_ids: List[str]
    documents: Dict[str, str]
    retrieval_time: float
    precision: float
    recall: float
    true_positives: int


@dataclass
class RerankerResult:
    """Container for reranker results with metadata."""
    reranked_chunk_ids: List[str]
    reranker_time: float
    precision: float
    recall: float
    f1: float
    mrr: float
    map: float
    true_positives: int


class RetrievalEvaluator:
    """Handles document retrieval and caching for evaluation."""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.bge_embeddings = BGEEmbeddings()
        self.vector_db = VectorDatabaseClient().get_collection(collection_name)
        self._retrieval_cache = {}
        
    def retrieve_documents(self, question: str, k: int, use_cache: bool = True) -> RetrievalResult:
        """Retrieve documents for a question, with optional caching."""
        cache_key = f"{question}_{k}"
        
        if use_cache and cache_key in self._retrieval_cache:
            return self._retrieval_cache[cache_key]
        
        start_time = time.time()
        retrieval_results = query_with_hybrid(self.collection_name, question, self.bge_embeddings, self.vector_db, k)
        retrieval_time = time.time() - start_time
        
        chunk_ids = list(retrieval_results.keys())
        
        result = RetrievalResult(
            chunk_ids=chunk_ids,
            documents=retrieval_results,
            retrieval_time=retrieval_time,
            precision=0.0,
            recall=0.0,
            true_positives=0
        )
        
        if use_cache:
            self._retrieval_cache[cache_key] = result
            
        return result
    
    def clear_cache(self):
        self._retrieval_cache.clear()


class RerankerEvaluator:
    """Handles reranking evaluation."""
    
    def __init__(self):
        self.bge_reranker = BGEReranker()
        self.hyphen_linebreak_fix = re.compile(r'(\w)-\n(\w)')
        self.newline_collapse = re.compile(r'\s*\n\s*')
    
    def rerank_documents(self, question: str, retrieval_result: RetrievalResult, reranker_k: int) -> RerankerResult:
        """Rerank retrieved documents."""
        cleaned_documents = [
            self.newline_collapse.sub(' ', self.hyphen_linebreak_fix.sub(r'\1\2', chunk)).strip() 
            for chunk in retrieval_result.documents.values()
        ]
        
        start_time = time.time()
        reranked_documents = self.bge_reranker.rerank_documents(question, cleaned_documents, reranker_k)
        reranker_time = time.time() - start_time
        
        reranked_chunk_ids = [retrieval_result.chunk_ids[doc['corpus_id']] for doc in reranked_documents]
        
        return RerankerResult(
            reranked_chunk_ids=reranked_chunk_ids,
            reranker_time=reranker_time,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            mrr=0.0,
            map=0.0,
            true_positives=0
        )


class BenchmarkManager:
    """Orchestrates evaluation runs and enables parameter sweeping."""
    
    def __init__(self, collection_name: str, test_data_path: str = "../Data/testing/questions_with_labels.jsonl"):
        self.collection_name = collection_name
        self.test_data_path = test_data_path
        self.retrieval_evaluator = RetrievalEvaluator(collection_name)
        self.reranker_evaluator = RerankerEvaluator()
        
    def load_test_queries(self) -> List[Dict]:
        """Load test queries from JSONL file."""
        queries = []
        with open(self.test_data_path, "r") as file:
            for line in file:
                query_data = json.loads(line)
                for query in query_data["chunks"]:
                    question = query["Question"]
                    original_file = query["chunk_file"]
                    labels = [original_file + ":" + str(label) for label in query["Labels"]]
                    queries.append({
                        "question": question,
                        "labels": labels,
                        "ground_truth_length": len(labels)
                    })
        return queries
    
    def evaluate_single_query(self, query_data: Dict, k: int, reranker_k: int, 
                            use_retrieval_cache: bool = True) -> Tuple[RetrievalResult, RerankerResult]:
        """Evaluate a single query with given parameters."""
        question = query_data["question"]
        labels = query_data["labels"]
        ground_truth_length = query_data["ground_truth_length"]
        
        retrieval_result = self.retrieval_evaluator.retrieve_documents(question, k, use_retrieval_cache)
        
        retrieval_true_positives = sum(1 for label in labels if label in retrieval_result.chunk_ids)
        retrieval_result.true_positives = retrieval_true_positives
        retrieval_result.precision = retrieval_true_positives / len(retrieval_result.chunk_ids) if len(retrieval_result.chunk_ids) > 0 else 0.0
        retrieval_result.recall = retrieval_true_positives / ground_truth_length if ground_truth_length > 0 else 0.0
        
        reranker_result = self.reranker_evaluator.rerank_documents(question, retrieval_result, reranker_k)
        
        # Calculate reranker metrics (precision, recall, F1, MRR, MAP)
        relevant_found = 0
        average_precision = 0.0
        for i, chunk_id in enumerate(reranker_result.reranked_chunk_ids):
            if chunk_id in labels:
                relevant_found += 1
                average_precision += relevant_found / (i + 1)
        
        reranker_result.true_positives = relevant_found
        reranker_result.precision = relevant_found / len(reranker_result.reranked_chunk_ids) if len(reranker_result.reranked_chunk_ids) > 0 else 0.0
        reranker_result.recall = relevant_found / ground_truth_length if ground_truth_length > 0 else 0.0
        reranker_result.f1 = 2 * ((reranker_result.precision * reranker_result.recall) / (reranker_result.precision + reranker_result.recall)) if reranker_result.precision + reranker_result.recall > 0 else 0.0
        reranker_result.map = average_precision / ground_truth_length if ground_truth_length > 0 else 0.0
        
        first_relevant_rank = next((i for i, chunk_id in enumerate(reranker_result.reranked_chunk_ids) if chunk_id in labels), None)
        reranker_result.mrr = 1 / (first_relevant_rank + 1) if first_relevant_rank is not None else 0.0
        
        return retrieval_result, reranker_result
    
    def run_evaluation(self, k: int, reranker_k: int, use_retrieval_cache: bool = True) -> Dict:
        """Run evaluation with given parameters."""
        queries = self.load_test_queries()
        
        retrieval_results = []
        reranker_results = []
        total_labels = 0
        total_retrieval_true_positives = 0
        total_reranker_true_positives = 0
        
        for i, query_data in enumerate(queries):
            total_labels += query_data["ground_truth_length"]
            
            retrieval_result, reranker_result = self.evaluate_single_query(
                query_data, k, reranker_k, use_retrieval_cache
            )
            
            retrieval_results.append(retrieval_result)
            reranker_results.append(reranker_result)
            total_retrieval_true_positives += retrieval_result.true_positives
            total_reranker_true_positives += reranker_result.true_positives
            
            if (i + 1) % 100 == 0:
                print(f"Queries processed: {i + 1}")
        
        total_queries = len(queries)
        total_retrieval_time = sum(r.retrieval_time for r in retrieval_results)
        total_reranker_time = sum(r.reranker_time for r in reranker_results)
        
        avg_retrieval_precision = sum(r.precision for r in retrieval_results) / total_queries
        avg_retrieval_recall = sum(r.recall for r in retrieval_results) / total_queries
        micro_retrieval_recall = total_retrieval_true_positives / total_labels
        
        avg_reranker_precision = sum(r.precision for r in reranker_results) / total_queries
        avg_reranker_recall = sum(r.recall for r in reranker_results) / total_queries
        avg_reranker_f1 = sum(r.f1 for r in reranker_results) / total_queries
        avg_reranker_mrr = sum(r.mrr for r in reranker_results) / total_queries
        avg_reranker_map = sum(r.map for r in reranker_results) / total_queries
        micro_reranker_recall = total_reranker_true_positives / total_labels
        
        return {
            "collection_name": self.collection_name,
            "k": k,
            "reranker_k": reranker_k,
            "total_queries": total_queries,
            "total_labels": total_labels,
            "total_time": total_retrieval_time + total_reranker_time,
            "avg_time_per_query": (total_retrieval_time + total_reranker_time) / total_queries,
            "retrieval": {
                "total_time": total_retrieval_time,
                "avg_time_per_query": total_retrieval_time / total_queries,
                "avg_precision": avg_retrieval_precision,
                "avg_recall": avg_retrieval_recall,
                "micro_recall": micro_retrieval_recall,
                "true_positives": total_retrieval_true_positives
            },
            "reranker": {
                "total_time": total_reranker_time,
                "avg_time_per_query": total_reranker_time / total_queries,
                "avg_precision": avg_reranker_precision,
                "avg_recall": avg_reranker_recall,
                "avg_f1": avg_reranker_f1,
                "avg_mrr": avg_reranker_mrr,
                "avg_map": avg_reranker_map,
                "micro_recall": micro_reranker_recall,
                "true_positives": total_reranker_true_positives
            }
        }
    
    def run_parameter_sweep(self, k_values: List[int], reranker_k_values: List[int], 
                           use_retrieval_cache: bool = True) -> List[Dict]:
        """Run evaluation across multiple parameter combinations."""
        results = []
        
        for k in k_values:
            print(f"\n=== Retrieval K={k} ===")
            self.retrieval_evaluator.clear_cache()
            
            for reranker_k in reranker_k_values:
                print(f"Testing reranker_k={reranker_k}")
                result = self.run_evaluation(k, reranker_k, use_retrieval_cache)
                results.append(result)
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted evaluation results."""
        print(f"\n{results['collection_name']}, total queries: {results['total_queries']}")
        print(f"Total time taken: {results['total_time']:.2f} seconds, avg time per query: {results['avg_time_per_query']:.2f} seconds")
        
        ret = results['retrieval']
        print(f"===retrieval@{results['k']}=== total time: {ret['total_time']:.2f}s, avg: {ret['avg_time_per_query']:.2f}s")
        print(f"mean_precision: {ret['avg_precision']:.4f}, mean_recall: {ret['avg_recall']:.4f} | TP: {ret['true_positives']}/{results['total_labels']}. Micro recall: {ret['micro_recall']:.4f}")
        
        rer = results['reranker']
        print(f"===reranker@{results['reranker_k']}=== total time: {rer['total_time']:.2f}s, avg: {rer['avg_time_per_query']:.2f}s")
        print(f"mean_precision: {rer['avg_precision']:.4f}, mean_recall: {rer['avg_recall']:.4f} | TP: {rer['true_positives']}/{results['total_labels']}. Micro recall: {rer['micro_recall']:.4f}")
        print(f"reranker_f1: {rer['avg_f1']:.4f}")
        print(f"mean_reciprocal_rank: {rer['avg_mrr']:.4f}")
        print(f"mean_average_precision: {rer['avg_map']:.4f}")


def query_with_hybrid(collection_name: str, query: str, bge_embeddings: BGEEmbeddings, 
                      vectordb_collection: VectorDatabaseCollection, topK: int):
    """Run hybrid search combining vector and keyword retrieval."""
    hybrid_results = hybrid_search(collection_name, query, bge_embeddings, vectordb_collection, topK)
    return hybrid_results


if __name__ == "__main__":
    print("=== Single Evaluation ===")
    benchmark_manager = BenchmarkManager("Iowa_Law_v6")
    results = benchmark_manager.run_evaluation(k=10, reranker_k=5)
    benchmark_manager.print_results(results)
    
    print("\n=== Parameter Sweep ===")
    k_values = [5, 10, 15]
    reranker_k_values = [3, 5, 7]
    sweep_results = benchmark_manager.run_parameter_sweep(k_values, reranker_k_values)
    for result in sweep_results:
        benchmark_manager.print_results(result)
