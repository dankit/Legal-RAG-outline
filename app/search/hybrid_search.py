"""Hybrid search combining dense vector search (ChromaDB) with sparse keyword search (Elasticsearch)."""

import numpy as np
from ..core.vector_database import VectorDatabaseClient
from ..utils.elasticsearch_loader import ElasticSearchClient
from ..core.embedders import BGEEmbeddings
from ..core.vector_database import VectorDatabaseCollection
from ..tools.search_tools import HybridSearchTool


def hybrid_search(collection_name: str, query: str, bge_embeddings: BGEEmbeddings, 
                  vectordb_collection: VectorDatabaseCollection, top_k: int = 25):
    """Fuse results from Elasticsearch (BM25) and ChromaDB (vector) using reciprocal rank fusion."""
    elasticsearch = ElasticSearchClient()
    elasticsearch_results = elasticsearch.search_documents(collection_name, query, top_k)
    chroma_results = vectordb_collection.similarity_search(bge_embeddings.create_query_embeddings(query), top_k)

    return weighted_rrf(elasticsearch_results, chroma_results, top_k)


def linear_combination(elasticsearch_results, chroma_results, top_k):
    """Alternative fusion: min-max normalized score combination with alpha weighting."""
    bm25_docs = [
        {
            "id": hit["_id"],
            "text": hit["_source"]["text"],
            "bm25_score": hit["_score"]
        }
        for hit in elasticsearch_results["hits"]["hits"]
    ]
    bm25_scores = np.array([doc["bm25_score"] for doc in bm25_docs])
    if bm25_scores.max() - bm25_scores.min() != 0:
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
    else:
        bm25_scores_norm = bm25_scores

    chroma_scores = {
        chroma_results["ids"][0][i]: chroma_results["distances"][0][i]
        for i in range(len(chroma_results["ids"][0]))
    }
    emb_scores = np.array(list(chroma_scores.values()))
    # Convert cosine distance to similarity: sim = 1 - distance
    emb_scores = 1 - np.array(emb_scores)
    if emb_scores.max() - emb_scores.min() != 0:
        emb_scores_norm = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min())
    else:
        emb_scores_norm = emb_scores

    chroma_ids = list(chroma_scores.keys())
    alpha = 0.7  # Weight for dense vs sparse (higher = more dense)
    ids_to_document_mapping = {}
    scores = {}
    for doc in bm25_docs:
        scores[doc["id"]] = bm25_scores_norm[bm25_docs.index(doc)] * (1 - alpha)
        ids_to_document_mapping[doc["id"]] = doc["text"]
    for id in chroma_ids:
        scores[id] = scores.get(id, 0) + (emb_scores_norm[chroma_ids.index(id)] * alpha)
        ids_to_document_mapping[id] = chroma_results["documents"][0][chroma_ids.index(id)]

    hybrid_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    result_dic = {}
    for sorted_id in hybrid_results:
        result_dic[sorted_id[0]] = ids_to_document_mapping[sorted_id[0]]

    return result_dic
    

def weighted_rrf(elasticsearch_results, chroma_results, top_k):
    """Weighted Reciprocal Rank Fusion: rank-based score combination."""
    scores = {}
    elastic_search_weight = 0.495
    dense_search_weight = 1 - elastic_search_weight
    k = 60  # RRF constant: larger values make rank differences matter less
    ids_to_document_mapping = {}

    for i, result in enumerate(elasticsearch_results["hits"]["hits"]):
        scores[result["_id"]] = (1 / (k + i)) * elastic_search_weight
        ids_to_document_mapping[result["_id"]] = result["_source"]["text"]

    for i, id in enumerate(chroma_results["ids"][0]):
        scores[id] = scores.get(id, 0) + ((1 / (k + i)) * dense_search_weight)
        ids_to_document_mapping[id] = chroma_results["documents"][0][i]

    hybrid_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    result_dic = {}
    for sorted_id in hybrid_results:
        result_dic[sorted_id[0]] = ids_to_document_mapping[sorted_id[0]]
    return result_dic
