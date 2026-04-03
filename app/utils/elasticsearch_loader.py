import os
import json5
import logging
from elasticsearch import Elasticsearch, helpers
from .file_utils import move_file_to_processed_folder

logging.getLogger("elastic_transport.transport").setLevel(logging.ERROR)


class ElasticSearchClient:
    """Client for Elasticsearch used for sparse/keyword search (BM25)."""

    def __init__(self, host="http://localhost:9200"):
        self.es = Elasticsearch(host)

    def create_index(self, index_name, mappings=None):
        if mappings is None:
            mappings = {
                "properties": {
                    "text": {"type": "text", "analyzer": "standard"},
                    "filename": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "chunk_id": {"type": "integer"},
                    "collection": {"type": "keyword"},
                    "chunk_size": {"type": "integer"},
                    "has_headers": {"type": "boolean"},
                    "has_section": {"type": "boolean"},
                    "has_numbering": {"type": "boolean"},
                    "source": {"type": "keyword"}
                }
            }
        return self.es.indices.create(index=index_name, mappings=mappings)

    def delete_index(self, index_name):
        return self.es.indices.delete(index=index_name)

    def check_index_exists(self, index_name):
        return self.es.indices.exists(index=index_name)

    def get_index_info(self, index_name):
        return self.es.indices.get(index=index_name)

    def search_documents(self, index_name, query, k=10):
        return self.es.search(index=index_name, query={"match": {"text": query}}, size=k)
    
    def count_documents(self, index_name):
        """Get the total number of documents in an index."""
        return self.es.count(index=index_name)
    
    def get_random_documents(self, index_name, size=1):
        """Get random documents from an index."""
        return self.es.search(
            index=index_name,
            query={"function_score": {"query": {"match_all": {}}, "random_score": {}}},
            size=size
        )

    def store_naively_preprocessed_documents(self, index_name, list_of_documents):
        if not self.check_index_exists(index_name):
            self.create_index(index_name)

        actions = []
        for document in list_of_documents:
            doc_id = document["_id"]
            doc_body = {k: v for k, v in document.items() if k != "_id"}
            actions.append({"_index": index_name, "_id": doc_id, "_source": doc_body})
        helpers.bulk(self.es, actions)

    def store_llm_preprocessed_documents(self, index_name, chunks_file_location):
        for file_name in os.listdir(chunks_file_location):
            file_path = os.path.join(chunks_file_location, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                documents = []
                current_chunk_id = 0
                for line in f:
                    chunks_list = json5.loads(line)["chunks"]
                    if not chunks_list:
                        continue
                    for chunk in chunks_list:
                        documents.append({
                            "_index": index_name, 
                            "_id": f"{file_name}:{current_chunk_id}", 
                            "_source": {"text": chunk["text"]}
                        })
                        current_chunk_id += 1
                        if current_chunk_id % 500 == 0:
                            print(f"Indexed {current_chunk_id} chunks")
                helpers.bulk(self.es, documents)
            move_file_to_processed_folder(index_name, file_name)
