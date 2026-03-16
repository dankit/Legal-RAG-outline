from chromadb.config import Settings
import chromadb
import threading


class VectorDatabaseClient:
    """Thread-safe singleton client for ChromaDB with HNSW index configuration."""
    _instance = None
    _lock = threading.Lock()
    _chroma_client = None
    _persist_directory = None
    
    def __new__(cls, persist_directory="./chroma_db"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, persist_directory="./chroma_db"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return
            
            self.persist_directory = persist_directory
            if VectorDatabaseClient._chroma_client is None or VectorDatabaseClient._persist_directory != persist_directory:
                VectorDatabaseClient._chroma_client = chromadb.PersistentClient(
                    path=persist_directory, 
                    settings=Settings(anonymized_telemetry=False)
                )
                VectorDatabaseClient._persist_directory = persist_directory
            
            self.chroma_client = VectorDatabaseClient._chroma_client
            self.space = "ip"
            self.ef_construction = 256
            self.ef_search = 256
            self.max_neighbors = 32
            self.configuration = {
                "hnsw": {
                    "space": self.space,
                    "ef_construction": self.ef_construction,
                    "ef_search": self.ef_search,
                    "max_neighbors": self.max_neighbors
                }
            }
            self._initialized = True
    
    def get_collection(self, collection_name):
        return VectorDatabaseCollection(
            self.chroma_client.get_or_create_collection(
                collection_name, 
                metadata={"collection_name": collection_name}, 
                configuration=self.configuration
            )
        )
    
    def list_collections(self):
        return self.chroma_client.list_collections()
    
    def delete_collection(self, collection_name):
        return self.chroma_client.delete_collection(collection_name)


class VectorDatabaseCollection:
    """Wrapper around a ChromaDB collection with search and storage operations."""

    def __init__(self, collection):
        self.collection = collection

    def get_collection_info(self):
        return {
            "number_of_documents": self.collection.count(),
            "collection_metadata": self.collection.metadata
        }

    def similarity_search(self, query: list[float], k: int):
        """Vector search for relevant documents."""
        return self.collection.query(
            query_embeddings=query,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

    def get_chunk_neighbors(self, filename: str, chunkid: int, k: int):
        """Get neighboring chunks by chunk ID within the same file."""
        chunk_neighbors = self.collection.get(
            where={
                "$and": [
                    {"filename": filename},
                    {"chunkId": {"$gte": chunkid - k}},
                    {"chunkId": {"$lte": chunkid + k}},
                    {"chunkId": {"$ne": chunkid}}
                ]
            }
        )
        return chunk_neighbors
    
    def delete_chunk_by_id(self, id: str):
        self.collection.delete(ids=[id])
    
    def get_chunk_by_id(self, id: str):
        return self.collection.get(
            ids=[id],
            include=["metadatas", "documents"]
        )

    def get_last_processed_chunk_id(self, filename: str):
        return self.collection.metadata.get(f'{filename}', -1)

    def store_chunk_embeddings(self, chunks: list[str], chunk_embeddings: list[list[float]], metadatas: list[dict]):
        if not chunks or not chunk_embeddings or not metadatas:
            return
        
        if not (len(chunks) == len(chunk_embeddings) == len(metadatas)):
            raise ValueError(f"Mismatched lengths: chunks={len(chunks)}, embeddings={len(chunk_embeddings)}, metadatas={len(metadatas)}")
        
        for metadata in metadatas:
            if 'filename' not in metadata or 'chunkId' not in metadata:
                raise ValueError(f"Metadata missing required fields: {metadata}")
        
        try:
            self.collection.add(
                documents=chunks,
                embeddings=chunk_embeddings,
                metadatas=metadatas,
                ids=[f"{metadata['filename']}:{metadata['chunkId']}" for metadata in metadatas]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to collection: {e}")
        
        file_max_chunks = {}
        for metadata in metadatas:
            filename = metadata['filename']
            chunk_id = metadata['chunkId']
            if filename not in file_max_chunks or chunk_id > file_max_chunks[filename]:
                file_max_chunks[filename] = chunk_id
        
        updates = {}
        for filename, max_chunk_id in file_max_chunks.items():
            current_max = self.collection.metadata.get(filename, -1)
            if max_chunk_id > current_max:
                updates[filename] = max_chunk_id
        
        if updates:
            self.update_collection_metadata(updates)

    def update_collection_metadata(self, metadatas: dict):
        self.collection.metadata.update(metadatas)
        self.collection.modify(metadata=self.collection.metadata)
