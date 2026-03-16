import logging
import json5
from ..utils import file_utils
from .vector_database import VectorDatabaseClient
from .embedders import BGEEmbeddings

logger = logging.getLogger(__name__)


class PDFIndexer:
    """Indexes LLM-preprocessed PDF chunks into the vector database."""

    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.vectordb = VectorDatabaseClient()
        self.embeddings = BGEEmbeddings()
    
    def process_file_chunks(self, file_name):
        try:
            collection = self.vectordb.get_collection(self.collection_name)
            chunks_file_location = file_utils.get_document_file_path(self.collection_name, file_name)
            starting_chunk_id = collection.get_last_processed_chunk_id(file_name) + 1
            logger.info(f"Processing file {file_name}, starting_chunk_id: {starting_chunk_id}")

            with open(chunks_file_location, "r", encoding="utf-8") as f:
                    curr_chunk_id = 0
                    for page_number, line in enumerate(f):
                        try:
                            chunks_list = json5.loads(line)["chunks"]

                            if not chunks_list:
                                continue

                            if curr_chunk_id < starting_chunk_id:
                                if curr_chunk_id + len(chunks_list) <= starting_chunk_id:
                                    curr_chunk_id += len(chunks_list)
                                    continue
                                else:
                                    chunks_list = chunks_list[starting_chunk_id - curr_chunk_id:]
                                    curr_chunk_id = starting_chunk_id

                            chunk_texts = []
                            chunk_metadata = []                            
                            for chunk in chunks_list:
                                chunk_texts.append(chunk["text"])
                                chunk_metadata.append(self.add_metadata_to_chunk(chunk["metadata"], page_number, curr_chunk_id, file_name))
                                curr_chunk_id += 1

                            chunk_embeddings = self.embeddings.create_document_embeddings(chunk_texts) 
                            collection.store_chunk_embeddings(chunk_texts, chunk_embeddings, chunk_metadata)
                        except Exception as e:
                            logger.error(f"File {file_name} indexing failed on line {page_number}: {e}")
                            return
                            
            file_utils.move_file_to_processed_folder(self.collection_name, file_name)
            logger.info(f"File {file_name} processed successfully")
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
    
    def add_metadata_to_chunk(self, chunk_metadata: dict, page_number: int, curr_chunk_id: int, file_name: str):
        chunk_metadata["filename"] = file_name
        chunk_metadata["chunkId"] = curr_chunk_id
        chunk_metadata["page"] = page_number + 1
        return self.strip_none_values_from_metadata(chunk_metadata)
    
    def strip_none_values_from_metadata(self, chunk_metadata: dict):
        """Remove None values from metadata (required by ChromaDB)."""
        return {k: v for k, v in chunk_metadata.items() if v is not None}
