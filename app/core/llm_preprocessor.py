import json5
import os
import re
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_vertexai import ChatVertexAI
from ..utils import file_utils
from ..prompts.preprocessing_prompts import get_pdf_chunking_prompt
from pathlib import Path

logger = logging.getLogger(__name__)


class LLM_PDF_Preprocessor:
    """LLM-based PDF preprocessor that uses Gemini to semantically chunk PDF pages."""

    def __init__(self, collection_name: str, chunk_size: int = 800):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.llm = ChatVertexAI(model="gemini-2.0-flash-001", temperature=0)
        self.default_chunk = '{"chunks": [{"text": "", "metadata": {}}], "text_carry_over": ""}'

    def chunk_document(self, input_text: str, carry_over: str = ""):
        prompt_template = get_pdf_chunking_prompt(self.chunk_size)
        prompt = prompt_template.format(carry_over_text=carry_over, input_text=input_text)
        response = self.llm.invoke(prompt)
        response = re.sub(r"```(?:json)?\n?|\n?```", "", str(response.content))
        return response

    def run(self, document_name: str):
        document_path = file_utils.get_document_file_path(self.collection_name, document_name)
        documents = PyMuPDFLoader(str(document_path)).load()
        document_chunks_path = file_utils.get_json_chunks_file_path(self.collection_name, document_name)

        last_processed_page, carry_over_text = self.get_last_processed_page(document_chunks_path)
        logger.info(f"Processing {document_name}. Last processed page: {last_processed_page}")

        for page_number in range(last_processed_page + 1, len(documents)):
            chunk_to_save = self.default_chunk
            page_content = documents[page_number].page_content

            if page_content:
                chunk_to_save = self.chunk_document(page_content, carry_over_text)
            elif carry_over_text:
                chunk_to_save = self.chunk_document(carry_over_text, "")

            chunk_to_save, carry_over_text = self.add_document_metadata(chunk_to_save, page_number)
            self.save_chunk(document_chunks_path, chunk_to_save)
        file_utils.move_file_to_processed_folder(self.collection_name, document_name)

    def add_document_metadata(self, chunk: str, page_number: int):
        try:
            json_chunk = json5.loads(chunk)
            for inner_chunk in json_chunk["chunks"]:
                inner_chunk["metadata"]["page"] = page_number
            carry_over_text = json_chunk.get("text_carry_over", "")
            return json_chunk, carry_over_text
        except Exception as e:
            logger.error(e, extra={"page_number": page_number, "error_type": "json_parse_error", "text": chunk})
            return self.add_document_metadata(self.default_chunk, page_number)

    def save_chunk(self, chunks_location: Path, chunk: str):
        with open(chunks_location, "a") as f:
            f.write(json5.dumps(chunk))
            f.write("\n")

    def get_last_processed_page(self, chunks_location: Path):
        try:
            if os.path.exists(chunks_location):
                with open(chunks_location, "r") as f:
                    loaded_json = json5.loads(f.readlines()[-1])
                    return int(loaded_json["chunks"][-1]["metadata"]["page"]), loaded_json["text_carry_over"]
            return -1, ""
        except Exception as e:
            logger.error(f"Error getting last processed page: {e}")


def reprocess_page(collection_name: str, document_name: str, page: int):
    """Re-process a single page from a document (useful for fixing bad chunks)."""
    preprocessor = LLM_PDF_Preprocessor(collection_name)
    loaded_document = PyMuPDFLoader(str(file_utils.get_document_file_path(collection_name, document_name))).load()
    prev_json_chunks = file_utils.get_json_chunks_file_path(collection_name, document_name)
    prev_carry_over_text = ""
    with open(prev_json_chunks, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == page:
                json_chunk = json5.loads(line)
                prev_carry_over_text = json_chunk.get("text_carry_over", "")
                print(preprocessor.chunk_document(loaded_document[page].page_content, prev_carry_over_text))
                break
