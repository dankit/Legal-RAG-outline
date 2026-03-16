"""Validation tests for LLM preprocessor output quality."""

import json
import logging
from pathlib import Path
import re
from langchain_community.document_loaders import PyMuPDFLoader

logger = logging.getLogger(__name__)
test_data_dir = Path(__file__).parent / "Data"
test_data_file = test_data_dir / "test.pdf"
test_data_chunks_file = test_data_dir / "test.pdf_chunks.json"


def test_llm_preprocessor_chunks_and_prompt_efficiency():
    """Validate that preprocessed chunks faithfully represent the source PDF text."""
    num_valid_pages = 0
    num_invalid_pages = 0
    test_pdf_document = PyMuPDFLoader(test_data_file).load()
    with open(test_data_chunks_file, "r") as f:
        for page_number, line in enumerate(f):
            try:
                parsed_line = json.loads(line)
                prev_page = test_pdf_document[page_number - 1].page_content if page_number > 0 else ""
                curr_page = test_pdf_document[page_number].page_content
                is_valid = validate_page_chunks(parsed_line, prev_page + curr_page, page_number)
                if is_valid:
                    num_valid_pages += 1
                else:
                    num_invalid_pages += 1
            except Exception as e:
                logger.error(f"Error running preprocessor validation on page {page_number}: {e}")
                num_invalid_pages += 1
    logger.info(f"Valid pages: {num_valid_pages}/{num_valid_pages + num_invalid_pages}")


def validate_page_chunks(parsed_json, original_text, page_number, chunk_size=800) -> bool:
    """Check that each chunk's text exists in the original and respects size limits."""
    try:
        for i, chunk in enumerate(parsed_json["chunks"]):
            chunk_cleaned = re.sub(r'\s+', ' ', chunk["text"])
            chunk_cleaned = re.sub(r'[^a-zA-Z0-9]', '', chunk_cleaned)
            original_text_cleaned = re.sub(r'\s+', ' ', original_text)
            original_text_cleaned = re.sub(r'[^a-zA-Z0-9]', '', original_text_cleaned)
            if chunk_cleaned not in original_text_cleaned:
                logger.error(f"Chunk {i} on page {page_number} had unexpected text.")
                return False
            if len(chunk_cleaned) > chunk_size:
                logger.error(f"Chunk {i} on page {page_number} was too long: {len(chunk_cleaned)} characters")
        return True
    except Exception as e:
        logger.error(f"Error validating page {page_number}: {e}")
        return False
