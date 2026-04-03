from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from ..utils import file_utils
from .vector_database import VectorDatabaseClient
from ..utils.elasticsearch_loader import ElasticSearchClient
from .embedders import BGEEmbeddings
from langchain_core.documents import Document
import re
import fitz  # PyMuPDF
import hashlib
import logging
from typing import Generator, List, Tuple, Dict, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("elastic_transport.transport").setLevel(logging.ERROR)


def _get_content_hash(text: str) -> str:
    """Generate hash for content deduplication."""
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _is_valid_chunk(content: str, min_size: int = 50, min_words: int = 5) -> bool:
    """Validate chunk quality before storage."""
    if not content or not content.strip():
        return False
    
    content = content.strip()
    if len(content) < min_size:
        return False
    
    citation_patterns = [
        r'^\[ARC \d+[A-Z]?,\s*IAB',
        r'^Ch \d+, p\.\d+\s+IAC',
        r'^\[Filed.*effective.*\]$',
        r'^IAC \d+/\d+/\d+.*\[ARC'
    ]
    for pattern in citation_patterns:
        if re.search(pattern, content):
            return False
    
    words = re.findall(r'\b\w+\b', content)
    if len(words) < min_words:
        return False
    
    alpha_ratio = sum(c.isalnum() or c.isspace() for c in content) / len(content)
    if alpha_ratio < 0.5:
        return False
    
    return True


def _clean_metadata_value(value: str, max_length: int = 100) -> str:
    """Extract clean section identifier from metadata, limiting to title only."""
    if not value:
        return value
    
    value = re.sub(r'#+', '', value).strip()
    
    section_pattern = r'^(\d+[—\-\.]*\d*\.\d+.*?[A-Za-z][A-Za-z\s]*\.?)(?=\s+[A-Z][a-z]+|$)'
    match = re.match(section_pattern, value)
    if match:
        return match.group(1).strip()
    
    descriptive_phrases = [' As used in ', ' The following ', ' This rule ', ' For purposes of ']
    for phrase in descriptive_phrases:
        pos = value.find(phrase)
        if pos != -1 and pos > 20:
            return value[:pos].strip()
    
    first_sentence_end = re.search(r'\.(?=\s+[A-Z][a-z]+)', value)
    if first_sentence_end and first_sentence_end.start() < max_length:
        return value[:first_sentence_end.start() + 1].strip()
    
    if len(value) > max_length:
        last_space = value[:max_length].rfind(' ')
        if last_space > max_length // 2:
            return value[:last_space].strip()
        return value[:max_length].strip()
    
    return value


def _detect_font_thresholds(doc: fitz.Document, sample_pages: int = 5) -> Dict[str, float]:
    """Dynamically detect font size thresholds from document."""
    font_sizes = []
    pages_to_sample = min(sample_pages, len(doc))
    
    for i in range(0, len(doc), max(1, len(doc) // pages_to_sample)):
        try:
            page = doc.load_page(i)
            for block in page.get_text("dict")["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
        except Exception:
            continue
    
    if not font_sizes:
        return {"large": 12.0, "medium": 10.0}
    
    font_sizes.sort()
    return {
        "large": font_sizes[int(len(font_sizes) * 0.85)],
        "medium": font_sizes[int(len(font_sizes) * 0.65)]
    }


class NaivePDFPreprocessor:
    """Rule-based PDF preprocessor using font analysis and markdown header splitting.
    
    Converts PDFs to markdown, splits on headers, validates chunks,
    deduplicates, and stores in both ChromaDB and Elasticsearch.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap, 
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
            length_function=len
        )
        self.vectordb = VectorDatabaseClient()
        self.elasticsearch = ElasticSearchClient()
        self.data_directory = file_utils.data_directory
        self.embeddings = BGEEmbeddings()
        self.batch_size = 16

    def store_in_elasticsearch(self, collection_name: str, chunks: List[Document]):
        """Store chunks in Elasticsearch with proper formatting."""
        if not chunks:
            return
        
        es_documents = []
        for chunk in chunks:
            try:
                metadata = {k: v for k, v in chunk.metadata.items() if v is not None and v != ""}
                doc = {
                    "_id": f"{metadata.get('filename', 'unknown')}:{metadata.get('chunk_id', 0)}",
                    "text": chunk.page_content,
                    **metadata
                }
                es_documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to format chunk for Elasticsearch: {e}")
                continue
        
        if es_documents:
            try:
                self.elasticsearch.store_naively_preprocessed_documents(collection_name, es_documents)
            except Exception as e:
                logger.error(f"Failed to store chunks in Elasticsearch: {e}")
    
    def store_in_vector_database(self, collection_name: str, chunks: List[Document]):
        """Store chunks in vector database with proper formatting."""
        if not chunks:
            return
            
        collection = self.vectordb.get_collection(collection_name)
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            
            try:
                chunk_texts = [chunk.page_content for chunk in batch_chunks]
                chunk_embeddings = self.embeddings.create_document_embeddings(chunk_texts)
                
                metadatas = []
                for chunk in batch_chunks:
                    metadata = {k: v for k, v in chunk.metadata.items() if v is not None and v != ""}
                    metadata['chunkId'] = metadata.pop('chunk_id', 0)
                    
                    if 'filename' not in metadata:
                        metadata['filename'] = 'unknown'
                    if 'page' not in metadata:
                        metadata['page'] = 0
                    
                    metadatas.append(metadata)
                
                collection.store_chunk_embeddings(chunk_texts, chunk_embeddings, metadatas)
                
            except Exception as e:
                logger.error(f"Failed to store batch in vector database: {e}")
                continue
    
    def run_enhanced_preprocessor(self):
        """Enhanced preprocessing with validation, deduplication, and streaming."""
        header_keys = {"Chapter", "Section", "Subsection", "Clause", "Item", "Subitem"}
        
        for directory in self.data_directory.glob("*"):
            if not directory.is_dir():
                continue
                
            pdf_files = list(directory.glob("*.pdf"))
            if not pdf_files:
                logger.info(f"Skipping empty directory: {directory.name}")
                continue
                
            collection_name = directory.name.lower()
            
            for file in pdf_files:
                logger.info(f"Processing {file.name}")
                try:
                    chunk_id_counter = 0
                    filename = file.name
                    
                    for chunk_batch in process_pdf_with_streaming(str(file), self.chunk_size, self.chunk_overlap):
                        valid_chunks = [c for c in chunk_batch if _is_valid_chunk(c.page_content)]
                        
                        for chunk in valid_chunks:
                            chunk.metadata.update({
                                "chunk_id": chunk_id_counter,
                                "filename": filename,
                                "collection": collection_name,
                                "chunk_size": len(chunk.page_content),
                                "has_headers": bool(header_keys & chunk.metadata.keys()),
                                "has_section": bool(re.search(r'Sec\.|Section|Art\.|Article', chunk.page_content)),
                                "has_numbering": bool(re.search(r'\(\d+\)|\(\w+\)', chunk.page_content)),
                                "source": str(file)
                            })
                            chunk_id_counter += 1
                        
                        self.store_in_elasticsearch(collection_name, valid_chunks)
                        self.store_in_vector_database(collection_name, valid_chunks)
                    
                    file_utils.move_file_to_processed_folder(collection_name, file.name)
                except Exception as e:
                    logger.error(f"Failed to process {file.name}: {e}")
            logger.info(f"Processed {directory.name} successfully")


def _classify_heading(line: str, font_size: float = 0, is_bold: bool = False, 
                      font_thresholds: Dict[str, float] = None) -> str:
    """Classify line and return appropriate markdown heading level or empty string."""
    if font_thresholds is None:
        font_thresholds = {"large": 12.0, "medium": 10.0}
    
    line_clean = line.strip()
    
    patterns = [
        (r'^(CHAPTER|PART|TITLE)\s+[\dIVXA-Z]', '#'),
        (r'^\d{3,4}[—\-]\d+\.\d+.*\)', '##'),
        (r'^(Sec\.|Section|Art\.|Article)\s+[\d]+[\.\-][\d]+', '##'),
        (r'^\d+\.\d+\.\d+\s+[A-Z]', '###'),
        (r'^\([a-z]\)\s+[A-Z]', '####'),
    ]
    
    if line_clean.isupper() and 10 < len(line_clean) < 80 and not line_clean.count(' ') > 10:
        return '#'
    
    for pattern, level in patterns:
        if re.match(pattern, line_clean):
            return level
    
    if is_bold and font_size >= font_thresholds["large"] and len(line_clean) < 100 and line_clean.count(' ') < 8:
        return '##'
    
    return ''


def format_pdf_page(page: fitz.Page, font_thresholds: Dict[str, float] = None) -> str:
    """Extract and format PDF page with markdown headings based on font properties."""
    if font_thresholds is None:
        font_thresholds = {"large": 12.0, "medium": 10.0}
    
    text_blocks = []
    try:
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_y = line["bbox"][1]
                line_text = ""
                max_font_size = 0
                has_bold = False
                
                for span in line["spans"]:
                    text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\{\}\"\'\/§©®™°]', '', span["text"])
                    max_font_size = max(max_font_size, span.get("size", 0))
                    has_bold = has_bold or "bold" in span.get("font", "").lower() or span.get("flags", 0) & 16
                    line_text += text
                
                if line_text.strip():
                    heading = _classify_heading(line_text.strip(), max_font_size, has_bold, font_thresholds)
                    formatted = f"{heading} {line_text}" if heading else line_text
                    text_blocks.append((line_y, formatted))
    except Exception as e:
        logger.warning(f"Error formatting page: {e}")
        return ""
    
    text_blocks.sort(key=lambda x: x[0])
    return "\n".join(block[1] for block in text_blocks)


def convert_pdf_to_markdown(text: str) -> str:
    """Convert PDF text to markdown with proper heading hierarchy."""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text).strip()
    
    markdown_lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        if not line.startswith('#'):
            heading = _classify_heading(line)
            line = f"{heading} {line}" if heading else line
        
        if '##' in line and not line.startswith('#'):
            parts = line.split('##')
            markdown_lines.append(parts[0].strip()) if parts[0].strip() else None
            markdown_lines.extend(f"## {p.strip()}" for p in parts[1:] if p.strip())
        else:
            markdown_lines.append(line)
    
    return '\n'.join(filter(None, markdown_lines))


def create_enhanced_chunks(markdown_content: str, page_number: int,
                          min_chunk_size: int = 150, seen_hashes: Set[str] = None, 
                          chunk_size: int = 500, chunk_overlap: int = 100) -> Tuple[List[Document], Set[str]]:
    """Create chunks with validation and deduplication."""
    if seen_hashes is None:
        seen_hashes = set()
    
    headers_to_split_on = [
        ("#", "Chapter"),
        ("##", "Section"),
        ("###", "Subsection"),
        ("####", "Clause"),
        ("#####", "Item"),
        ("######", "Subitem")
    ]
    
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    
    try:
        chunks = header_splitter.split_text(markdown_content)
    except Exception as e:
        logger.warning(f"Header splitting failed on page {page_number}: {e}")
        return [], seen_hashes
    
    enhanced_chunks = []
    for i, chunk in enumerate(chunks):
        content = re.sub(r'\[Release Point [^\]]+\]', '', chunk.page_content).strip()
        content = re.sub(r'#+', '', content).strip()
        
        for key, value in chunk.metadata.items():
            cleaned_val = _clean_metadata_value(value)
            if len(cleaned_val) > 80 and not re.match(r'^(CHAPTER|PART|Sec\.|Section|\d{3,4}[—\-])', cleaned_val):
                content = f"{cleaned_val} {content}" if content else cleaned_val
                chunk.metadata[key] = ""
        
        if i < len(chunks) - 1 and len(content) < min_chunk_size:
            chunks[i + 1].page_content = f"{content}\n{chunks[i + 1].page_content}"
            for key, value in chunk.metadata.items():
                if value and (key not in chunks[i + 1].metadata or not chunks[i + 1].metadata[key]):
                    chunks[i + 1].metadata[key] = value
        else:
            if not _is_valid_chunk(content):
                logger.debug(f"Skipping invalid chunk on page {page_number}")
                continue
            
            content_hash = _get_content_hash(content)
            if content_hash in seen_hashes:
                logger.debug(f"Skipping duplicate chunk on page {page_number}")
                continue
            seen_hashes.add(content_hash)
            
            metadata = {}
            for k, v in chunk.metadata.items():
                if v is None or v == "":
                    continue
                if isinstance(v, str) and len(v) > 150:
                    continue
                metadata[k] = v
            metadata["page"] = page_number
            enhanced_chunks.append(Document(page_content=content, metadata=metadata))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
        length_function=len
    )
    
    split_chunks = []
    for doc in enhanced_chunks:
        try:
            splits = text_splitter.split_documents([doc])
            for split in splits:
                if _is_valid_chunk(split.page_content, min_size=min_chunk_size, min_words=15):
                    split_chunks.append(split)
                elif len(split_chunks) > 0:
                    split_chunks[-1].page_content += f" {split.page_content}"
        except Exception as e:
            logger.warning(f"Text splitting failed on page {page_number}: {e}")
            continue
    
    return split_chunks, seen_hashes


def process_pdf_with_streaming(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> Generator[List[Document], None, None]:
    """Memory-efficient PDF processing with streaming, progress tracking, and validation."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        return
    
    total_pages = len(doc)
    logger.info(f"Processing {pdf_path} ({total_pages} pages)")
    
    font_thresholds = _detect_font_thresholds(doc)
    logger.info(f"Detected font thresholds: {font_thresholds}")
    
    seen_hashes = set()
    total_chunks = 0
    
    for page_num in range(total_pages):
        try:
            page = doc.load_page(page_num)
            formatted_text = format_pdf_page(page, font_thresholds)
            
            if not formatted_text:
                logger.warning(f"Empty page {page_num + 1}/{total_pages}")
                continue
            
            markdown_content = convert_pdf_to_markdown(formatted_text)
            chunks, seen_hashes = create_enhanced_chunks(
                markdown_content, page_num, min_chunk_size=150, 
                seen_hashes=seen_hashes, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            
            total_chunks += len(chunks)
            
            if chunks:
                yield chunks
            
            if (page_num + 1) % 10 == 0:
                logger.info(f"Progress: {page_num + 1}/{total_pages} pages ({(page_num + 1) / total_pages * 100:.1f}%) - {total_chunks} chunks")
        
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            continue
    
    doc.close()
    
    stats = {
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "unique_chunks": len(seen_hashes),
        "avg_chunks_per_page": total_chunks / total_pages if total_pages > 0 else 0
    }
    
    logger.info(f"Completed processing {pdf_path}: {stats}")


if __name__ == "__main__":
    preprocessor = NaivePDFPreprocessor()
    preprocessor.run_enhanced_preprocessor()
