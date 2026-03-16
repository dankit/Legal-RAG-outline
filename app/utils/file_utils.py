from pathlib import Path
import shutil
import logging

logger = logging.getLogger(__name__)

data_directory = Path(__file__).parent.parent.parent / "Data"
processed_folder = "Processed"
chunks_file_suffix = "_chunks.jsonl"


def move_file_to_processed_folder(collection_name: str, file_name: str): 
    try:
        file_path = get_document_file_path(collection_name, file_name)
        dst_path = get_processed_file_path(collection_name)
        if not dst_path.exists():
            dst_path.mkdir(parents=True)
        shutil.move(file_path, dst_path)
    except Exception as e:
        logger.error(f"Error moving file {file_name} to processed folder: {e}")


def get_document_file_path(collection_name: str, file_name: str) -> Path:
    return data_directory / collection_name / file_name


def get_processed_file_path(collection_name: str) -> Path:
    return data_directory / collection_name / processed_folder


def get_json_chunks_file_path(collection_name: str, file_name: str) -> Path:
    return data_directory / collection_name / (file_name + chunks_file_suffix)
