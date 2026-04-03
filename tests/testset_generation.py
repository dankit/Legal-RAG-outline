"""Generate test questions from random chunks for Recall@K evaluation."""

import os
import random
import re
import json

from langchain_google_vertexai import ChatVertexAI
from app.core.vector_database import VectorDatabaseClient
import logging

logger = logging.getLogger(__name__)
vector_db = VectorDatabaseClient()


def get_random_chunk(collection_name: str):
    collection = vector_db.get_collection(collection_name)
    curr_files_in_db = collection.collection.metadata
    curr_files_in_db.pop('collection_name')
    random_file = random.choice(list(curr_files_in_db))
    max_chunk = curr_files_in_db.get(random_file) - 1
    random_chunk_id = random.randint(0, max_chunk)
    random_chunk = collection.get_chunk_by_id(f'{random_file}:{random_chunk_id}')
    return random_chunk


def create_question_from_chunk(random_chunk: dict):
    """Use LLM to generate test questions from a chunk and its neighbors."""
    collection = vector_db.get_collection("Iowa_Law_v6")
    chunk_neighbors = collection.get_chunk_neighbors(
        random_chunk["metadatas"][0]["filename"], 
        random_chunk["metadatas"][0]["chunkId"], k=1
    )

    chunk_neighbors_docs = [document for document in chunk_neighbors["documents"]]
    chunk_neighbors_docs.append(random_chunk["documents"][0])

    chunk_neighbors_metadata = [str(metadata["chunkId"]) for metadata in chunk_neighbors["metadatas"]]
    chunk_neighbors_metadata.append(str(random_chunk["metadatas"][0]["chunkId"]))

    prompt_string = str(["chunkId: " + y + "\n" + x for x, y in zip(chunk_neighbors_docs, chunk_neighbors_metadata)])

    prompt = f"""You are helping generate test data for recall@K with random chunks from a document.
    Task:
    1.If the chunks do not provide enough context to create a question, return "Not enough context".
    2.Create question(s) that would be asked by human users, that can be used to test Recall@K.
    3.Ask questions that can be derived from the data, not questions about the section itself.
    4.Keep answer concise, and include relevant chunkIds for labels. Make sure chunkIds are ints.
    Example output format (JSON, minified):
    '{{"chunks": [{{"Question": "Question here", "Labels": [chunkIds]}}]}}'
    Input:
'''{prompt_string}'''"""
    model = ChatVertexAI(model="gemini-2.0-flash-001", temperature=0)
    response = model.invoke(prompt)
    return response.content


def write_question_to_file(question: str, chunk_file: str, chunk_page: int):
    question = re.sub(r"```(?:json)?\n?|\n?```", "", question)
    json_question = json.loads(question)
    for q in json_question["chunks"]:
        q["chunk_file"] = chunk_file
        q["chunk_page"] = chunk_page
    if not os.path.exists("../Data/testing"):
        os.makedirs("../Data/testing")
    with open("../Data/testing/questions_with_labels.jsonl", "a") as file:
        file.write(json.dumps(json_question, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    for i in range(195):
        random_chunk = get_random_chunk("Iowa_Law_v6")
        chunk_file = random_chunk["metadatas"][0]["filename"]
        chunk_page = random_chunk["metadatas"][0]["page"]
        question = create_question_from_chunk(random_chunk)
        write_question_to_file(question, chunk_file, chunk_page)
