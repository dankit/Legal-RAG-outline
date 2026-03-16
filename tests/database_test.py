"""Tests for VectorDatabaseClient and VectorDatabaseCollection."""

from app.core.vector_database import VectorDatabaseClient


db = VectorDatabaseClient()
collection = db.get_collection("tests")


def clear_test_data():
    collection.collection.delete(where={"filename": "test_file"})


def test_store_and_retrieve_chunk():
    clear_test_data()
    collection.store_chunk_embeddings(
        ["test_chunk"], 
        [[0.1, 0.2, 0.3]], 
        [{"filename": "test_file", "chunkId": 100, "page": 1}]
    )
    result = collection.get_chunk_by_id("test_file:100")
    assert result['ids'] == ["test_file:100"]
    assert result['documents'] == ["test_chunk"]
    collection.delete_chunk_by_id("test_file:100")


def test_store_multiple_chunks():
    clear_test_data()
    collection.store_chunk_embeddings(
        ["chunk_a", "chunk_b"],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [
            {"filename": "test_file", "chunkId": 200, "page": 1},
            {"filename": "test_file", "chunkId": 201, "page": 2}
        ]
    )
    result_a = collection.get_chunk_by_id("test_file:200")
    result_b = collection.get_chunk_by_id("test_file:201")
    assert result_a['ids'] == ["test_file:200"]
    assert result_b['ids'] == ["test_file:201"]
    collection.delete_chunk_by_id("test_file:200")
    collection.delete_chunk_by_id("test_file:201")


def test_get_chunk_neighbors():
    clear_test_data()
    collection.store_chunk_embeddings(
        ["chunk_0", "chunk_1", "chunk_2"],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        [
            {"filename": "test_file", "chunkId": 300, "page": 1},
            {"filename": "test_file", "chunkId": 301, "page": 1},
            {"filename": "test_file", "chunkId": 302, "page": 2}
        ]
    )
    neighbors = collection.get_chunk_neighbors("test_file", 301, k=1)
    neighbor_ids = set(neighbors['ids'])
    assert "test_file:300" in neighbor_ids
    assert "test_file:302" in neighbor_ids
    assert "test_file:301" not in neighbor_ids
    collection.delete_chunk_by_id("test_file:300")
    collection.delete_chunk_by_id("test_file:301")
    collection.delete_chunk_by_id("test_file:302")


def test_last_processed_chunk_id_tracking():
    clear_test_data()
    assert collection.get_last_processed_chunk_id("test_file") == -1

    collection.store_chunk_embeddings(
        ["chunk_a", "chunk_b"],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [
            {"filename": "test_file", "chunkId": 0, "page": 1},
            {"filename": "test_file", "chunkId": 1, "page": 2}
        ]
    )
    assert collection.get_last_processed_chunk_id("test_file") == 1
    collection.delete_chunk_by_id("test_file:0")
    collection.delete_chunk_by_id("test_file:1")
