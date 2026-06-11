import pytest
import json
from pathlib import Path
from src.ingestion import clean_text, chunk_text, load_chunk_store, save_chunk_store, add_chunks_to_store


# ─── Test clean_text ───────────────────────────────────────────

def test_clean_text_removes_extra_spaces():
    result = clean_text("hello    world")
    assert result == "hello world"

def test_clean_text_removes_urls():
    result = clean_text("visit https://nvidia.com today")
    assert "https" not in result

def test_clean_text_handles_numbers():
    result = clean_text("revenue was $26.9 billion")
    assert result == "revenue was $26.9 billion"

def test_clean_text_handles_empty():
    result = clean_text("")
    assert result == ""


# ─── Test chunk_text ───────────────────────────────────────────

def test_chunk_text_creates_multiple_chunks():
    text = " ".join([f"word{i}" for i in range(1000)])
    chunks = chunk_text(text, source="test.pdf")
    assert len(chunks) > 1

def test_chunk_text_correct_keys():
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunk_text(text, source="test.pdf")
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "source" in chunk

def test_chunk_text_correct_source():
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunk_text(text, source="AAPL_2023.pdf")
    for chunk in chunks:
        assert chunk["source"] == "AAPL_2023.pdf"

def test_chunk_text_overlap():
    text = " ".join([f"word{i}" for i in range(1000)])
    chunks = chunk_text(text, source="test.pdf")
    words_0 = set(chunks[0]["text"].split())
    words_1 = set(chunks[1]["text"].split())
    assert len(words_0 & words_1) > 0



# ─── Test chunk store ──────────────────────────────────────────

def test_save_and_load_chunk_store(tmp_path, monkeypatch):
    from src import ingestion
    monkeypatch.setattr(ingestion, "CHUNK_PATH", tmp_path / "chunks.json")

    test_chunks = [{"chunk_id": "abc123", "text": "hello world", "source": "test"}]
    save_chunk_store(test_chunks)
    loaded = load_chunk_store()
    assert loaded == test_chunks

def test_load_empty_store(tmp_path, monkeypatch):
    from src import ingestion
    monkeypatch.setattr(ingestion, "CHUNK_PATH", tmp_path / "chunks.json")

    result = load_chunk_store()
    assert result == []

def test_add_chunks_no_duplicates(tmp_path, monkeypatch):
    from src import ingestion
    monkeypatch.setattr(ingestion, "CHUNK_PATH", tmp_path / "chunks.json")

    chunk = {"chunk_id": "abc123", "text": "hello", "source": "test"}
    add_chunks_to_store([chunk])
    add_chunks_to_store([chunk])  # add same chunk again
    stored = load_chunk_store()
    assert len(stored) == 1

def test_add_chunks_different_ids(tmp_path, monkeypatch):
    from src import ingestion
    monkeypatch.setattr(ingestion, "CHUNK_PATH", tmp_path / "chunks.json")

    chunk1 = {"chunk_id": "abc123", "text": "hello", "source": "test"}
    chunk2 = {"chunk_id": "xyz789", "text": "world", "source": "test"}
    add_chunks_to_store([chunk1])
    add_chunks_to_store([chunk2])
    stored = load_chunk_store()
    assert len(stored) == 2