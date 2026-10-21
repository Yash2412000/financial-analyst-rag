import numpy as np

from src.retrieval import HybridRetriever


class FakeEncoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, **kwargs):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class FailingEncoder:
    def encode(self, texts, **kwargs):
        raise RuntimeError("encoder unavailable")


def _sample_chunks():
    return [
        {"chunk_id": "a", "text": "apple revenue growth strong", "source": "10k_a", "metadata": {"year": 2023}},
        {"chunk_id": "b", "text": "banana smoothie market trends", "source": "10k_b", "metadata": {"year": 2022}},
        {"chunk_id": "c", "text": "apple services segment outlook", "source": "10k_c", "metadata": {"year": 2024}},
    ]


def test_search_empty_store_returns_empty():
    retriever = HybridRetriever([], use_dense=False)
    assert retriever.search("apple") == []


def test_search_blank_query_returns_empty():
    retriever = HybridRetriever(_sample_chunks(), use_dense=False)
    assert retriever.search("   ") == []


def test_score_fusion_changes_top_result():
    chunks = _sample_chunks()
    encoder = FakeEncoder(
        {
            "apple revenue growth strong": [1.0, 0.0],
            "banana smoothie market trends": [0.0, 1.0],
            "apple services segment outlook": [0.2, 0.8],
            "apple revenue": [0.0, 1.0],
        }
    )

    lexical_heavy = HybridRetriever(chunks, semantic_weight=0.1, encoder=encoder)
    semantic_heavy = HybridRetriever(chunks, semantic_weight=0.9, encoder=encoder)

    lexical_results = lexical_heavy.search("apple revenue", top_k=3)
    semantic_results = semantic_heavy.search("apple revenue", top_k=3)

    assert lexical_results[0]["chunk_id"] == "a"
    assert semantic_results[0]["chunk_id"] == "b"


def test_preserves_metadata_and_caps_top_k():
    retriever = HybridRetriever(_sample_chunks(), use_dense=False)
    results = retriever.search("apple", top_k=10)

    assert len(results) == 3
    assert results[0]["metadata"]["year"] == 2023


def test_duplicate_chunk_ids_are_deduplicated():
    chunks = [
        {"chunk_id": "dup", "text": "first chunk apple", "source": "s1"},
        {"chunk_id": "dup", "text": "second chunk banana", "source": "s2"},
    ]
    retriever = HybridRetriever(chunks, use_dense=False)
    results = retriever.search("banana", top_k=5)

    assert len(results) == 1
    assert results[0]["text"] == "first chunk apple"


def test_dense_failure_falls_back_to_bm25():
    retriever = HybridRetriever(_sample_chunks(), encoder=FailingEncoder())
    results = retriever.search("apple revenue", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "a"


def test_invalid_embedding_name_falls_back_when_not_raising():
    retriever = HybridRetriever(_sample_chunks(), embedding_model_name="", use_dense=True)
    results = retriever.search("apple revenue", top_k=2)

    assert len(results) == 2
