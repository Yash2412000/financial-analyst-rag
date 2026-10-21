# 📊 Financial Analyst RAG

> Work in progress — being built phase by phase.

## What this will be
A production-grade RAG system for querying financial 10-K filings and annual reports.

- Hybrid BM25 + FAISS retrieval
- Cross-encoder reranking
- Groq Llama 3.1 70B
- Confidence estimation
- RAGAS-inspired evaluation
- PDF upload pipeline

## Status
✅ Phase 2 complete — hybrid retrieval is implemented.

### Phase 2 usage (local example)
```python
from src.retrieval import HybridRetriever

chunks = [
    {"chunk_id": "1", "text": "Apple reported strong revenue growth in 2023.", "source": "AAPL_2023_10K.pdf"},
    {"chunk_id": "2", "text": "The company expanded services and operating margin.", "source": "AAPL_2023_10K.pdf"},
]

retriever = HybridRetriever(chunks, semantic_weight=0.6)
results = retriever.search("revenue growth", top_k=2)
```

### Next phase
- 🚧 Phase 3 planned — reranking + answer generation pipeline.

## Author
**Yashraj Rai** — MSc Data Science, University of Bath
[GitHub](https://github.com/Yash2412000)