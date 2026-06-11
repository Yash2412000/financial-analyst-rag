import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import pdfplumber


CHUNK_PATH = Path("db/chunks.json")

# How big each chunk is and how much they overlap
CHUNK_SIZE = 400    
CHUNK_OVERLAP = 80


def clean_text(text: str) -> str:
    """Remove noise from raw text."""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)           # multiple spaces → single space
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # remove non-English characters
    return text.strip()


def chunk_text(text: str, source: str) -> List[Dict]:
    """Split a long text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_str = " ".join(words[start:end])
        chunk_id = hashlib.md5(f"{source}:{start}".encode()).hexdigest()[:12]

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_str,
            "source": source,
        })

        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def load_csv_dataset(csv_path: str) -> List[Dict]:
    """Load and process a CSV file."""
    df = pd.read_csv(csv_path)
    df.drop_duplicates(subset=["question", "answer"], inplace=True)
    df.fillna("", inplace=True)  # Replace NaN with empty strings
    chunks = []
    for _, row in df.iterrows():
        source = f"{row.get('ticker', 'Unknown')} | {row.get('filing', 'Unknown')}"
        combined = (
            f"Question: {clean_text(row.get('question', ''))}\n"
            f"Answer: {clean_text(row.get('answer', ''))}\n"
            f"Context: {clean_text(row.get('context', ''))}"
        )
        chunk_id = hashlib.md5(combined[:100].encode()).hexdigest()[:12]
        chunks.append({
            "chunk_id": chunk_id,
            "text": combined,
            "source": source,
        })
    return chunks


def load_pdf(pdf_path: str, filename: Optional[str] = None) -> List[Dict]:
    """Extract text from a PDF and split into chunks."""
    source_name = filename or Path(pdf_path).name
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(clean_text(text))

    combined = " ".join(full_text)

    if len(combined.split()) < 50:
        raise ValueError(
            f"Could not extract readable text from {source_name}. "
            "The PDF may be scanned. Try an OCR tool first."
        )

    return chunk_text(combined, source=source_name)


def load_chunk_store() -> List[Dict]:
    """Load all chunks from disk."""
    if CHUNK_PATH.exists():
        with open(CHUNK_PATH, "r") as f:
            return json.load(f)
    return []


def save_chunk_store(chunks: List[Dict]) -> None:
    """Save all chunks to disk."""
    CHUNK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNK_PATH, "w") as f:
        json.dump(chunks, f)


def add_chunks_to_store(new_chunks: List[Dict]) -> List[Dict]:
    """Add new chunks, skip duplicates, save."""
    existing = load_chunk_store()
    existing_ids = {c["chunk_id"] for c in existing}
    added = [c for c in new_chunks if c["chunk_id"] not in existing_ids]
    all_chunks = existing + added
    save_chunk_store(all_chunks)
    return all_chunks


def get_unique_sources(chunks: List[Dict]) -> List[str]:
    """Return list of all unique source names."""
    return sorted(set(c["source"] for c in chunks))


def bootstrap_from_csv(csv_path: str) -> List[Dict]:
    """First-time setup: load CSV into chunk store."""
    print(f"Loading dataset from {csv_path}...")
    chunks = load_csv_dataset(csv_path)
    all_chunks = add_chunks_to_store(chunks)
    print(f"Done. {len(all_chunks)} chunks from {len(get_unique_sources(all_chunks))} sources.")
    return all_chunks

from src.ingestion import bootstrap_from_csv, load_chunk_store, get_unique_sources

# Load CSV into chunk store
chunks = bootstrap_from_csv("db/Financial-QA-10k.csv")

# Check results
print(f"Total chunks: {len(chunks)}")
print(f"Total sources: {len(get_unique_sources(chunks))}")
print(f"\nSample chunk:")
print(chunks[0])