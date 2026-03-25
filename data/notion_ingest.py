"""
Pull and chunk Notion workspace documents.

This script is designed to be called programmatically — the actual Notion API calls
happen via the MCP tools (notion-search, notion-fetch) from the Claude Code session.

This module provides the chunking and processing logic.
"""

import json
import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """Clean Notion block text — strip formatting artifacts."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Chunk text into spans of approximately chunk_size tokens.
    Uses word-level splitting with overlap (matching Notion's span approach).

    Args:
        text: input text
        chunk_size: approximate tokens per chunk (words as proxy)
        overlap: overlap between chunks in words
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def process_page(page_id: str, title: str, content: str, chunk_size: int = 512) -> List[Dict[str, Any]]:
    """
    Process a single Notion page into chunks with metadata.

    Returns list of chunk dicts ready for embedding.
    """
    cleaned = clean_text(content)
    if not cleaned:
        return []

    chunks = chunk_text(cleaned, chunk_size=chunk_size)

    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "page_id": page_id,
            "title": title,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk,
        })

    return results


def save_chunks(chunks: List[Dict[str, Any]], output_path: str):
    """Save chunks as JSONL."""
    with open(output_path, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    print(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: str) -> List[Dict[str, Any]]:
    """Load chunks from JSONL."""
    chunks = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks
