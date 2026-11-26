"""
Document loader: reads PDFs, TXT, Markdown, and CSV files from a folder.
Returns a list of Document objects with text and metadata.
"""
from __future__ import annotations

import os
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Document:
    source: str           # filename
    content: str          # full text
    doc_type: str = ""    # pdf | txt | csv | md
    metadata: dict = field(default_factory=dict)


def load_documents(folder: str) -> List[Document]:
    """Load all supported documents from a folder (recursive)."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Document folder not found: {folder}")

    docs: List[Document] = []
    supported = {".pdf", ".txt", ".md", ".csv", ".json"}

    for fpath in sorted(folder_path.rglob("*")):
        if fpath.suffix.lower() not in supported:
            continue
        try:
            doc = _load_file(fpath)
            if doc and doc.content.strip():
                docs.append(doc)
        except Exception as e:
            print(f"[WARNING] Could not load {fpath.name}: {e}")

    return docs


def _load_file(fpath: Path) -> Document | None:
    ext = fpath.suffix.lower()
    source = fpath.name

    if ext == ".pdf":
        return _load_pdf(fpath, source)
    elif ext == ".csv":
        return _load_csv(fpath, source)
    elif ext in (".txt", ".md"):
        return _load_text(fpath, source, ext.lstrip("."))
    elif ext == ".json":
        return _load_text(fpath, source, "json")
    return None


def _load_pdf(fpath: Path, source: str) -> Document:
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(fpath))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i+1}]\n{text}")
        content = "\n\n".join(pages)
    except ImportError:
        # Fallback: read raw bytes as text (lossy but functional)
        content = fpath.read_text(encoding="utf-8", errors="replace")
    return Document(source=source, content=content, doc_type="pdf")


def _load_csv(fpath: Path, source: str) -> Document:
    rows = []
    with open(fpath, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows.append("Columns: " + ", ".join(headers))
        for row in reader:
            rows.append(str(dict(row)))
    content = "\n".join(rows)
    return Document(source=source, content=content, doc_type="csv")


def _load_text(fpath: Path, source: str, doc_type: str) -> Document:
    content = fpath.read_text(encoding="utf-8", errors="replace")
    return Document(source=source, content=content, doc_type=doc_type)


# ── Simple chunker (no external vector DB needed) ──────────────────────────

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[dict]:
    """
    Split documents into overlapping text chunks.
    Returns list of {id, source, chunk_index, text}.
    """
    chunks = []
    for doc in docs:
        words = doc.content.split()
        step = chunk_size - overlap
        idx = 0
        chunk_num = 0
        while idx < len(words):
            window = words[idx : idx + chunk_size]
            chunks.append(
                {
                    "id": f"{doc.source}::chunk{chunk_num}",
                    "source": doc.source,
                    "chunk_index": chunk_num,
                    "text": " ".join(window),
                }
            )
            idx += step
            chunk_num += 1
    return chunks


def keyword_search(
    query: str,
    chunks: List[dict],
    top_k: int = 5,
) -> List[dict]:
    """
    Simple TF-style keyword retrieval (no embeddings required).
    Falls back gracefully if sentence-transformers aren't installed.
    """
    q_words = set(query.lower().split())
    scored = []
    for c in chunks:
        c_words = c["text"].lower().split()
        c_word_set = set(c_words)
        overlap = len(q_words & c_word_set)
        # Prefer earlier chunks slightly (freshness of document)
        score = overlap / (len(c_words) + 1) + 0.001 / (c["chunk_index"] + 1)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def build_context(query: str, chunks: List[dict], top_k: int = 5) -> str:
    """Return a formatted context string from top-k retrieved chunks."""
    hits = keyword_search(query, chunks, top_k)
    parts = []
    for h in hits:
        parts.append(f"[Source: {h['source']}]\n{h['text']}")
    return "\n\n---\n\n".join(parts)
