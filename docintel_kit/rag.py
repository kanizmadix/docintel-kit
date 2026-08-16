"""RAG / semantic search over documents and tables.

Provides :func:`index_documents` and :func:`search_documents`, backed by
`sentence-transformers <https://www.sbert.net/>`_ embeddings and an in-memory
vector store for v1. Documents are chunked into overlapping text windows;
tables are chunked row-by-row (each row rendered as "col: value, ..." text) so
semantic search can retrieve individual rows as well as prose passages.

The vector store is deliberately hidden behind :class:`BaseVectorStore` so a
persistent backend (PGVector, Qdrant, Chroma, ...) can be dropped in later
without changing :func:`index_documents`/:func:`search_documents`.
"""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Union

from .parsing import parse_document
from .types import RagChunk, RagMatch, RagSearchResult, Table

__all__ = [
    "BaseVectorStore",
    "InMemoryVectorStore",
    "index_documents",
    "search_documents",
    "index_tables",
]

_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_CHUNK_SIZE = 800
_DEFAULT_CHUNK_OVERLAP = 100


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping character windows.

    A simple, dependency-free chunker. It breaks on the requested size but
    tries to end on a whitespace boundary when one is nearby, to avoid
    splitting mid-word.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Prefer to break on whitespace within the last 20% of the window.
            search_from = max(start, end - int(chunk_size * 0.2))
            boundary = text.rfind(" ", search_from, end)
            if boundary != -1:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _table_to_row_texts(table: Table) -> list[str]:
    """Render each table row as a compact "header: value, ..." text chunk."""
    texts: list[str] = []
    headers = table.headers
    for row in table.rows:
        if headers:
            parts = [f"{h}: {v}" for h, v in zip(headers, row)]
        else:
            parts = list(row)
        texts.append(", ".join(parts))
    return texts


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class BaseVectorStore(ABC):
    """Interface for a vector store backing RAG indexing/search.

    Implement this to swap the in-memory v1 store for a persistent backend
    (PGVector, Qdrant, Chroma, ...) without changing
    :func:`index_documents`/:func:`search_documents`.
    """

    @abstractmethod
    def upsert(self, collection: str, chunks: list[RagChunk]) -> None:
        """Add or replace ``chunks`` (each already carrying an embedding) in ``collection``."""
        raise NotImplementedError

    @abstractmethod
    def query(self, collection: str, query_embedding: list[float], top_k: int) -> list[RagMatch]:
        """Return the ``top_k`` most similar chunks in ``collection`` to ``query_embedding``."""
        raise NotImplementedError


class InMemoryVectorStore(BaseVectorStore):
    """Simple process-local vector store using cosine similarity brute force.

    Suitable for prototyping and small-to-medium corpora. Not persisted
    across process restarts.
    """

    def __init__(self) -> None:
        self._collections: dict[str, list[RagChunk]] = {}

    def upsert(self, collection: str, chunks: list[RagChunk]) -> None:
        self._collections.setdefault(collection, [])
        existing = {c.chunk_id: c for c in self._collections[collection]}
        for chunk in chunks:
            existing[chunk.chunk_id] = chunk
        self._collections[collection] = list(existing.values())

    def query(self, collection: str, query_embedding: list[float], top_k: int) -> list[RagMatch]:
        chunks = self._collections.get(collection, [])
        scored = [
            RagMatch(chunk=chunk, score=_cosine_similarity(query_embedding, chunk.embedding or []))
            for chunk in chunks
            if chunk.embedding is not None
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]


class _EmbeddingModel:
    """Lazy wrapper around a sentence-transformers model.

    Keeps `sentence-transformers`/`torch` out of the import path until an
    embedding is actually requested, and caches one loaded model per name.
    """

    _cache: dict[str, "_EmbeddingModel"] = {}

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @classmethod
    def get(cls, model_name: str) -> "_EmbeddingModel":
        if model_name not in cls._cache:
            cls._cache[model_name] = cls(model_name)
        return cls._cache[model_name]

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]


_STORE = InMemoryVectorStore()


def register_vector_store(store: BaseVectorStore) -> None:
    """Replace the module-level vector store with a custom :class:`BaseVectorStore`.

    Call this once at startup to point `index_documents`/`search_documents` at
    a persistent backend instead of the default in-memory store.
    """
    global _STORE
    _STORE = store


def index_documents(
    files: list[Union[str, bytes]],
    collection: str,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> None:
    """Parse, chunk, embed, and index a set of documents for semantic search.

    Args:
        files: Filesystem paths or raw bytes for the documents to index. Each
            is parsed via :func:`docintel_kit.parsing.parse_document`, so any
            format supported there (PDF, DOCX, PPTX, HTML) works here. For
            scanned/image documents, OCR first and index the resulting text
            via lower-level chunk construction, or extend this function with
            a custom loader.
        collection: Name of the collection to index into. Re-running with the
            same collection name upserts (replaces) chunks with matching ids;
            it does not deduplicate across differently-chunked re-runs.
        embedding_model: Hugging Face sentence-transformers model id.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks, in characters.

    Returns:
        None. Indexed chunks are stored in the active vector store (see
        :func:`register_vector_store`).
    """
    embedder = _EmbeddingModel.get(embedding_model)

    all_chunks: list[RagChunk] = []
    for file in files:
        parse_result = parse_document(file)
        document_id = parse_result.document.id
        for page in parse_result.pages:
            texts = _chunk_text(page.text, chunk_size, chunk_overlap)
            for i, text in enumerate(texts):
                all_chunks.append(
                    RagChunk(
                        chunk_id=f"{document_id}-p{page.index}-c{i}",
                        document_id=document_id,
                        text=text,
                        chunk_index=i,
                        page_index=page.index,
                    )
                )

    if not all_chunks:
        return

    embeddings = embedder.encode([c.text for c in all_chunks])
    for chunk, embedding in zip(all_chunks, embeddings):
        chunk.embedding = embedding

    _STORE.upsert(collection, all_chunks)


def index_tables(
    tables: list[Table],
    collection: str,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
) -> None:
    """Index tables for semantic search over individual rows.

    Each row is rendered as a compact "header: value, ..." text chunk and
    embedded/indexed the same way as document chunks, so
    :func:`search_documents` can retrieve matching rows alongside prose
    passages from the same collection.

    Args:
        tables: Tables to index, e.g. from
            :func:`docintel_kit.tables.extract_tables_from_document` or
            :func:`docintel_kit.spreadsheet.parse_spreadsheet`.
        collection: Name of the collection to index into.
        embedding_model: Hugging Face sentence-transformers model id.
    """
    embedder = _EmbeddingModel.get(embedding_model)

    all_chunks: list[RagChunk] = []
    for table in tables:
        row_texts = _table_to_row_texts(table)
        for i, text in enumerate(row_texts):
            all_chunks.append(
                RagChunk(
                    chunk_id=f"{table.table_id}-row{i}",
                    document_id=table.table_id,
                    text=text,
                    chunk_index=i,
                    page_index=table.page_index,
                    metadata={"source": table.source, "sheet_name": table.sheet_name},
                )
            )

    if not all_chunks:
        return

    embeddings = embedder.encode([c.text for c in all_chunks])
    for chunk, embedding in zip(all_chunks, embeddings):
        chunk.embedding = embedding

    _STORE.upsert(collection, all_chunks)


def search_documents(
    query: str,
    collection: str,
    top_k: int = 5,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
) -> RagSearchResult:
    """Semantically search a previously indexed collection.

    Args:
        query: Natural-language search query.
        collection: Name of the collection to search (must have been
            populated via :func:`index_documents` and/or :func:`index_tables`).
        top_k: Maximum number of matches to return.
        embedding_model: Must match the model used at index time, so query
            and chunk embeddings live in the same vector space.

    Returns:
        A :class:`RagSearchResult` with matches ranked by cosine similarity,
        highest first.
    """
    embedder = _EmbeddingModel.get(embedding_model)
    query_embeddings = embedder.encode([query])
    query_embedding = query_embeddings[0] if query_embeddings else []
    matches = _STORE.query(collection, query_embedding, top_k)
    return RagSearchResult(query=query, collection=collection, matches=matches)
