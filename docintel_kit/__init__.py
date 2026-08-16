"""docintel_kit: unified, local-first document processing and analysis toolkit.

This package exposes a small set of top-level functions that cover the full
document-AI pipeline — parsing, OCR, layout analysis, table extraction,
spreadsheet ingestion, structured field extraction, LLM tasks, and RAG — while
keeping each capability's implementation in its own module behind a swappable
backend interface.

Typical usage::

    from docintel_kit import parse_document, analyze_layout, extract_tables_from_document

    result = parse_document("invoice.pdf")
    layout = analyze_layout("invoice.pdf")
    tables = extract_tables_from_document("invoice.pdf")

Nothing in this package makes network calls by default. The only exceptions
are the LLM-backed functions (`summarize_document`, `classify_document`,
`qa_over_tables`) when invoked with a network engine (e.g. ``engine="claude"``),
and RAG embedding models the first time they are downloaded from Hugging Face.
"""

from __future__ import annotations

from .types import (
    BlockType,
    BoundingBox,
    Document,
    ExtractedField,
    ExtractionResult,
    LayoutBlock,
    LayoutResult,
    OcrResult,
    OcrWord,
    Page,
    ParseResult,
    RagChunk,
    RagMatch,
    RagSearchResult,
    Table,
)
from .parsing import BaseParserBackend, parse_document
from .ocr import BaseOcrBackend, run_ocr
from .layout import BaseLayoutBackend, analyze_layout
from .tables import extract_tables_from_document
from .spreadsheet import parse_spreadsheet
from .extraction import BaseExtractionBackend, extract_fields
from .llm_tasks import (
    BaseLlmClient,
    ClaudeClient,
    classify_document,
    qa_over_tables,
    summarize_document,
)
from .rag import index_documents, search_documents

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # types
    "BlockType",
    "BoundingBox",
    "Document",
    "ExtractedField",
    "ExtractionResult",
    "LayoutBlock",
    "LayoutResult",
    "OcrResult",
    "OcrWord",
    "Page",
    "ParseResult",
    "RagChunk",
    "RagMatch",
    "RagSearchResult",
    "Table",
    # parsing
    "BaseParserBackend",
    "parse_document",
    # ocr
    "BaseOcrBackend",
    "run_ocr",
    # layout
    "BaseLayoutBackend",
    "analyze_layout",
    # tables
    "extract_tables_from_document",
    # spreadsheet
    "parse_spreadsheet",
    # extraction
    "BaseExtractionBackend",
    "extract_fields",
    # llm tasks
    "BaseLlmClient",
    "ClaudeClient",
    "summarize_document",
    "classify_document",
    "qa_over_tables",
    # rag
    "index_documents",
    "search_documents",
]
