"""Shared data models for docintel_kit.

Every module in the package consumes and/or produces the models defined here so
that data flows cleanly between capabilities (e.g. a `Table` produced by
`tables.py` or `spreadsheet.py` can be handed to `llm_tasks.qa_over_tables`
without any conversion, and a `ParseResult` from `parsing.py` can be fed into
`layout.py` or `extraction.py`).

All models are Pydantic v2 `BaseModel`s so that they get free validation,
`.model_dump()` / `.model_dump_json()` serialization, and editor autocompletion.
Convenience methods (`get_page_text`, `to_dataframe`, etc.) are added directly
on the models to keep call sites in the rest of the library short.
"""

from __future__ import annotations

import csv
import io
import json
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BoundingBox",
    "Document",
    "Page",
    "ParseResult",
    "OcrWord",
    "OcrResult",
    "BlockType",
    "LayoutBlock",
    "LayoutResult",
    "Table",
    "ExtractedField",
    "ExtractionResult",
    "RagChunk",
    "RagMatch",
    "RagSearchResult",
]


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in page coordinates (points or pixels).

    Coordinates are expressed with the origin at the top-left of the page,
    matching the convention used by pdfplumber, layoutparser, and most OCR
    engines. ``x1``/``y1`` are the bottom-right corner.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    page_index: int = Field(default=0, description="0-indexed page this box belongs to.")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class Document(BaseModel):
    """Metadata about a source document being processed.

    This is intentionally lightweight — it does not hold page content (see
    :class:`ParseResult`/:class:`Page` for that). It exists so downstream
    results (OCR, layout, extraction, RAG) can all reference the same
    document identity without duplicating source-loading logic.
    """

    id: str = Field(description="Stable identifier, e.g. file path or a content hash.")
    source_path: Optional[str] = Field(default=None, description="Filesystem path, if any.")
    mime_type: Optional[str] = Field(default=None, description="Detected or provided MIME type.")
    page_count: Optional[int] = Field(default=None, description="Number of pages, if known.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Page(BaseModel):
    """A single page (or slide/sheet, depending on format) of a document."""

    index: int = Field(description="0-indexed page number.")
    width: Optional[float] = Field(default=None, description="Page width in points/pixels.")
    height: Optional[float] = Field(default=None, description="Page height in points/pixels.")
    text: str = Field(default="", description="Extracted plain text for this page.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseResult(BaseModel):
    """Result of :func:`docintel_kit.parsing.parse_document`.

    Holds normalized per-page text plus the originating :class:`Document`
    metadata. This is the common currency consumed by layout analysis, OCR
    fallback logic, structured extraction, and RAG indexing.
    """

    document: Document
    pages: list[Page] = Field(default_factory=list)
    backend: str = Field(description="Name of the parser backend that produced this result.")
    warnings: list[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenation of all page texts, separated by form-feed-style breaks."""
        return "\n\n".join(page.text for page in self.pages)

    def get_page_text(self, page_index: int) -> str:
        """Return the extracted text for a single page.

        Raises:
            IndexError: if ``page_index`` is out of range.
        """
        for page in self.pages:
            if page.index == page_index:
                return page.text
        raise IndexError(f"No page with index {page_index} in ParseResult")


class OcrWord(BaseModel):
    """A single recognized word/token from an OCR backend, with its location."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0, description="Normalized confidence in [0, 1].")
    bbox: BoundingBox


class OcrResult(BaseModel):
    """Result of :func:`docintel_kit.ocr.run_ocr`.

    Stores flat word-level output (easiest to reason about across backends)
    plus reconstructed per-page text for convenience.
    """

    document: Document
    words: list[OcrWord] = Field(default_factory=list)
    backend: str = Field(description="Name of the OCR backend that produced this result.")
    page_texts: dict[int, str] = Field(
        default_factory=dict, description="Reconstructed text per page index."
    )

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.page_texts.get(i, "") for i in sorted(self.page_texts))

    def get_page_text(self, page_index: int) -> str:
        """Return reconstructed OCR text for a page, or "" if none was recognized."""
        return self.page_texts.get(page_index, "")

    def words_for_page(self, page_index: int) -> list[OcrWord]:
        return [w for w in self.words if w.bbox.page_index == page_index]


class BlockType(str, Enum):
    """Coarse structural categories detected by layout analysis."""

    TITLE = "title"
    HEADING = "heading"
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"
    OTHER = "other"


class LayoutBlock(BaseModel):
    """A single detected layout region on a page (paragraph, table, figure, ...)."""

    block_id: str
    block_type: BlockType
    bbox: BoundingBox
    text: Optional[str] = Field(default=None, description="Text content, if extracted.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reading_order: Optional[int] = Field(
        default=None, description="Position of this block within the page's reading order."
    )


class LayoutResult(BaseModel):
    """Result of :func:`docintel_kit.layout.analyze_layout`."""

    document: Document
    blocks: list[LayoutBlock] = Field(default_factory=list)
    backend: str = Field(description="Name of the layout backend that produced this result.")

    def get_blocks_by_type(self, block_type: BlockType) -> list[LayoutBlock]:
        return [b for b in self.blocks if b.block_type == block_type]

    def get_page_blocks(self, page_index: int) -> list[LayoutBlock]:
        return [b for b in self.blocks if b.bbox.page_index == page_index]


class Table(BaseModel):
    """A generic tabular structure.

    This is the single representation shared by PDF/image table extraction
    (:mod:`docintel_kit.tables`) and spreadsheet ingestion
    (:mod:`docintel_kit.spreadsheet`), so downstream consumers (LLM table QA,
    RAG over tables, structured field extraction) never need to know the
    original source format.
    """

    table_id: str
    headers: Optional[list[str]] = Field(default=None)
    rows: list[list[str]] = Field(
        default_factory=list, description="Row-major cell values, all coerced to str."
    )
    page_index: Optional[int] = Field(default=None, description="Source page, if applicable.")
    bbox: Optional[BoundingBox] = Field(default=None)
    sheet_name: Optional[str] = Field(default=None, description="Source sheet, for spreadsheets.")
    source: str = Field(description="Origin, e.g. 'camelot', 'tabula', 'layout', 'spreadsheet'.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_dataframe(self):
        """Return this table as a :class:`pandas.DataFrame`.

        Imports pandas lazily so that constructing/validating a ``Table``
        never requires pandas to be installed — only calling this method does.
        """
        import pandas as pd

        if self.headers:
            return pd.DataFrame(self.rows, columns=self.headers)
        return pd.DataFrame(self.rows)

    def to_csv(self, path: Optional[str] = None) -> Optional[str]:
        """Serialize the table as CSV.

        Args:
            path: If given, write the CSV to this path and return ``None``.
                Otherwise return the CSV content as a string.
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        if self.headers:
            writer.writerow(self.headers)
        writer.writerows(self.rows)
        content = buffer.getvalue()
        if path is not None:
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write(content)
            return None
        return content

    def to_json(self) -> str:
        """Serialize the table as a JSON object with ``headers`` and ``rows`` keys."""
        return json.dumps({"headers": self.headers, "rows": self.rows}, ensure_ascii=False)


class ExtractedField(BaseModel):
    """A single extracted scalar field, with model confidence and provenance."""

    name: str
    value: Any = Field(default=None)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: Optional[BoundingBox] = Field(default=None)
    page_index: Optional[int] = Field(default=None)


class ExtractionResult(BaseModel):
    """Result of :func:`docintel_kit.extraction.extract_fields`.

    Scalar fields (dates, amounts, names, ...) live in ``fields``. Fields
    declared as table-typed in the schema (e.g. invoice line items) are
    returned as :class:`Table` objects in ``tables`` instead, keyed by field
    name, so callers get first-class row/column access rather than raw text.
    """

    document: Document
    backend: str
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    tables: dict[str, list[Table]] = Field(default_factory=dict)
    raw_schema: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def get_field(self, name: str) -> Optional[ExtractedField]:
        return self.fields.get(name)


class RagChunk(BaseModel):
    """A chunk of text (from a document or a table) stored in a RAG index."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int = 0
    page_index: Optional[int] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = Field(
        default=None, description="Dense embedding vector, populated at index time."
    )


class RagMatch(BaseModel):
    """A single search hit: a chunk plus its similarity score to the query."""

    chunk: RagChunk
    score: float = Field(description="Higher is more similar (cosine similarity by default).")


class RagSearchResult(BaseModel):
    """Result of :func:`docintel_kit.rag.search_documents`."""

    query: str
    collection: str
    matches: list[RagMatch] = Field(default_factory=list)

    def top_texts(self) -> list[str]:
        """Convenience accessor: just the matched chunk texts, in rank order."""
        return [m.chunk.text for m in self.matches]
