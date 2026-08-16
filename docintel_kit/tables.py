"""Table detection & extraction from PDFs, documents, and images.

Primary strategy: use `camelot-py <https://camelot-py.readthedocs.io/>`_
(``lattice`` and ``stream`` flavors) to extract tables directly from PDFs.
Camelot requires Ghostscript to be installed as a system dependency.

For scanned/image-based documents where camelot finds nothing, this module
falls back to :func:`docintel_kit.layout.analyze_layout` to locate ``TABLE``
regions and :func:`docintel_kit.ocr.run_ocr` to recover cell text within those
regions (a coarse, best-effort fallback — for tabular scans use a
dedicated fine-tuned model if accuracy is critical).

An optional `tabula-py <https://github.com/chezou/tabula-py>`_-based path is
available via ``method="tabula"`` for users who prefer/require it (it depends
on a local Java runtime).

Results are returned as a list of the shared :class:`~docintel_kit.types.Table`
model, identical to the representation used by
:mod:`docintel_kit.spreadsheet`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union

from .types import BlockType, BoundingBox, Table

__all__ = ["extract_tables_from_document"]


def _make_document_id(input_: Union[str, bytes], source_path: Optional[str]) -> str:
    if source_path:
        return source_path
    data = input_ if isinstance(input_, bytes) else input_.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:32]


def _is_pdf(input_: Union[str, bytes]) -> bool:
    if isinstance(input_, str):
        return Path(input_).suffix.lower() == ".pdf"
    return input_.startswith(b"%PDF")


def _extract_with_camelot(input_: Union[str, bytes], document_id: str) -> list[Table]:
    import camelot

    # camelot-py only accepts file paths, so bytes input is spooled to a
    # temporary file first.
    if isinstance(input_, bytes):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(input_)
            tmp_path = tmp.name
        try:
            return _run_camelot(tmp_path, document_id)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return _run_camelot(input_, document_id)


def _run_camelot(path: str, document_id: str) -> list[Table]:
    import camelot

    tables: list[Table] = []
    for flavor in ("lattice", "stream"):
        try:
            camelot_tables = camelot.read_pdf(path, pages="all", flavor=flavor)
        except Exception:
            continue
        for i, ct in enumerate(camelot_tables):
            df = ct.df
            if df.empty:
                continue
            if df.shape[1] < 2:
                # A single-column "table" is almost always camelot's `stream`
                # flavor mistaking ordinary paragraph text for tabular data
                # (each text line becomes a 1-column row). Real tables have at
                # least two columns, so this is a cheap, effective false-
                # positive filter without needing a stricter accuracy cutoff
                # that would also reject genuinely sparse/low-confidence
                # tables.
                continue
            headers = [str(c) for c in df.iloc[0].tolist()]
            rows = [[str(v) for v in row] for row in df.iloc[1:].values.tolist()]
            page_index = int(ct.page) - 1 if str(ct.page).isdigit() else None
            x0, y0, x1, y1 = ct._bbox if hasattr(ct, "_bbox") else (0.0, 0.0, 0.0, 0.0)
            tables.append(
                Table(
                    table_id=f"{document_id}-camelot-{flavor}-{i}",
                    headers=headers,
                    rows=rows,
                    page_index=page_index,
                    bbox=BoundingBox(
                        x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                        page_index=page_index or 0,
                    ),
                    source=f"camelot-{flavor}",
                    confidence=float(ct.parsing_report.get("accuracy", 100.0)) / 100.0,
                )
            )
        if tables:
            # Prefer the first flavor that produced results to avoid duplicates.
            break
    return tables


def _extract_with_tabula(input_: Union[str, bytes], document_id: str) -> list[Table]:
    import tabula

    if isinstance(input_, bytes):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(input_)
            tmp_path = tmp.name
        try:
            dfs = tabula.read_pdf(tmp_path, pages="all", multiple_tables=True)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        dfs = tabula.read_pdf(input_, pages="all", multiple_tables=True)

    tables: list[Table] = []
    for i, df in enumerate(dfs):
        if df.empty:
            continue
        headers = [str(c) for c in df.columns.tolist()]
        rows = [[str(v) for v in row] for row in df.values.tolist()]
        tables.append(
            Table(
                table_id=f"{document_id}-tabula-{i}",
                headers=headers,
                rows=rows,
                source="tabula",
                confidence=0.8,
            )
        )
    return tables


def _extract_with_layout_and_ocr(input_: Union[str, bytes], document_id: str) -> list[Table]:
    """Fallback for scanned documents: locate table regions via layout analysis,
    then OCR within each region to recover a best-effort grid of text lines.

    This is intentionally coarse (it does not infer column boundaries) and is
    meant as a last resort when camelot/tabula find nothing, e.g. for
    image-only PDFs.
    """
    from .layout import analyze_layout
    from .ocr import run_ocr

    layout_result = analyze_layout(input_)
    table_blocks = layout_result.get_blocks_by_type(BlockType.TABLE)
    if not table_blocks:
        return []

    ocr_result = run_ocr(input_)
    tables: list[Table] = []
    for i, block in enumerate(table_blocks):
        words_in_block = [
            w
            for w in ocr_result.words_for_page(block.bbox.page_index)
            if w.bbox.x0 >= block.bbox.x0
            and w.bbox.x1 <= block.bbox.x1
            and w.bbox.y0 >= block.bbox.y0
            and w.bbox.y1 <= block.bbox.y1
        ]
        if not words_in_block:
            continue
        # Group words into rows by rounding their vertical position, then sort
        # left-to-right within each row. This is a heuristic, not true column
        # detection.
        rows_map: dict[int, list] = {}
        for w in words_in_block:
            row_key = round(w.bbox.y0 / 10)
            rows_map.setdefault(row_key, []).append(w)
        rows = [
            [w.text for w in sorted(row_words, key=lambda w: w.bbox.x0)]
            for _, row_words in sorted(rows_map.items())
        ]
        tables.append(
            Table(
                table_id=f"{document_id}-layout-ocr-{i}",
                headers=None,
                rows=rows,
                page_index=block.bbox.page_index,
                bbox=block.bbox,
                source="layout+ocr",
                confidence=block.confidence * 0.6,
            )
        )
    return tables


def extract_tables_from_document(
    input: Union[str, bytes],
    method: str = "camelot",
) -> list[Table]:
    """Extract tables from a PDF, document, or scanned image.

    Args:
        input: A filesystem path or raw bytes for a PDF (or image, for the
            layout+OCR fallback path).
        method: Extraction strategy. One of:

            - ``"camelot"`` (default): use camelot-py (lattice, then stream).
              Falls back to layout+OCR if camelot finds nothing.
            - ``"tabula"``: use tabula-py instead of camelot.
            - ``"layout"``: skip camelot/tabula and go straight to the
              layout+OCR fallback (useful for scanned/image-only documents).

    Returns:
        A list of :class:`~docintel_kit.types.Table`, using the same
        representation as :func:`docintel_kit.spreadsheet.parse_spreadsheet`
        so downstream code (CSV/DataFrame export, LLM table QA, RAG over
        tables) is source-agnostic.

    Raises:
        ValueError: if ``method`` is not recognized, or if ``method`` requires
            a PDF but a non-PDF, non-image input was given.
    """
    source_path = input if isinstance(input, str) else None
    document_id = _make_document_id(input, source_path)

    if method not in {"camelot", "tabula", "layout"}:
        raise ValueError(f"Unknown method '{method}'. Expected one of: camelot, tabula, layout")

    if method == "layout":
        return _extract_with_layout_and_ocr(input, document_id)

    if not _is_pdf(input):
        raise ValueError(
            f"method='{method}' requires a PDF input. Use method='layout' for images, "
            "or docintel_kit.spreadsheet.parse_spreadsheet for Excel/CSV files."
        )

    tables = (
        _extract_with_camelot(input, document_id)
        if method == "camelot"
        else _extract_with_tabula(input, document_id)
    )
    if tables:
        return tables

    # No native tables found (likely a scanned PDF) — fall back to layout+OCR.
    return _extract_with_layout_and_ocr(input, document_id)
