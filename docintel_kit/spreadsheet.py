"""Excel / CSV parsing and table handling.

Loads spreadsheet and CSV data using pandas (`read_excel`, `read_csv`) and
returns the shared :class:`~docintel_kit.types.Table` model — the same
representation produced by :mod:`docintel_kit.tables` for PDF/image tables —
so downstream code (LLM table QA, RAG over tables, exports) never needs to
know whether a table originated from a spreadsheet or a document.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Optional, Union

from .types import Table

__all__ = ["parse_spreadsheet"]

_EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xls"}
_CSV_EXTENSIONS = {".csv", ".tsv"}


def _make_document_id(input_: Union[str, bytes], source_path: Optional[str]) -> str:
    if source_path:
        return source_path
    data = input_ if isinstance(input_, bytes) else input_.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:32]


def _infer_kind(input_: Union[str, bytes]) -> str:
    """Return "excel" or "csv" based on extension or content sniffing."""
    if isinstance(input_, str):
        suffix = Path(input_).suffix.lower()
        if suffix in _EXCEL_EXTENSIONS:
            return "excel"
        if suffix in _CSV_EXTENSIONS:
            return "csv"
        raise ValueError(
            f"Could not infer spreadsheet type from extension '{suffix}'. "
            "Expected one of: " + ", ".join(sorted(_EXCEL_EXTENSIONS | _CSV_EXTENSIONS))
        )

    # Raw bytes: XLSX/XLSM are zip archives (PK magic number); XLS is an OLE
    # compound file. Anything else is assumed to be delimited text.
    if input_.startswith(b"PK\x03\x04") or input_.startswith(b"\xd0\xcf\x11\xe0"):
        return "excel"
    return "csv"


def _dataframe_to_table(df, table_id: str, sheet_name: Optional[str], source: str) -> Table:
    headers = [str(c) for c in df.columns.tolist()]
    # Coerce every cell to str for a uniform Table representation; NaN -> "".
    rows = [
        ["" if v is None or (isinstance(v, float) and v != v) else str(v) for v in row]
        for row in df.values.tolist()
    ]
    return Table(
        table_id=table_id,
        headers=headers,
        rows=rows,
        sheet_name=sheet_name,
        source=source,
        confidence=1.0,
    )


def parse_spreadsheet(input: Union[str, bytes]) -> list[Table]:
    """Parse an Excel workbook or CSV/TSV file into a list of tables.

    Excel workbooks yield one :class:`~docintel_kit.types.Table` per sheet.
    CSV/TSV files yield a single table.

    Args:
        input: A filesystem path or raw bytes for an ``.xlsx``/``.xlsm``/``.xls``
            workbook, or a ``.csv``/``.tsv`` file.

    Returns:
        A list of :class:`~docintel_kit.types.Table`, using the same shared
        representation as :func:`docintel_kit.tables.extract_tables_from_document`.

    Raises:
        ValueError: if the spreadsheet type cannot be inferred from a path's
            extension (bytes input is sniffed automatically).
    """
    import pandas as pd

    source_path = input if isinstance(input, str) else None
    document_id = _make_document_id(input, source_path)
    kind = _infer_kind(input)

    if kind == "excel":
        source = io.BytesIO(input) if isinstance(input, bytes) else input
        # engine="openpyxl" covers .xlsx/.xlsm; pandas falls back to xlrd for
        # legacy .xls automatically if xlrd is installed, otherwise raises a
        # clear error from pandas itself.
        sheets = pd.read_excel(source, sheet_name=None, engine="openpyxl")
        tables: list[Table] = []
        for sheet_name, df in sheets.items():
            tables.append(
                _dataframe_to_table(
                    df,
                    table_id=f"{document_id}-{sheet_name}",
                    sheet_name=str(sheet_name),
                    source="spreadsheet",
                )
            )
        return tables

    # CSV/TSV
    source = io.BytesIO(input) if isinstance(input, bytes) else input
    sep = "\t" if (isinstance(input, str) and input.lower().endswith(".tsv")) else ","
    df = pd.read_csv(source, sep=sep)
    return [
        _dataframe_to_table(
            df, table_id=f"{document_id}-sheet0", sheet_name=None, source="spreadsheet"
        )
    ]
