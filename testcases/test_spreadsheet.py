"""High-difficulty tests for docintel_kit.spreadsheet.

Covers multi-sheet XLSX workbooks with NaN/None cells and mixed types, CSV
files with quoted commas/embedded quotes/unicode/ragged rows, TSV files, raw
bytes input, and error handling for unrecognized extensions. Also verifies
that spreadsheet-derived Tables are structurally identical (same shared
Table model, same export method behavior) to document-derived Tables from
docintel_kit.tables, since the spec requires them to be interchangeable.
"""

from __future__ import annotations

import math

import pytest

from docintel_kit.spreadsheet import parse_spreadsheet
from docintel_kit.types import Table


class TestXlsxParsing:
    def test_multiple_sheets_each_become_a_table(self, fixtures_dir):
        tables = parse_spreadsheet(str(fixtures_dir / "workbook.xlsx"))
        assert len(tables) == 2
        sheet_names = {t.sheet_name for t in tables}
        assert sheet_names == {"Sales", "Regions"}

    def test_sheet_headers_and_row_values(self, fixtures_dir):
        tables = parse_spreadsheet(str(fixtures_dir / "workbook.xlsx"))
        sales = next(t for t in tables if t.sheet_name == "Sales")
        assert sales.headers == ["Product", "Units Sold", "Revenue"]
        # Note: "Units Sold" is "120.0" not "120" because the column contains
        # a None value elsewhere, which forces pandas to upcast the whole
        # column to float64 — this is standard pandas dtype coercion (not a
        # docintel_kit behavior) and is expected here.
        assert sales.rows[0] == ["Widget", "120.0", "1200.5"]

    def test_none_and_nan_cells_become_empty_string(self, fixtures_dir):
        """The 4th row of the Sales sheet has Product=None, Units Sold=None
        (from the leading None in the list) at various positions, and
        Revenue=NaN. All must be coerced to "" per _dataframe_to_table's
        documented behavior, not the literal strings 'None' or 'nan'."""
        tables = parse_spreadsheet(str(fixtures_dir / "workbook.xlsx"))
        sales = next(t for t in tables if t.sheet_name == "Sales")
        last_row = sales.rows[-1]
        assert "nan" not in [v.lower() for v in last_row if v]
        assert "none" not in [v.lower() for v in last_row if v]
        # Specifically: Product was None -> "", Revenue was NaN -> ""
        assert last_row[0] == ""
        assert last_row[2] == ""

    def test_unicode_preserved_in_excel_cells(self, fixtures_dir):
        tables = parse_spreadsheet(str(fixtures_dir / "workbook.xlsx"))
        regions = next(t for t in tables if t.sheet_name == "Regions")
        notes_col_index = regions.headers.index("Notes")
        notes = [row[notes_col_index] for row in regions.rows]
        assert "café" in notes
        assert "naïve résumé" in notes
        assert "日本語ノート" in notes

    def test_negative_and_float_values_preserved(self, fixtures_dir):
        tables = parse_spreadsheet(str(fixtures_dir / "workbook.xlsx"))
        regions = next(t for t in tables if t.sheet_name == "Regions")
        growth_index = regions.headers.index("Growth %")
        growth_values = [row[growth_index] for row in regions.rows]
        assert "-3.5" in growth_values

    def test_xlsx_bytes_input_matches_path_input(self, fixtures_dir):
        path = fixtures_dir / "workbook.xlsx"
        from_path = parse_spreadsheet(str(path))
        from_bytes = parse_spreadsheet(path.read_bytes())
        assert len(from_path) == len(from_bytes)
        from_path_sorted = sorted(from_path, key=lambda t: t.sheet_name or "")
        from_bytes_sorted = sorted(from_bytes, key=lambda t: t.sheet_name or "")
        for a, b in zip(from_path_sorted, from_bytes_sorted):
            assert a.headers == b.headers
            assert a.rows == b.rows

    def test_source_field_is_spreadsheet(self, fixtures_dir):
        tables = parse_spreadsheet(str(fixtures_dir / "workbook.xlsx"))
        assert all(t.source == "spreadsheet" for t in tables)
        assert all(t.confidence == 1.0 for t in tables)


class TestCsvParsing:
    def test_quoted_commas_and_embedded_quotes_parsed_correctly(self, fixtures_dir):
        tables = parse_spreadsheet(str(fixtures_dir / "tricky.csv"))
        assert len(tables) == 1
        table = tables[0]
        assert table.headers == ["name", "description", "amount"]
        # "Smith, John" must stay as one field, not split on the embedded comma.
        first_row = table.rows[0]
        assert first_row[0] == "Smith, John"
        assert 'quoted' in first_row[1]
        assert "," in first_row[1]  # the embedded comma survived within the field

    def test_unicode_in_csv_preserved(self, fixtures_dir):
        tables = parse_spreadsheet(str(fixtures_dir / "tricky.csv"))
        names = [row[0] for row in tables[0].rows]
        assert "Café Owner" in names

    def test_ragged_row_does_not_crash_and_pads_or_errors_predictably(self, fixtures_dir):
        """The fixture's last row has only 2 of 3 fields. pandas pads missing
        trailing fields with NaN by default; verify this doesn't crash and
        produces a well-defined (empty-string-coerced) result rather than
        misaligning columns."""
        tables = parse_spreadsheet(str(fixtures_dir / "tricky.csv"))
        table = tables[0]
        ragged_row = next(r for r in table.rows if r[0] == "Ragged Row")
        assert len(ragged_row) == 3
        assert ragged_row[2] == ""  # missing trailing amount -> NaN -> ""

    def test_csv_bytes_input(self, fixtures_dir):
        path = fixtures_dir / "tricky.csv"
        result = parse_spreadsheet(path.read_bytes())
        assert result[0].headers == ["name", "description", "amount"]


class TestTsvParsing:
    def test_tsv_parsed_with_tab_delimiter(self, fixtures_dir):
        tables = parse_spreadsheet(str(fixtures_dir / "tricky.tsv"))
        assert len(tables) == 1
        table = tables[0]
        assert table.headers == ["name", "amount"]
        assert ["Alice", "100"] in table.rows

    def test_tsv_unicode_preserved(self, fixtures_dir):
        tables = parse_spreadsheet(str(fixtures_dir / "tricky.tsv"))
        names = [row[0] for row in tables[0].rows]
        assert "Böb" in names


class TestErrorHandling:
    def test_unrecognized_extension_raises_value_error(self, fixtures_dir):
        with pytest.raises(ValueError):
            parse_spreadsheet(str(fixtures_dir / "unknown.xyz"))

    def test_bytes_without_excel_magic_number_defaults_to_csv_parsing(self, fixtures_dir):
        """Raw bytes that don't match the XLSX/XLS magic numbers are assumed
        to be delimited text per _infer_kind's documented fallback. Feeding
        it genuine CSV bytes should succeed; this exercises that fallback
        path explicitly (as opposed to the extension-based path)."""
        csv_bytes = b"a,b,c\n1,2,3\n"
        tables = parse_spreadsheet(csv_bytes)
        assert tables[0].headers == ["a", "b", "c"]


class TestCrossModuleTableConsistency:
    """Validates that Table objects produced by spreadsheet.py are drop-in
    compatible with the same Table model/export methods used by tables.py,
    per the spec's requirement that downstream code not care about origin.
    """

    def test_table_type_is_shared_model(self, fixtures_dir):
        tables = parse_spreadsheet(str(fixtures_dir / "workbook.xlsx"))
        assert all(isinstance(t, Table) for t in tables)

    def test_export_methods_work_identically_on_spreadsheet_tables(self, fixtures_dir):
        tables = parse_spreadsheet(str(fixtures_dir / "workbook.xlsx"))
        sales = next(t for t in tables if t.sheet_name == "Sales")

        df = sales.to_dataframe()
        assert list(df.columns) == ["Product", "Units Sold", "Revenue"]
        assert df.shape == (4, 3)

        csv_text = sales.to_csv()
        assert "Widget" in csv_text
        assert "Product,Units Sold,Revenue" in csv_text

        json_text = sales.to_json()
        assert "Gizmo" in json_text
