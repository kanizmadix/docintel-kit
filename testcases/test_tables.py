"""High-difficulty tests for docintel_kit.tables.

Covers the camelot (default) extraction path against a real ruled-table PDF,
the layout+OCR fallback for scanned/table-less PDFs, and error handling for
non-PDF inputs with method="camelot"/"tabula".

Note on method="tabula": tabula-py requires a local Java runtime, which is
NOT installed in this environment (verified: `java -version` reports no JRE
found). Those tests are skipped with a clear reason rather than silently
omitted, so absence is visible rather than assumed.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from docintel_kit.tables import extract_tables_from_document


def _java_available() -> bool:
    try:
        subprocess.run(
            ["java", "-version"], capture_output=True, timeout=5, check=True
        )
        return True
    except Exception:
        return False


class TestCamelotExtraction:
    def test_extracts_ruled_table_with_correct_headers_and_rows(self, fixtures_dir):
        tables = extract_tables_from_document(str(fixtures_dir / "invoice_table.pdf"))
        assert len(tables) == 1
        table = tables[0]
        assert table.source == "camelot-lattice"
        assert table.headers == ["Item", "Qty", "Unit Price", "Total"]
        assert table.rows == [
            ["Widget A", "12", "$5.00", "$60.00"],
            ["Widget B", "3", "$19.99", "$59.97"],
            ["Gizmo", "1", "$250.00", "$250.00"],
        ]

    def test_table_confidence_is_high_for_clean_lattice_table(self, fixtures_dir):
        tables = extract_tables_from_document(str(fixtures_dir / "invoice_table.pdf"))
        assert tables[0].confidence > 0.9

    def test_table_export_methods_round_trip(self, fixtures_dir):
        tables = extract_tables_from_document(str(fixtures_dir / "invoice_table.pdf"))
        table = tables[0]

        csv_text = table.to_csv()
        assert "Widget A" in csv_text
        assert "Item,Qty,Unit Price,Total" in csv_text

        df = table.to_dataframe()
        assert list(df.columns) == ["Item", "Qty", "Unit Price", "Total"]
        assert df.shape == (3, 4)

        json_text = table.to_json()
        assert "Gizmo" in json_text

    def test_bytes_input_matches_path_input(self, fixtures_dir):
        path = fixtures_dir / "invoice_table.pdf"
        from_path = extract_tables_from_document(str(path))
        from_bytes = extract_tables_from_document(path.read_bytes())
        assert from_path[0].rows == from_bytes[0].rows

    def test_table_less_pdf_falls_back_to_layout_and_returns_empty_or_list(self, fixtures_dir):
        """text_report.pdf has no tables at all. Camelot should find nothing,
        triggering the layout+OCR fallback — which itself requires a working
        layout backend. Since the default layout backend is broken upstream
        (see test_layout.py), this should raise the same actionable
        RuntimeError rather than crash unpredictably, OR return an empty list
        if the fallback path tolerates the failure. We assert one of these
        two well-defined behaviors, not a silent wrong answer.
        """
        try:
            tables = extract_tables_from_document(str(fixtures_dir / "text_report.pdf"))
            assert tables == []
        except RuntimeError as e:
            assert "Dropbox" in str(e) or "model zoo" in str(e).lower()

    def test_non_pdf_input_with_camelot_method_raises_value_error(self, fixtures_dir):
        with pytest.raises(ValueError):
            extract_tables_from_document(str(fixtures_dir / "form.png"), method="camelot")

    def test_unknown_method_raises_value_error(self, fixtures_dir):
        with pytest.raises(ValueError):
            extract_tables_from_document(str(fixtures_dir / "invoice_table.pdf"), method="bogus")


class TestLayoutOcrFallbackMethod:
    def test_layout_method_on_image_requires_broken_layout_backend(self, fixtures_dir):
        """method="layout" skips camelot entirely and goes straight to layout
        analysis. Since the default layout backend is broken upstream (dead
        PubLayNet model hosting — see test_layout.py), this must surface that
        same actionable error rather than crash with an unrelated traceback.
        """
        with pytest.raises(RuntimeError) as exc_info:
            extract_tables_from_document(str(fixtures_dir / "form.png"), method="layout")
        assert "Dropbox" in str(exc_info.value) or "model zoo" in str(exc_info.value).lower()


@pytest.mark.skipif(
    not _java_available(),
    reason="Java runtime not installed; required by tabula-py (method='tabula').",
)
class TestTabulaExtraction:
    def test_extracts_ruled_table_via_tabula(self, fixtures_dir):
        tables = extract_tables_from_document(
            str(fixtures_dir / "invoice_table.pdf"), method="tabula"
        )
        assert len(tables) >= 1
        assert any("Widget" in "".join(row) for row in tables[0].rows)
