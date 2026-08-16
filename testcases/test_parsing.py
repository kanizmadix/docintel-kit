"""High-difficulty tests for docintel_kit.parsing.

Covers every format claimed in the spec (PDF, DOCX, PPTX, HTML, images),
both path-based and raw-bytes input, unicode content, multi-page documents,
empty pages, corrupted/truncated files, and unsupported formats.
"""

from __future__ import annotations

import pytest

from docintel_kit.parsing import parse_document


class TestPdfParsing:
    def test_multi_page_text_extraction(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "text_report.pdf"))
        assert result.backend == "default"
        assert len(result.pages) == 3
        for i, page in enumerate(result.pages):
            assert f"Page {i + 1}" in page.text

    def test_unicode_multilingual_text_preserved(self, fixtures_dir):
        """Covers Latin-extended, CJK, Greek, and Cyrillic scripts in a single
        text-layer PDF, rendered with a broad-coverage Unicode TTF so this
        tests docintel_kit's extraction, not the test fixture's font support."""
        result = parse_document(str(fixtures_dir / "text_report.pdf"))
        text = result.full_text
        assert "café" in text
        assert "naïve" in text
        assert "日本語" in text
        assert "Ελληνικά" in text
        assert "Русский" in text

    def test_bytes_input_matches_path_input(self, fixtures_dir):
        path = fixtures_dir / "text_report.pdf"
        from_path = parse_document(str(path))
        from_bytes = parse_document(path.read_bytes(), mime_type="application/pdf")
        assert from_path.full_text == from_bytes.full_text
        assert len(from_path.pages) == len(from_bytes.pages)

    def test_empty_page_produces_warning(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "empty_page.pdf"))
        assert len(result.pages) == 1
        assert result.pages[0].text.strip() == ""
        assert any("OCR" in w or "text" in w.lower() for w in result.warnings)

    def test_image_only_pdf_has_no_text_but_warns(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "scanned_invoice.pdf"))
        assert result.pages[0].text.strip() == ""
        assert result.warnings, "Expected a warning suggesting OCR for a scanned page"

    def test_corrupted_pdf_raises_cleanly(self, fixtures_dir):
        with pytest.raises(Exception):
            parse_document(str(fixtures_dir / "corrupted.pdf"))

    def test_truncated_pdf_raises_cleanly(self, fixtures_dir):
        with pytest.raises(Exception):
            parse_document(str(fixtures_dir / "truncated.pdf"))

    def test_page_geometry_populated(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "text_report.pdf"))
        for page in result.pages:
            assert page.width and page.width > 0
            assert page.height and page.height > 0

    def test_get_page_text_out_of_range_raises_index_error(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "text_report.pdf"))
        with pytest.raises(IndexError):
            result.get_page_text(99)

    def test_document_id_is_stable_source_path(self, fixtures_dir):
        path = str(fixtures_dir / "text_report.pdf")
        result = parse_document(path)
        assert result.document.id == path
        assert result.document.page_count == 3


class TestDocxParsing:
    def test_paragraphs_and_table_extracted(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "contract.docx"))
        text = result.full_text
        assert "Contract Agreement" in text
        assert "Party A" in text
        assert "Widget" in text and "Gadget" in text
        # Table cells joined with " | "
        assert "Item | Qty | Price" in text

    def test_unicode_in_docx(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "contract.docx"))
        assert "café" in result.full_text
        assert "日本語" in result.full_text

    def test_docx_bytes_input(self, fixtures_dir):
        path = fixtures_dir / "contract.docx"
        result = parse_document(
            path.read_bytes(),
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert "Contract Agreement" in result.full_text

    def test_docx_treated_as_single_page(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "contract.docx"))
        assert len(result.pages) == 1
        assert result.document.page_count == 1


class TestPptxParsing:
    def test_slide_text_and_table_extracted(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "slides.pptx"))
        assert len(result.pages) == 2
        assert "Q3 Results" in result.pages[0].text
        assert "Revenue grew" in result.pages[0].text
        assert "APAC" in result.pages[1].text
        assert "18%" in result.pages[1].text

    def test_pptx_unicode(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "slides.pptx"))
        assert "café" in result.full_text
        assert "日本語" in result.full_text

    def test_pptx_bytes_input(self, fixtures_dir):
        path = fixtures_dir / "slides.pptx"
        result = parse_document(
            path.read_bytes(),
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
        assert len(result.pages) == 2


class TestHtmlParsing:
    def test_malformed_html_still_parses(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "malformed.html"))
        text = result.full_text
        assert "Unclosed heading" in text
        assert "bold and" in text
        assert "Cell1" in text and "Cell2" in text

    def test_script_and_style_are_stripped(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "malformed.html"))
        text = result.full_text
        assert "alert(" not in text
        assert "color: red" not in text

    def test_html_unicode_and_emoji(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "malformed.html"))
        assert "café" in result.full_text
        assert "🚀" in result.full_text

    def test_html_from_raw_bytes(self, fixtures_dir):
        raw = (fixtures_dir / "malformed.html").read_bytes()
        result = parse_document(raw, mime_type="text/html")
        assert "Unclosed heading" in result.full_text

    def test_html_inline_string_not_a_path(self):
        """A raw HTML string passed as `input` (not a filesystem path) should
        still parse — this exercises the fallback branch in _parse_html where
        Path(input_) doesn't point to an existing file."""
        html_string = "<html><body><p>Inline HTML string, not a file</p></body></html>"
        result = parse_document(html_string, mime_type="text/html")
        assert "Inline HTML string" in result.full_text


class TestImageParsing:
    def test_image_has_no_text_layer_and_warns(self, fixtures_dir):
        result = parse_document(str(fixtures_dir / "form.png"))
        assert result.pages[0].text == ""
        assert any("OCR" in w for w in result.warnings)

    def test_image_bytes_input(self, fixtures_dir):
        raw = (fixtures_dir / "form.png").read_bytes()
        result = parse_document(raw, mime_type="image/png")
        assert result.pages[0].text == ""


class TestErrorHandling:
    def test_unsupported_extension_raises_value_error(self, fixtures_dir):
        with pytest.raises(ValueError):
            parse_document(str(fixtures_dir / "unknown.xyz"))

    def test_unknown_backend_raises_key_error(self, fixtures_dir):
        with pytest.raises(KeyError):
            parse_document(str(fixtures_dir / "text_report.pdf"), backend="nonexistent-backend")

    def test_bytes_without_mime_type_and_unrecognized_magic_raises(self):
        with pytest.raises(ValueError):
            parse_document(b"\x00\x01\x02\x03 random junk that matches no known signature")

    def test_explicit_mime_type_overrides_inference(self, fixtures_dir):
        """Passing an explicit (wrong) mime_type should be honored over
        extension-based inference, and fail predictably rather than silently
        misparsing."""
        with pytest.raises(Exception):
            parse_document(str(fixtures_dir / "text_report.pdf"), mime_type="text/html")
