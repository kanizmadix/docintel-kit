"""High-difficulty tests for docintel_kit.ocr.

Exercises the default Tesseract backend against clean form images, scanned
(image-only) PDFs, rotated text, blank images, and corrupted image files.
Requires the Tesseract binary to be installed on the system (verified once
at module scope via a skip-guard).
"""

from __future__ import annotations

import shutil

import pytest

from docintel_kit.ocr import run_ocr

pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="Tesseract binary not found on PATH; required for OCR tests.",
)


class TestImageOcr:
    def test_recognizes_form_fields(self, fixtures_dir):
        result = run_ocr(str(fixtures_dir / "form.png"))
        assert result.backend == "tesseract"
        text = result.full_text
        assert "PURCHASE" in text.upper()
        assert "PO-778812" in text or "778812" in text
        assert "1,250.75" in text or "1250.75" in text

    def test_words_have_valid_bounding_boxes_and_confidence(self, fixtures_dir):
        result = run_ocr(str(fixtures_dir / "form.png"))
        assert len(result.words) > 0
        for word in result.words:
            assert 0.0 <= word.confidence <= 1.0
            assert word.bbox.x1 > word.bbox.x0
            assert word.bbox.y1 > word.bbox.y0
            assert word.bbox.page_index == 0

    def test_bytes_input_matches_path_input(self, fixtures_dir):
        path = fixtures_dir / "form.png"
        from_path = run_ocr(str(path))
        from_bytes = run_ocr(path.read_bytes())
        # OCR isn't guaranteed byte-perfect deterministic across calls in
        # theory, but tesseract is deterministic in practice for a fixed
        # image; require the recognized text to match exactly.
        assert from_path.full_text == from_bytes.full_text

    def test_rotated_text_still_recognized_reasonably(self, fixtures_dir):
        """A ~7deg rotation is a realistic 'scanned on an angle' case. We
        don't require perfect accuracy, but Tesseract should recover at
        least some of the known words."""
        result = run_ocr(str(fixtures_dir / "form_rotated.png"))
        text = result.full_text.upper()
        assert "PURCHASE" in text or "ORDER" in text

    def test_blank_image_returns_no_words_without_crashing(self, fixtures_dir):
        result = run_ocr(str(fixtures_dir / "blank.png"))
        assert result.words == []
        assert result.get_page_text(0) == ""

    def test_corrupted_image_raises_cleanly(self, fixtures_dir):
        with pytest.raises(Exception):
            run_ocr(str(fixtures_dir / "corrupted.png"))

    def test_unknown_backend_raises_key_error(self, fixtures_dir):
        with pytest.raises(KeyError):
            run_ocr(str(fixtures_dir / "form.png"), backend="nonexistent-ocr-backend")

    def test_words_for_page_filters_correctly(self, fixtures_dir):
        result = run_ocr(str(fixtures_dir / "form.png"))
        page0_words = result.words_for_page(0)
        assert page0_words == result.words  # single-page image
        assert result.words_for_page(5) == []


class TestScannedPdfOcr:
    def test_scanned_pdf_rasterizes_and_recognizes_text(self, fixtures_dir):
        result = run_ocr(str(fixtures_dir / "scanned_invoice.pdf"))
        text = result.full_text.upper()
        assert "SCANNED" in text or "INVOICE" in text
        assert "90210" in text or "INV" in text

    def test_scanned_pdf_bytes_input(self, fixtures_dir):
        path = fixtures_dir / "scanned_invoice.pdf"
        result = run_ocr(path.read_bytes())
        assert len(result.words) > 0
        assert result.document.page_count == 1
