"""OCR for image-only PDFs, scans, and photos.

Provides :func:`run_ocr`, a backend-agnostic entry point for text recognition.
The default backend wraps `pytesseract <https://github.com/madmaz23/pytesseract>`_
(a Python binding for Tesseract OCR, which must be installed separately as a
system binary). Additional backends (EasyOCR, PaddleOCR) can be added later by
implementing :class:`BaseOcrBackend` and registering them.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from .types import BoundingBox, Document, OcrResult, OcrWord

__all__ = ["BaseOcrBackend", "TesseractOcrBackend", "run_ocr", "register_ocr_backend"]


def _make_document_id(input_: Union[str, bytes], source_path: Optional[str]) -> str:
    if source_path:
        return source_path
    data = input_ if isinstance(input_, bytes) else input_.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:32]


def _load_images(input_: Union[str, bytes]):
    """Return a list of PIL Images, one per page, for a path or raw bytes.

    Supports image files directly, and PDFs by rasterizing each page via
    `pdf2image` (which requires the `poppler` system binary).
    """
    from PIL import Image

    if isinstance(input_, str):
        path = Path(input_)
        if path.suffix.lower() == ".pdf":
            from pdf2image import convert_from_path

            return convert_from_path(str(path))
        return [Image.open(path)]

    # Raw bytes: sniff PDF magic number, else treat as an image.
    if input_.startswith(b"%PDF"):
        from pdf2image import convert_from_bytes

        return convert_from_bytes(input_)

    import io

    return [Image.open(io.BytesIO(input_))]


class BaseOcrBackend(ABC):
    """Interface for an OCR backend.

    Implementations receive a list of page images and must return word-level
    recognitions with bounding boxes, which :func:`run_ocr` assembles into an
    :class:`OcrResult`.
    """

    name: str = "base"

    @abstractmethod
    def recognize(self, images: list, document: Document) -> OcrResult:
        """Run OCR over ``images`` (one PIL Image per page) and return an OcrResult."""
        raise NotImplementedError


class TesseractOcrBackend(BaseOcrBackend):
    """Default OCR backend using pytesseract (Tesseract OCR).

    Requires the Tesseract binary to be installed and available on PATH.
    See https://github.com/tesseract-ocr/tesseract for installation instructions.
    """

    name = "tesseract"

    def __init__(self, lang: str = "eng") -> None:
        self.lang = lang

    def recognize(self, images: list, document: Document) -> OcrResult:
        import pytesseract
        from pytesseract import Output

        words: list[OcrWord] = []
        page_texts: dict[int, str] = {}

        for page_index, image in enumerate(images):
            data = pytesseract.image_to_data(
                image, lang=self.lang, output_type=Output.DICT
            )
            line_texts: list[str] = []
            n = len(data.get("text", []))
            for i in range(n):
                text = data["text"][i].strip()
                if not text:
                    continue
                # Tesseract reports confidence as 0-100, or -1 for non-text blocks.
                raw_conf = float(data.get("conf", ["-1"] * n)[i])
                confidence = max(0.0, raw_conf) / 100.0
                left, top = float(data["left"][i]), float(data["top"][i])
                width, height = float(data["width"][i]), float(data["height"][i])
                words.append(
                    OcrWord(
                        text=text,
                        confidence=confidence,
                        bbox=BoundingBox(
                            x0=left,
                            y0=top,
                            x1=left + width,
                            y1=top + height,
                            page_index=page_index,
                        ),
                    )
                )
                line_texts.append(text)
            page_texts[page_index] = " ".join(line_texts)

        document.page_count = len(images)
        return OcrResult(
            document=document,
            words=words,
            backend=self.name,
            page_texts=page_texts,
        )


_BACKENDS: dict[str, BaseOcrBackend] = {"tesseract": TesseractOcrBackend()}


def register_ocr_backend(backend: BaseOcrBackend) -> None:
    """Register a custom :class:`BaseOcrBackend` under its ``name``.

    Use this to add e.g. EasyOCR or PaddleOCR implementations without
    modifying this module.
    """
    _BACKENDS[backend.name] = backend


def run_ocr(input: Union[str, bytes], backend: str = "tesseract") -> OcrResult:
    """Run OCR on an image or image-only PDF.

    Args:
        input: A filesystem path or raw bytes for an image (PNG/JPEG/TIFF/...)
            or a PDF. PDFs are rasterized page-by-page before recognition.
        backend: Name of a registered :class:`BaseOcrBackend`. Defaults to
            ``"tesseract"``.

    Returns:
        An :class:`OcrResult` with word-level recognitions, confidences, and
        bounding boxes, plus reconstructed per-page text.

    Raises:
        KeyError: if ``backend`` is not registered.
    """
    if backend not in _BACKENDS:
        raise KeyError(f"Unknown OCR backend '{backend}'. Registered backends: {sorted(_BACKENDS)}")

    source_path = input if isinstance(input, str) else None
    document = Document(
        id=_make_document_id(input, source_path),
        source_path=source_path,
    )
    images = _load_images(input)
    return _BACKENDS[backend].recognize(images, document)
