"""Structured field extraction from forms, invoices, and IDs.

Provides :func:`extract_fields`, which performs schema-driven extraction of
scalar fields (dates, amounts, names, ...) and table fields (e.g. invoice line
items) using Hugging Face `transformers` Document AI models.

The default backend (:class:`LayoutLmExtractionBackend`) uses
``impira/layoutlm-document-qa`` via transformers' built-in
``document-question-answering`` pipeline. This model has an actual
fine-tuned question-answering head (unlike the generic
``microsoft/layoutlmv3-base`` checkpoint, which has no classification head
and cannot answer arbitrary field questions out of the box). Each scalar
schema field is turned into a natural-language question (e.g. "What is the
invoice number?") and answered against the document image, which is what
lets this work zero-shot against arbitrary schemas without fine-tuning.

An OCR-free Donut-based backend can be added later by implementing
:class:`BaseExtractionBackend` and registering it via
:func:`register_extraction_backend`, without changing this module's public API.

Internally, this module composes the other capabilities in the package:
:func:`docintel_kit.ocr.run_ocr` supplies word-level OCR (fed to the QA
pipeline as precomputed ``word_boxes`` so OCR is only ever run once), and
:func:`docintel_kit.tables.extract_tables_from_document` handles table-typed
schema fields.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from .types import BoundingBox, Document, ExtractedField, ExtractionResult, Table

__all__ = ["BaseExtractionBackend", "LayoutLmExtractionBackend", "extract_fields"]


def _make_document_id(input_: Union[str, bytes], source_path: Optional[str]) -> str:
    if source_path:
        return source_path
    data = input_ if isinstance(input_, bytes) else input_.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:32]


def _load_images(input_: Union[str, bytes]):
    from PIL import Image

    if isinstance(input_, str):
        path = Path(input_)
        if path.suffix.lower() == ".pdf":
            from pdf2image import convert_from_path

            return convert_from_path(str(path))
        return [Image.open(path)]

    if input_.startswith(b"%PDF"):
        from pdf2image import convert_from_bytes

        return convert_from_bytes(input_)

    import io

    return [Image.open(io.BytesIO(input_))]


class BaseExtractionBackend(ABC):
    """Interface for a structured field extraction backend.

    Implementations receive the raw input plus a JSON-schema-like field
    specification and must return values with confidences for each requested
    field, splitting scalar fields from table fields.
    """

    name: str = "base"

    @abstractmethod
    def extract(
        self,
        input_: Union[str, bytes],
        schema: dict[str, Any],
        document: Document,
    ) -> ExtractionResult:
        raise NotImplementedError


def _word_boxes_for_page(input_: Union[str, bytes], page_index: int, image) -> list:
    """Build the ``word_boxes`` list the document-question-answering pipeline
    expects: ``[(word, [x0, y0, x1, y1]), ...]`` with pixel-space boxes for
    a single page's image.

    Reuses :func:`docintel_kit.ocr.run_ocr` so OCR runs once per document via
    the same Tesseract backend used elsewhere in the package, rather than
    letting the transformers pipeline re-run its own internal OCR pass.
    """
    from .ocr import run_ocr

    ocr_result = run_ocr(input_)
    word_boxes = []
    for word in ocr_result.words_for_page(page_index):
        box = [
            int(word.bbox.x0),
            int(word.bbox.y0),
            int(word.bbox.x1),
            int(word.bbox.y1),
        ]
        word_boxes.append((word.text, box))
    return word_boxes


def _field_to_question(field_name: str, spec: dict[str, Any]) -> str:
    """Turn a schema field name/spec into a natural-language question for
    the document-question-answering pipeline.

    Uses an explicit ``spec["question"]`` if provided (for full control),
    otherwise derives a reasonable question from the field name and type
    (e.g. ``invoice_date`` with ``type: "date"`` -> "What is the invoice
    date?"), which is what makes this backend usable zero-shot against
    arbitrary schemas without per-field configuration.
    """
    if "question" in spec:
        return spec["question"]
    readable = field_name.replace("_", " ").replace("-", " ").strip()
    return f"What is the {readable}?"


class LayoutLmExtractionBackend(BaseExtractionBackend):
    """Default extraction backend using Hugging Face's document-question-answering
    pipeline with the ``impira/layoutlm-document-qa`` checkpoint.

    Unlike a generic pretrained LayoutLM checkpoint (which has no
    classification head and cannot answer arbitrary field questions without
    task-specific fine-tuning), this model was fine-tuned specifically for
    document QA, so it can answer natural-language questions about a
    document's scalar fields out of the box. Each requested scalar field is
    converted into a question (see :func:`_field_to_question`) and answered
    independently; the pipeline's own answer confidence becomes the field's
    confidence.

    For production use on a specific document type, fine-tune LayoutLMv3 (or
    another Document AI model) on your own labeled data and register a
    custom :class:`BaseExtractionBackend` via
    :func:`register_extraction_backend` for higher accuracy than a zero-shot
    QA model can offer.
    """

    name = "layoutlm"

    def __init__(self, model_name: str = "impira/layoutlm-document-qa") -> None:
        self.model_name = model_name
        self._qa_pipeline = None

    def _get_pipeline(self):
        if self._qa_pipeline is None:
            from transformers import pipeline

            self._qa_pipeline = pipeline("document-question-answering", model=self.model_name)
        return self._qa_pipeline

    def extract(
        self,
        input_: Union[str, bytes],
        schema: dict[str, Any],
        document: Document,
    ) -> ExtractionResult:
        images = _load_images(input_)

        fields: dict[str, ExtractedField] = {}
        tables: dict[str, list[Table]] = {}
        warnings: list[str] = []

        field_specs = {
            name: spec for name, spec in schema.items() if spec.get("type") != "table"
        }
        table_specs = {name: spec for name, spec in schema.items() if spec.get("type") == "table"}

        if field_specs:
            word_boxes = _word_boxes_for_page(input_, page_index=0, image=images[0])
            if not word_boxes:
                warnings.append("No OCR words were recognized; cannot extract fields.")
            else:
                qa_pipeline = self._get_pipeline()
                for field_name, spec in field_specs.items():
                    question = _field_to_question(field_name, spec)
                    try:
                        answers = qa_pipeline(
                            image=images[0], question=question, word_boxes=word_boxes
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        warnings.append(f"Question answering failed for '{field_name}': {exc}")
                        fields[field_name] = ExtractedField(name=field_name, value=None, confidence=0.0)
                        continue
                    best = answers[0] if isinstance(answers, list) else answers
                    value = best.get("answer")
                    confidence = float(best.get("score", 0.0))
                    fields[field_name] = ExtractedField(
                        name=field_name,
                        value=value,
                        confidence=confidence,
                        page_index=0,
                    )
                    if value is None or confidence < 0.3:
                        warnings.append(
                            f"Field '{field_name}' was answered with low confidence "
                            f"({confidence:.2f}); result may be unreliable."
                        )

        if table_specs:
            from .tables import extract_tables_from_document

            try:
                detected_tables = extract_tables_from_document(input_, method="layout")
            except Exception as exc:
                detected_tables = []
                warnings.append(f"Table extraction failed: {exc}")
            for table_name in table_specs:
                tables[table_name] = detected_tables

        return ExtractionResult(
            document=document,
            backend=self.name,
            fields=fields,
            tables=tables,
            raw_schema=schema,
            warnings=warnings,
        )


_BACKENDS: dict[str, BaseExtractionBackend] = {}


def register_extraction_backend(backend: BaseExtractionBackend) -> None:
    """Register a custom :class:`BaseExtractionBackend` under its ``name``."""
    _BACKENDS[backend.name] = backend


def extract_fields(
    input: Union[str, bytes],
    schema: dict[str, Any],
    backend: str = "layoutlm",
) -> ExtractionResult:
    """Extract structured fields from a form, invoice, or ID document.

    Args:
        input: A filesystem path or raw bytes for a PDF or image.
        schema: A mapping from field name to a spec dict. Scalar fields use
            ``{"type": "string" | "date" | "amount"}``; table fields use
            ``{"type": "table", "columns": [...]}`` (columns are advisory —
            detected tables are returned as-is via
            :func:`docintel_kit.tables.extract_tables_from_document`). Scalar
            fields may optionally include ``"question"`` to override the
            auto-generated natural-language question sent to the QA model
            (e.g. for a field name that doesn't read naturally, such as
            ``{"type": "string", "question": "Who is the bill-to party?"}``).

            Example::

                schema = {
                    "invoice_number": {"type": "string"},
                    "invoice_date": {"type": "date"},
                    "total_amount": {"type": "amount", "question": "What is the total amount due?"},
                    "line_items": {"type": "table", "columns": ["description", "qty", "price"]},
                }

        backend: Name of a registered :class:`BaseExtractionBackend`.
            Defaults to ``"layoutlm"``.

    Returns:
        An :class:`ExtractionResult` with scalar ``fields`` (value +
        confidence) and any table-typed schema entries in ``tables``.

    Raises:
        KeyError: if ``backend`` is not registered.
    """
    if backend == "layoutlm":
        if "layoutlm" not in _BACKENDS:
            _BACKENDS["layoutlm"] = LayoutLmExtractionBackend()
        selected = _BACKENDS["layoutlm"]
    elif backend in _BACKENDS:
        selected = _BACKENDS[backend]
    else:
        raise KeyError(
            f"Unknown extraction backend '{backend}'. Registered backends: "
            f"{sorted(set(_BACKENDS) | {'layoutlm'})}"
        )

    source_path = input if isinstance(input, str) else None
    document = Document(
        id=_make_document_id(input, source_path),
        source_path=source_path,
    )
    return selected.extract(input, schema, document)
