# docintel_kit

**[Live overview page →](https://kanizmadix.github.io/docintel-kit/)**

A local-first Python toolkit for unified document processing and analysis: parsing,
OCR, layout analysis, table extraction, spreadsheet ingestion, structured field
extraction, LLM tasks (summarization / classification / table QA), and RAG /
semantic search — all behind one consistent set of shared data types.

No SaaS calls are required by default. Everything runs on your own machine except
the LLM-backed functions (`summarize_document`, `classify_document`, `qa_over_tables`),
which are optional and only call out to a network engine (e.g. Claude) when you
explicitly ask for one.

## Why

Most document-AI stacks bolt together five or six single-purpose libraries with
incompatible data models. docintel_kit gives you one `Table`, one `Document`, one
`ParseResult` — produced consistently whether the source was a PDF, a scanned image,
an Excel sheet, or a PowerPoint deck — so downstream code (an LLM prompt, a RAG
index, a CSV export) never needs to know where the data came from.

## Capabilities

| Capability | Function | Backend |
|---|---|---|
| Parsing (PDF/DOCX/PPTX/HTML/images) | `parse_document()` | pdfplumber, python-docx, python-pptx, BeautifulSoup |
| OCR | `run_ocr()` | Tesseract (pytesseract) |
| Layout analysis | `analyze_layout()` | layoutparser (Detectron2 / PubLayNet)* |
| Table extraction (PDF/scanned) | `extract_tables_from_document()` | camelot, with layout+OCR fallback |
| Spreadsheet/CSV parsing | `parse_spreadsheet()` | pandas |
| Structured field extraction | `extract_fields()` | Hugging Face `impira/layoutlm-document-qa` |
| Summarization / classification / table QA | `summarize_document()`, `classify_document()`, `qa_over_tables()` | Claude (pluggable) |
| RAG / semantic search | `index_documents()`, `search_documents()` | sentence-transformers + in-memory vector store |

Every capability is behind a `Base*Backend` abstract class, so you can register your
own implementation (a different OCR engine, a fine-tuned extraction model, a
persistent vector store) without touching call sites — see `register_*_backend()`
in each module.

\* See [Known limitations](#known-limitations) — layoutparser's default model
hosting is currently broken upstream.

## Install

Requires Python 3.11.

```bash
pip install -e .
```

System binaries needed by some capabilities (not pip-installable):

- **Tesseract OCR** — required for `run_ocr()`. `brew install tesseract` (macOS) or see the [Tesseract docs](https://github.com/tesseract-ocr/tesseract).
- **Ghostscript** — required for `extract_tables_from_document()` (camelot). `brew install ghostscript`.
- **Poppler** — required for PDF-to-image rasterization (`pdf2image`, used by OCR/layout/extraction on PDFs). `brew install poppler`.
- **Java (JRE)** — only needed for the optional `method="tabula"` path in table extraction; the default `camelot` path does not need it.

## Quick start

```python
from docintel_kit import parse_document, run_ocr, extract_tables_from_document, extract_fields

# Parse a text-layer PDF, DOCX, PPTX, or HTML file
result = parse_document("invoice.pdf")
print(result.get_page_text(0))

# OCR a scanned document or photo
ocr_result = run_ocr("scanned_invoice.pdf")
print(ocr_result.full_text)

# Extract tables (PDF ruled tables via camelot, with OCR fallback for scans)
tables = extract_tables_from_document("invoice.pdf")
df = tables[0].to_dataframe()

# Ask natural-language questions about a form/invoice's fields
result = extract_fields("invoice.pdf", schema={
    "invoice_number": {"type": "string"},
    "total_amount": {"type": "amount", "question": "What is the total amount due?"},
})
print(result.get_field("invoice_number").value)
```

## Testing

A high-difficulty end-to-end test suite lives in `testcases/`, covering every
supported file format, unicode/CJK/RTL text, corrupted/truncated files, and both
success and error paths — with real model inference (Tesseract OCR, camelot,
LayoutLM-based QA, sentence-transformers embeddings), not mocks, wherever the
capability doesn't require a paid API key.

```bash
pip install pytest reportlab
pytest testcases/ -v
```

As of the last full run: **117 passed, 2 skipped**. Both skips are documented,
environment-level limitations (see below), not docintel_kit bugs.

## Known limitations

- **`analyze_layout()`'s default backend is currently broken upstream.**
  layoutparser's built-in PubLayNet model zoo downloads its Detectron2 weights
  from Dropbox share links that now return "File Deleted" pages instead of the
  actual files (a long-standing, unresolved issue —
  see [Layout-Parser/layout-parser#168](https://github.com/Layout-Parser/layout-parser/issues/168)).
  `analyze_layout()` will raise a clear `RuntimeError` explaining this rather
  than an obscure YAML parsing error. Workarounds: supply your own local
  Detectron2 weights via `LayoutParserBackend(weights_path=...)`, or register
  an alternative `BaseLayoutBackend`. Table extraction's `method="layout"`
  fallback path is affected by the same issue when camelot finds no tables.
- **`extract_fields()`'s table-typed fields** route through the same layout
  backend, so they're subject to the limitation above until a working layout
  backend is registered.
- **`method="tabula"`** in table extraction requires a local Java runtime,
  which is not bundled — install a JRE if you need this path instead of the
  default `camelot`.

## Design notes

- **Structured extraction uses a QA model, not a generic pretrained checkpoint.**
  An earlier version used `microsoft/layoutlmv3-base` directly, which turned out
  to have no fine-tuned classification head (its output labels are generic
  placeholders), making it structurally incapable of answering arbitrary field
  questions. `extract_fields()` now uses `impira/layoutlm-document-qa` (a
  checkpoint fine-tuned for document question-answering) and converts each
  scalar schema field into a natural-language question, which is what makes
  zero-shot extraction against arbitrary schemas actually work.
- All backends are swappable. See `register_parser_backend`,
  `register_ocr_backend`, `register_layout_backend`,
  `register_extraction_backend`, `register_llm_client`, and
  `register_vector_store`.

## Project layout

```
docintel_kit/
├── types.py         # shared Pydantic models: Document, Page, ParseResult,
│                     # OcrResult, LayoutResult, Table, ExtractionResult,
│                     # RagSearchResult, ...
├── parsing.py        # PDF / DOCX / PPTX / HTML / image parsing
├── ocr.py            # Tesseract-backed OCR
├── layout.py         # layoutparser-backed layout analysis
├── tables.py         # camelot / tabula / layout+OCR table extraction
├── spreadsheet.py     # pandas-backed Excel/CSV parsing
├── extraction.py      # LayoutLM document-QA structured field extraction
├── llm_tasks.py       # Claude-backed summarize/classify/table-QA
└── rag.py             # sentence-transformers embeddings + vector search
```
