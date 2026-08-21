# doc-intel-kit

[![CI](https://github.com/kanizmadix/docintel-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/kanizmadix/docintel-kit/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-34d399.svg)](LICENSE)

**[Try the no-key interactive showcase](https://kanizmadix.github.io/docintel-kit/)**

`doc-intel-kit` is a composable Python 3.11 toolkit for turning documents into shared, typed representations. It connects text-layer parsing, Tesseract OCR, PDF table extraction, spreadsheet ingestion, schema-driven field extraction, optional Claude tasks, and small-scale semantic search without hiding the underlying backends.

The distribution name is `doc-intel-kit`; the Python import remains `docintel_kit`.

## Why use it?

Document workflows often glue together libraries that disagree about pages, coordinates, tables, and metadata. This project provides common Pydantic models such as `Document`, `ParseResult`, `OcrResult`, and `Table`, while keeping every capability independently callable and exposing registration hooks for parser, OCR, layout, extraction, LLM, and vector-store backends. That makes it useful for prototypes, evaluations, and as a typed integration layer—not as a claim of one-click, production-grade document automation.

> **Why this toolkit?** It gives developers a composable, local-first Python integration layer with shared typed models across parsing, OCR, tables, extraction, optional LLM tasks, and small-scale RAG—without requiring a cloud document platform or hiding each backend behind one automatic pipeline.

**Best for:** teams combining local document libraries, projects that need inspectable intermediate results, privacy-sensitive prototypes, custom backend evaluations, and applications that want one typed contract while retaining control of routing, persistence, quality thresholds, and infrastructure.

## How it compares

This is a scope comparison, not a claim that the projects are interchangeable or that `doc-intel-kit` is universally better.

| Alternative | Stronger when you need | Why use `doc-intel-kit` instead |
|---|---|---|
| [Docling](https://docling-project.github.io/docling/) | Mature multi-format conversion, advanced PDF understanding, a unified document model, rich exports, CLI/server options, and broad ecosystem integrations. | You want smaller independently callable Python capabilities and shared Pydantic contracts while explicitly choosing parsing, OCR, tables, extraction, LLM, and RAG branches. |
| [Unstructured](https://docs.unstructured.io/open-source/core-functionality/overview) | A broad partition/chunk/clean/stage pipeline plus ingestion and destination connectors for production data preparation. | You need a compact SDK for direct local composition and typed document/table/OCR results rather than a connector-oriented ingestion platform. |
| [LlamaParse](https://developers.llamaindex.ai/python/cloud/llamaparse/) | Hosted agentic OCR, complex layout/chart parsing, many formats, structured extraction, indexing, and production document-agent services. | You prefer local execution for core parsing/OCR/table paths, no required service account for those paths, and control over each underlying backend. |
| [Amazon Textract](https://docs.aws.amazon.com/textract/latest/dg/what-is.html) | Managed extraction of text, handwriting, forms, tables, queries, signatures, confidence, and geometry at AWS scale. | You want cloud-neutral Python components, local Tesseract/Camelot paths, and a shared model that can accept custom backends without uploading core documents to AWS. |
| [Azure Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/) | Managed prebuilt/custom document models, key-value/table/layout extraction, cloud operations, and Azure integration. | You want an infrastructure-neutral integration layer and direct ownership of local execution, routing, storage, and model choices. |
| [Google Document AI](https://docs.cloud.google.com/document-ai/docs) | Managed processors that transform documents into structured data with Google Cloud operations and specialized processors. | You need a lightweight Python toolkit for explicit local branches and prototypes that are not committed to a cloud document-processing platform. |

**Not a replacement for:** production ingestion connectors, managed autoscaling, human-review queues, mature layout models, enterprise governance, persistent vector infrastructure, or service-level guarantees. Docling and Unstructured are broader open-source systems; LlamaParse and hyperscaler services are more complete managed offerings. `doc-intel-kit` is intentionally a transparent integration toolkit, and its current layout, first-page extraction, and in-memory RAG limitations are documented below.

> Comparison content is paraphrased from the linked official documentation for clarity and licensing compliance. Capabilities and commercial offerings evolve; verify current vendor documentation before choosing a production platform.

## Capability status

| Capability | API | Status | What is actually provided |
|---|---|---:|---|
| PDF, DOCX, PPTX, HTML parsing | `parse_document()` | Supported | Native text extraction into logical pages. Image inputs return an empty page plus an OCR warning. |
| Image/scanned-PDF OCR | `run_ocr()` | Supported | Tesseract word text, confidence, and pixel-space boxes. |
| `.xlsx`, `.xlsm`, CSV, TSV ingestion | `parse_spreadsheet()` | Supported | One shared `Table` per sheet/file. Legacy `.xls` is **not** supported. |
| Native PDF tables | `extract_tables_from_document()` | Supported with caveats | Camelot lattice/stream; optional Tabula path. Scanned fallback depends on experimental layout. |
| Layout detection | `analyze_layout()` | Experimental / currently blocked by upstream model hosting | PubLayNet Text, Title, List, Table, and Figure regions when compatible local weights are supplied. |
| Structured scalar extraction | `extract_fields()` | Experimental | LayoutLM document QA over OCR words; scalar answers currently use only page 1. |
| Semantic search / RAG | `index_documents()`, `search_documents()` | Experimental | Sentence-transformer embeddings and a process-local, in-memory vector store. |
| LLM tasks | `summarize_document()`, `classify_document()`, `qa_over_tables()` | Optional | Claude is the only built-in network client. Custom providers require implementing and registering `BaseLlmClient`. |

EasyOCR and PaddleOCR backends are not implemented. The registry interfaces make alternative backends possible, but installing another OCR library alone does not register it.

## Architecture and real execution flow

The library exposes building blocks; your application chooses and orchestrates the branches. It does **not** automatically run every stage end to end.

```text
Input path / bytes
       |
       +--> parse_document() ------> ParseResult --------+--> your application
       |                                                 |
       +--> run_ocr() -------------> OcrResult           +--> index_documents()
       |        (images/scans)                           |       |
       +--> parse_spreadsheet() ----> list[Table] --------+       v
       |                                                 |   in-memory vectors
       +--> extract_tables_from_document() -> list[Table]+       |
       |                                                         v
       +--> extract_fields() -> ExtractionResult             search_documents()

Parsed text or Tables --explicit call--> Claude task --network + API key--> result
```

Parsing does not automatically OCR empty pages. OCR does not automatically feed extraction. RAG indexing parses native text and does not OCR scans. Compose those steps explicitly when your workflow needs them.

## Installation

Python **3.11** is required. The initial release intentionally does not claim Python 3.12+ support.

```bash
pip install doc-intel-kit
```

For the optional Java-backed Tabula path:

```bash
pip install "doc-intel-kit[tabula]"
```

For a source checkout:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

### System dependencies

Some Python packages wrap local executables that pip cannot install:

- **Tesseract OCR**: required by `run_ocr()` and the default extraction path.
- **Poppler**: required to rasterize PDF pages for OCR, layout, and structured extraction.
- **Ghostscript**: required by Camelot's lattice table path on common installations.
- **Java JRE**: required only for `method="tabula"`.
- **Detectron2 plus compatible local weights**: required by the default LayoutParser backend; Detectron2 is not distributed by this package.

On macOS, the common binaries are available with `brew install tesseract poppler ghostscript`. On Debian/Ubuntu, use `sudo apt-get install tesseract-ocr poppler-utils ghostscript`.

## Quickstart

```python
from docintel_kit import parse_document
from docintel_kit.spreadsheet import parse_spreadsheet

# Fast native-text path: no API key and no model download.
parsed = parse_document("report.pdf")
print(parsed.get_page_text(0))

# Spreadsheet ingestion uses the same Table model as PDF extraction.
tables = parse_spreadsheet("forecast.xlsx")
print(tables[0].headers)
print(tables[0].to_dataframe().head())
```

OCR and table extraction are explicit branches:

```python
from docintel_kit import run_ocr, extract_tables_from_document

ocr = run_ocr("scanned-invoice.pdf")
print(ocr.full_text)

pdf_tables = extract_tables_from_document("text-layer-invoice.pdf")
if pdf_tables:
    print(pdf_tables[0].to_csv())
```

Small in-memory RAG is also explicit:

```python
from docintel_kit import index_documents, search_documents

index_documents(["report.pdf"], collection="demo")
hits = search_documents("What changed in revenue?", collection="demo", top_k=3)
for hit in hits.matches:
    print(hit.score, hit.chunk.text)
```

## Keys, network use, and model downloads

- Importing the package, native parsing, spreadsheet ingestion, Tesseract OCR, and Camelot extraction do not require an API key.
- Calling a built-in Claude task without `ANTHROPIC_API_KEY` raises before an HTTP request is made. Set the variable only when you intentionally use those functions; never commit keys.
- `extract_fields()` lazily downloads `impira/layoutlm-document-qa` from Hugging Face on first use unless it is already cached.
- RAG lazily downloads `sentence-transformers/all-MiniLM-L6-v2` on first embedding request unless cached.
- The default layout backend attempts to obtain PubLayNet artifacts, but the upstream links are currently unavailable. Use trusted local config/weights or register a different backend.

Consequently, the toolkit is local-execution-oriented but does **not** promise a no-network default for every capability. Cache model artifacts ahead of time for controlled or offline environments and verify their licenses and provenance.

## Known limitations

- **Layout outage:** LayoutParser's built-in PubLayNet model-zoo URLs currently resolve to deleted Dropbox resources ([upstream issue #168](https://github.com/Layout-Parser/layout-parser/issues/168)). `analyze_layout()` raises an actionable error. This also affects the scanned-table layout+OCR fallback and table-typed fields in `extract_fields()`.
- **First-page scalar extraction:** `extract_fields()` rasterizes the document but currently supplies page index 0 to the scalar QA model. Fields that exist only on later pages will not be found.
- **In-memory RAG:** the default vector store is process-local, brute-force, and nonpersistent. Reuse the same embedding model at index and query time. It is intended for prototypes and small corpora.
- **Scans are not automatically OCRed for RAG:** `index_documents()` uses `parse_document()` text. Build an OCR-to-index path in your application for image-only content.
- **Table fallback is heuristic:** layout+OCR groups nearby words; it is not robust cell reconstruction. Camelot exceptions may result in fallback behavior.
- **No legacy `.xls`:** use `.xlsx`, `.xlsm`, CSV, or TSV.
- **No automatic pipeline:** each function is independently invoked; applications own routing, retries, caching, persistence, and quality thresholds.

## Testing and release checks

The repository contains deterministic unit/format tests plus environment-heavy integration tests that use Tesseract, Camelot, LayoutLM, and sentence-transformers. Counts vary with installed binaries, model cache state, and the known layout outage, so this README does not freeze a pass count.

```bash
python -m pytest testcases -v
python -m build
python -m twine check dist/*
```

CI runs Python 3.11 deterministic tests and package build/twine checks. Model-backed integration tests (extraction, layout, RAG), two OCR text-accuracy assertions, and the Java-backed Tabula test are excluded from CI because their results depend on local binaries, fonts, model caches, and OCR engine versions. Run the full suite locally to exercise them.

## License

MIT © Kanishk S. See [LICENSE](LICENSE).
