"""LLM-backed tasks: summarization, classification, and table question-answering.

Unlike every other module in this package, these functions make outbound
network calls by default (to the Anthropic Claude API) and are the only place
in docintel_kit that requires an API key. Everything else in the library is
fully local.

Design:

- :class:`BaseLlmClient` is a minimal abstraction (`complete(prompt) -> str`)
  so any provider can be plugged in.
- :class:`ClaudeClient` is the reference implementation, wrapping the Claude
  Messages API over HTTP via `httpx`. It reads its API key from the
  ``ANTHROPIC_API_KEY`` environment variable by default — never hardcode keys.
- Prompt templates for summarization, classification, and table QA are kept
  deliberately simple and ask the model for a single, easy-to-parse output
  (plain text for summaries, a compact ``label: score`` list for
  classification), so response parsing stays robust across model versions.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Optional, Union

from .parsing import parse_document
from .types import Table

__all__ = [
    "BaseLlmClient",
    "ClaudeClient",
    "summarize_document",
    "classify_document",
    "qa_over_tables",
]

_DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-latest"
_CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
_CLAUDE_API_VERSION = "2023-06-01"


class BaseLlmClient(ABC):
    """Minimal interface an LLM client must implement to be used by this module."""

    name: str = "base"

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """Send ``prompt`` to the model and return its raw text response."""
        raise NotImplementedError


class ClaudeClient(BaseLlmClient):
    """`BaseLlmClient` implementation backed by the Anthropic Claude API.

    Requires the ``ANTHROPIC_API_KEY`` environment variable (or an explicit
    ``api_key``) to actually call the network. The HTTP call itself is
    isolated in :meth:`complete` so it can be monkeypatched/stubbed in tests
    without needing real credentials.
    """

    name = "claude"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_CLAUDE_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """Call the Claude Messages API with a single user-turn prompt.

        Raises:
            RuntimeError: if no API key is configured.
            httpx.HTTPStatusError: if the API returns a non-2xx response.
        """
        if not self.api_key:
            raise RuntimeError(
                "No Anthropic API key configured. Set the ANTHROPIC_API_KEY "
                "environment variable, or pass api_key=... to ClaudeClient()."
            )

        import httpx

        response = httpx.post(
            _CLAUDE_API_URL,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": _CLAUDE_API_VERSION,
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        # The Messages API returns a list of content blocks; concatenate any
        # text blocks in order.
        return "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        ).strip()


_CLIENTS: dict[str, BaseLlmClient] = {}


def register_llm_client(client: BaseLlmClient) -> None:
    """Register a custom :class:`BaseLlmClient` under its ``name``."""
    _CLIENTS[client.name] = client


def _get_client(engine: str) -> BaseLlmClient:
    if engine not in _CLIENTS:
        if engine == "claude":
            _CLIENTS["claude"] = ClaudeClient()
        else:
            raise KeyError(
                f"Unknown LLM engine '{engine}'. Registered engines: "
                f"{sorted(set(_CLIENTS) | {'claude'})}. Register a custom "
                "BaseLlmClient via register_llm_client() first."
            )
    return _CLIENTS[engine]


def _resolve_text(input_: Union[str, bytes]) -> str:
    """Extract plain text from a document input, reusing parse_document."""
    parse_result = parse_document(input_)
    text = parse_result.full_text
    if not text.strip():
        raise ValueError(
            "No extractable text found in the input. If this is a scanned "
            "document or image, run docintel_kit.ocr.run_ocr() first and pass "
            "its text instead."
        )
    return text


# Prompt templates. Kept simple and explicit about the desired output format
# so responses are easy to parse deterministically across model versions.

_SUMMARIZE_PROMPT = """You are summarizing a document for a busy professional reader.

Write a concise summary in plain prose (no headers, no bullet points, 3-6 sentences).
Do not include any preamble like "Here is a summary" — output only the summary itself.

Document:
\"\"\"
{text}
\"\"\"
"""

_CLASSIFY_PROMPT = """Classify the document below against the following labels: {labels}.

Respond with exactly one line per label, in the exact format:
label: score

where score is a number between 0 and 1 representing how well the document matches
that label. Output nothing else — no preamble, no explanation, no extra formatting.

Document:
\"\"\"
{text}
\"\"\"
"""

_TABLE_QA_PROMPT = """You are answering a question using only the table data below.
Tables are given in CSV format. If the answer cannot be determined from the
tables, say so explicitly instead of guessing.

Answer in 1-3 sentences of plain prose. Output only the answer.

Question: {question}

Tables:
{tables_csv}
"""


def summarize_document(input: Union[str, bytes], engine: str = "claude") -> str:
    """Summarize a document using an LLM.

    Args:
        input: A filesystem path or raw bytes for a document with a native
            text layer (PDF, DOCX, PPTX, HTML). For scanned/image documents,
            run :func:`docintel_kit.ocr.run_ocr` first.
        engine: Name of a registered :class:`BaseLlmClient`. Defaults to
            ``"claude"``.

    Returns:
        A short plain-text summary (typically 3-6 sentences).

    Raises:
        ValueError: if no text could be extracted from ``input``.
        RuntimeError: if the selected engine has no API key configured.
    """
    text = _resolve_text(input)
    client = _get_client(engine)
    prompt = _SUMMARIZE_PROMPT.format(text=text)
    return client.complete(prompt).strip()


def classify_document(
    input: Union[str, bytes], labels: list[str], engine: str = "claude"
) -> dict[str, float]:
    """Classify a document against a set of candidate labels using an LLM.

    Args:
        input: A filesystem path or raw bytes for a document with a native
            text layer.
        labels: Candidate labels to score the document against.
        engine: Name of a registered :class:`BaseLlmClient`. Defaults to
            ``"claude"``.

    Returns:
        A mapping from each label in ``labels`` to a confidence score in
        ``[0, 1]``. Labels the model's response doesn't mention default to 0.0.

    Raises:
        ValueError: if no text could be extracted from ``input``, or
            ``labels`` is empty.
        RuntimeError: if the selected engine has no API key configured.
    """
    if not labels:
        raise ValueError("labels must be a non-empty list.")

    text = _resolve_text(input)
    client = _get_client(engine)
    prompt = _CLASSIFY_PROMPT.format(labels=", ".join(labels), text=text)
    response = client.complete(prompt)

    scores: dict[str, float] = {label: 0.0 for label in labels}
    # Parse "label: score" lines, matching case-insensitively against the
    # requested labels so minor casing differences from the model don't drop
    # a match.
    label_lookup = {label.lower(): label for label in labels}
    for line in response.splitlines():
        match = re.match(r"^\s*(.+?)\s*:\s*([0-9]*\.?[0-9]+)\s*$", line)
        if not match:
            continue
        raw_label, raw_score = match.groups()
        canonical = label_lookup.get(raw_label.strip().lower())
        if canonical is None:
            continue
        try:
            score = float(raw_score)
        except ValueError:
            continue
        scores[canonical] = max(0.0, min(1.0, score))
    return scores


def qa_over_tables(tables: list[Table], question: str, engine: str = "claude") -> str:
    """Answer a natural-language question over one or more tables using an LLM.

    Args:
        tables: Tables to answer over — from :func:`docintel_kit.tables.extract_tables_from_document`,
            :func:`docintel_kit.spreadsheet.parse_spreadsheet`, or any other
            source producing the shared :class:`~docintel_kit.types.Table` model.
        question: A natural-language question about the table data.
        engine: Name of a registered :class:`BaseLlmClient`. Defaults to
            ``"claude"``.

    Returns:
        A short plain-text answer grounded in the given tables.

    Raises:
        ValueError: if ``tables`` is empty.
        RuntimeError: if the selected engine has no API key configured.
    """
    if not tables:
        raise ValueError("tables must be a non-empty list.")

    tables_csv = "\n\n".join(
        f"Table {i} ({table.table_id}):\n{table.to_csv()}" for i, table in enumerate(tables)
    )
    client = _get_client(engine)
    prompt = _TABLE_QA_PROMPT.format(question=question, tables_csv=tables_csv)
    return client.complete(prompt).strip()
