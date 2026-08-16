"""High-difficulty tests for docintel_kit.llm_tasks.

Per the user's instruction, real Claude API calls are explicitly OUT OF SCOPE
for this test run and are tested later, separately. Instead, this file:

1. Validates the prompt-building and response-parsing logic (the actual
   "business logic" of this module) using a fully deterministic, in-process
   StubLlmClient -- no network calls, no API key required.
2. Validates ClaudeClient's own request-construction and error paths using
   monkeypatched HTTP (still no real network access), so the Anthropic
   integration code itself is exercised without hitting the network.
3. Explicitly verifies that ClaudeClient refuses to run without an API key,
   which is a security-relevant behavior worth locking down with a test.
"""

from __future__ import annotations

import pytest

from docintel_kit.llm_tasks import (
    BaseLlmClient,
    ClaudeClient,
    classify_document,
    qa_over_tables,
    register_llm_client,
    summarize_document,
)
from docintel_kit.types import Table


class StubLlmClient(BaseLlmClient):
    """Deterministic, no-network stand-in for a real LLM, used to validate
    docintel_kit's own prompt construction and response parsing logic in
    isolation from any actual model's behavior."""

    name = "stub"

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.last_prompt: str | None = None

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        self.last_prompt = prompt
        return self.response


@pytest.fixture
def stub_client():
    client = StubLlmClient()
    register_llm_client(client)
    return client


class TestSummarizeDocument:
    def test_returns_stub_response_stripped(self, fixtures_dir, stub_client):
        stub_client.response = "  This document discusses quarterly results.  \n"
        summary = summarize_document(str(fixtures_dir / "text_report.pdf"), engine="stub")
        assert summary == "This document discusses quarterly results."

    def test_prompt_includes_full_document_text(self, fixtures_dir, stub_client):
        summarize_document(str(fixtures_dir / "text_report.pdf"), engine="stub")
        assert "Quarterly revenue increased" in stub_client.last_prompt
        assert "café" in stub_client.last_prompt  # unicode preserved into the prompt

    def test_html_and_docx_inputs_also_work(self, fixtures_dir, stub_client):
        stub_client.response = "summary"
        assert summarize_document(str(fixtures_dir / "malformed.html"), engine="stub") == "summary"
        assert summarize_document(str(fixtures_dir / "contract.docx"), engine="stub") == "summary"

    def test_textless_image_input_raises_value_error(self, fixtures_dir, stub_client):
        """An image has no native text layer; summarize_document must fail
        clearly (pointing at OCR) rather than silently summarizing nothing."""
        with pytest.raises(ValueError, match="run_ocr"):
            summarize_document(str(fixtures_dir / "form.png"), engine="stub")

    def test_unknown_engine_raises_key_error(self, fixtures_dir):
        with pytest.raises(KeyError):
            summarize_document(str(fixtures_dir / "text_report.pdf"), engine="not-a-real-engine")


class TestClassifyDocument:
    def test_parses_well_formed_label_score_lines(self, fixtures_dir, stub_client):
        stub_client.response = "invoice: 0.9\ncontract: 0.05\nresume: 0.0"
        scores = classify_document(
            str(fixtures_dir / "text_report.pdf"),
            labels=["invoice", "contract", "resume"],
            engine="stub",
        )
        assert scores == {"invoice": 0.9, "contract": 0.05, "resume": 0.0}

    def test_missing_labels_in_response_default_to_zero(self, fixtures_dir, stub_client):
        stub_client.response = "invoice: 0.9"
        scores = classify_document(
            str(fixtures_dir / "text_report.pdf"),
            labels=["invoice", "contract"],
            engine="stub",
        )
        assert scores == {"invoice": 0.9, "contract": 0.0}

    def test_case_insensitive_label_matching(self, fixtures_dir, stub_client):
        stub_client.response = "INVOICE: 0.8\nContract: 0.2"
        scores = classify_document(
            str(fixtures_dir / "text_report.pdf"),
            labels=["invoice", "contract"],
            engine="stub",
        )
        assert scores == {"invoice": 0.8, "contract": 0.2}

    def test_scores_are_clamped_to_0_1_range(self, fixtures_dir, stub_client):
        """A misbehaving model might return an out-of-range score (e.g. 1.5
        or -0.3); the parser must clamp rather than propagate garbage."""
        stub_client.response = "invoice: 1.5\ncontract: -0.3"
        scores = classify_document(
            str(fixtures_dir / "text_report.pdf"),
            labels=["invoice", "contract"],
            engine="stub",
        )
        assert scores == {"invoice": 1.0, "contract": 0.0}

    def test_malformed_response_lines_are_ignored_not_crashed_on(self, fixtures_dir, stub_client):
        stub_client.response = (
            "This is a preamble sentence the model wasn't supposed to add.\n"
            "invoice: 0.7\n"
            "not a valid line at all\n"
            "contract:notanumber\n"
        )
        scores = classify_document(
            str(fixtures_dir / "text_report.pdf"),
            labels=["invoice", "contract"],
            engine="stub",
        )
        assert scores["invoice"] == 0.7
        assert scores["contract"] == 0.0  # malformed line ignored, default kept

    def test_empty_labels_list_raises_value_error(self, fixtures_dir, stub_client):
        with pytest.raises(ValueError):
            classify_document(str(fixtures_dir / "text_report.pdf"), labels=[], engine="stub")

    def test_prompt_lists_all_requested_labels(self, fixtures_dir, stub_client):
        stub_client.response = "a: 1.0\nb: 0.0\nc: 0.0"
        classify_document(str(fixtures_dir / "text_report.pdf"), labels=["a", "b", "c"], engine="stub")
        assert "a, b, c" in stub_client.last_prompt


class TestQaOverTables:
    def test_returns_stub_answer_stripped(self, stub_client):
        stub_client.response = "  The total revenue was $310.00.  "
        table = Table(table_id="t1", headers=["Item", "Total"], rows=[["Widget", "310.00"]], source="test")
        answer = qa_over_tables([table], "What is the total revenue?", engine="stub")
        assert answer == "The total revenue was $310.00."

    def test_prompt_includes_csv_rendering_of_all_tables(self, stub_client):
        table1 = Table(table_id="t1", headers=["A", "B"], rows=[["1", "2"]], source="test")
        table2 = Table(table_id="t2", headers=["X", "Y"], rows=[["9", "8"]], source="test")
        qa_over_tables([table1, table2], "some question", engine="stub")
        assert "t1" in stub_client.last_prompt
        assert "t2" in stub_client.last_prompt
        assert "1,2" in stub_client.last_prompt or "1,2".replace(",", ", ") in stub_client.last_prompt
        assert "9,8" in stub_client.last_prompt or "9,8".replace(",", ", ") in stub_client.last_prompt

    def test_prompt_includes_the_question_verbatim(self, stub_client):
        table = Table(table_id="t1", headers=["A"], rows=[["1"]], source="test")
        qa_over_tables([table], "What is the meaning of row 1?", engine="stub")
        assert "What is the meaning of row 1?" in stub_client.last_prompt

    def test_empty_tables_list_raises_value_error(self, stub_client):
        with pytest.raises(ValueError):
            qa_over_tables([], "any question", engine="stub")


class TestClaudeClientWithoutNetwork:
    """Exercises ClaudeClient's own logic (API key handling, request
    construction, response parsing) without making real network calls, by
    monkeypatching httpx.post. Real, live Claude API calls are explicitly out
    of scope for this test run per the user's instructions.
    """

    def test_missing_api_key_raises_runtime_error_before_any_network_call(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = ClaudeClient(api_key=None)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            client.complete("does not matter, should never be sent")

    def test_explicit_api_key_overrides_environment(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-should-be-ignored")
        client = ClaudeClient(api_key="explicit-key")
        assert client.api_key == "explicit-key"

    def test_complete_parses_text_blocks_from_response(self, monkeypatch):
        client = ClaudeClient(api_key="fake-key-for-mocked-request")

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "tool_use", "text": "should be skipped"},
                        {"type": "text", "text": "world."},
                    ]
                }

        def fake_post(url, headers=None, json=None, timeout=None):
            assert url == "https://api.anthropic.com/v1/messages"
            assert headers["x-api-key"] == "fake-key-for-mocked-request"
            assert json["messages"][0]["content"] == "test prompt"
            return FakeResponse()

        import httpx

        monkeypatch.setattr(httpx, "post", fake_post)
        result = client.complete("test prompt")
        assert result == "Hello world."

    def test_complete_propagates_http_errors(self, monkeypatch):
        client = ClaudeClient(api_key="fake-key")

        class FakeErrorResponse:
            def raise_for_status(self):
                import httpx

                raise httpx.HTTPStatusError("bad request", request=None, response=None)

        def fake_post(*args, **kwargs):
            return FakeErrorResponse()

        import httpx

        monkeypatch.setattr(httpx, "post", fake_post)
        with pytest.raises(httpx.HTTPStatusError):
            client.complete("test prompt")
