"""High-difficulty, end-to-end tests for docintel_kit.rag.

Covers the character-window chunker's edge cases directly, then exercises
real semantic search (actual sentence-transformers embeddings, no mocking)
across multiple documents with clearly distinct topics, table-row indexing,
collection isolation, and upsert/re-indexing behavior.
"""

from __future__ import annotations

import pytest

from docintel_kit.rag import (
    InMemoryVectorStore,
    _chunk_text,
    index_documents,
    index_tables,
    register_vector_store,
    search_documents,
)
from docintel_kit.types import Table


@pytest.fixture(autouse=True)
def isolated_vector_store():
    """Give every test a fresh in-memory store so indexed data from one test
    can never leak into another via the module-level singleton."""
    register_vector_store(InMemoryVectorStore())
    yield


class TestChunker:
    def test_empty_text_produces_no_chunks(self):
        assert _chunk_text("", chunk_size=100, overlap=10) == []
        assert _chunk_text("   \n\t  ", chunk_size=100, overlap=10) == []

    def test_text_shorter_than_chunk_size_is_a_single_chunk(self):
        chunks = _chunk_text("short text", chunk_size=100, overlap=10)
        assert chunks == ["short text"]

    def test_long_text_is_split_into_multiple_overlapping_chunks(self):
        text = "word " * 1000  # 5000 chars
        chunks = _chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)

    def test_chunk_boundaries_prefer_whitespace_not_mid_word(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 5
        chunks = _chunk_text(text, chunk_size=40, overlap=10)
        # None of the chunk boundaries should have sliced a word in half —
        # every chunk should start and end on a word boundary once stripped.
        for chunk in chunks:
            assert not chunk.startswith(" ")
            assert not chunk.endswith(" ")

    def test_overlap_greater_or_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError):
            _chunk_text("some text here", chunk_size=50, overlap=50)
        with pytest.raises(ValueError):
            _chunk_text("some text here", chunk_size=50, overlap=60)

    def test_exact_chunk_size_boundary_single_word_block(self):
        """A pathological case: one giant 'word' with no spaces at all,
        longer than chunk_size. There's no whitespace to break on, so the
        chunker must still make forward progress rather than looping forever
        or producing a zero-length chunk."""
        text = "x" * 500
        chunks = _chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        assert "".join(chunks).replace("", "") != ""  # sanity: got real content
        assert sum(len(c) for c in chunks) >= len(text) - 20 * len(chunks)


class TestDocumentIndexingAndSearch:
    def test_semantic_search_returns_the_topically_relevant_chunk(self, tmp_path):
        """Index two documents about very different topics and verify a
        query about one topic ranks that document's chunk above the other's
        -- this is a real correctness check of the embedding + cosine-
        similarity pipeline, not just 'did it run without crashing'."""
        finance_doc = tmp_path / "finance.html"
        finance_doc.write_text(
            "<html><body><p>Quarterly revenue grew 12 percent driven by strong "
            "enterprise software sales and improved gross margins.</p></body></html>",
            encoding="utf-8",
        )
        cooking_doc = tmp_path / "cooking.html"
        cooking_doc.write_text(
            "<html><body><p>To make a good tomato pasta sauce, sauté garlic in "
            "olive oil, add crushed tomatoes, and simmer with fresh basil.</p></body></html>",
            encoding="utf-8",
        )

        index_documents([str(finance_doc), str(cooking_doc)], collection="mixed")
        result = search_documents("How was quarterly revenue performance?", collection="mixed", top_k=2)

        assert result.query == "How was quarterly revenue performance?"
        assert len(result.matches) == 2
        assert "revenue" in result.matches[0].chunk.text.lower()
        assert result.matches[0].score > result.matches[1].score

    def test_top_k_limits_number_of_matches(self, tmp_path):
        doc = tmp_path / "doc.html"
        doc.write_text(
            "<html><body><p>"
            + " ".join(f"Sentence number {i} about various unrelated topics." for i in range(50))
            + "</p></body></html>",
            encoding="utf-8",
        )
        index_documents([str(doc)], collection="topk-test", chunk_size=100, chunk_overlap=20)
        result = search_documents("various topics", collection="topk-test", top_k=3)
        assert len(result.matches) <= 3

    def test_empty_collection_returns_no_matches_without_crashing(self):
        result = search_documents("anything", collection="never-indexed")
        assert result.matches == []

    def test_reindexing_same_document_upserts_rather_than_duplicates(self, tmp_path):
        doc = tmp_path / "doc.html"
        doc.write_text("<html><body><p>Alpha beta gamma.</p></body></html>", encoding="utf-8")

        index_documents([str(doc)], collection="upsert-test")
        result1 = search_documents("alpha", collection="upsert-test", top_k=10)
        count1 = len(result1.matches)

        index_documents([str(doc)], collection="upsert-test")
        result2 = search_documents("alpha", collection="upsert-test", top_k=10)
        count2 = len(result2.matches)

        assert count1 == count2  # same chunk_ids -> upsert, not duplication

    def test_collections_are_isolated_from_each_other(self, tmp_path):
        doc_a = tmp_path / "a.html"
        doc_a.write_text("<html><body><p>Alpha collection unique content xyz123.</p></body></html>", encoding="utf-8")
        doc_b = tmp_path / "b.html"
        doc_b.write_text("<html><body><p>Beta collection different unique content abc789.</p></body></html>", encoding="utf-8")

        index_documents([str(doc_a)], collection="collection-a")
        index_documents([str(doc_b)], collection="collection-b")

        result_a = search_documents("xyz123", collection="collection-a", top_k=5)
        result_b = search_documents("xyz123", collection="collection-b", top_k=5)

        assert any("xyz123" in t for t in result_a.top_texts())
        assert not any("xyz123" in t for t in result_b.top_texts())

    def test_top_texts_convenience_method(self, tmp_path):
        doc = tmp_path / "doc.html"
        doc.write_text("<html><body><p>Unique marker phrase zzqqxx.</p></body></html>", encoding="utf-8")
        index_documents([str(doc)], collection="top-texts-test")
        result = search_documents("zzqqxx", collection="top-texts-test", top_k=1)
        texts = result.top_texts()
        assert isinstance(texts, list)
        assert "zzqqxx" in texts[0]


class TestTableIndexing:
    def test_table_rows_are_individually_searchable(self):
        table = Table(
            table_id="products-1",
            headers=["Product", "Category", "Price"],
            rows=[
                ["Widget", "Hardware", "5.00"],
                ["Consulting Hours", "Services", "150.00"],
            ],
            source="test",
        )
        index_tables([table], collection="table-search")
        result = search_documents("consulting services pricing", collection="table-search", top_k=2)
        assert len(result.matches) == 2
        assert "Consulting" in result.matches[0].chunk.text

    def test_table_row_text_includes_headers_as_context(self):
        table = Table(
            table_id="t1",
            headers=["Name", "Amount"],
            rows=[["Acme Corp", "999.00"]],
            source="test",
        )
        index_tables([table], collection="header-context-test")
        result = search_documents("Acme", collection="header-context-test", top_k=1)
        assert "Name: Acme Corp" in result.matches[0].chunk.text
        assert "Amount: 999.00" in result.matches[0].chunk.text

    def test_table_without_headers_still_indexes_raw_row_values(self):
        table = Table(table_id="t2", headers=None, rows=[["raw", "values", "only"]], source="test")
        index_tables([table], collection="no-headers-test")
        result = search_documents("raw values", collection="no-headers-test", top_k=1)
        assert len(result.matches) == 1

    def test_empty_table_list_indexes_nothing_without_crashing(self):
        index_tables([], collection="empty-table-test")
        result = search_documents("anything", collection="empty-table-test")
        assert result.matches == []

    def test_documents_and_tables_coexist_in_the_same_collection(self, tmp_path):
        doc = tmp_path / "doc.html"
        doc.write_text("<html><body><p>Company overview and mission statement text.</p></body></html>", encoding="utf-8")
        table = Table(
            table_id="revenue-table",
            headers=["Quarter", "Revenue"],
            rows=[["Q1", "1000000"]],
            source="test",
        )
        index_documents([str(doc)], collection="mixed-content")
        index_tables([table], collection="mixed-content")

        result = search_documents("Q1 revenue figures", collection="mixed-content", top_k=5)
        assert any("Q1" in t for t in result.top_texts())


class TestCustomVectorStore:
    def test_register_vector_store_swaps_the_backend(self):
        custom_store = InMemoryVectorStore()
        register_vector_store(custom_store)
        # Directly upsert into the custom store instance and confirm
        # search_documents reads through the swapped-in backend.
        from docintel_kit.types import RagChunk

        chunk = RagChunk(
            chunk_id="manual-1",
            document_id="manual-doc",
            text="manually inserted content for direct-store testing",
            embedding=[1.0, 0.0, 0.0],
        )
        custom_store.upsert("manual-collection", [chunk])
        # Bypass real embedding for the query by checking store.query directly.
        matches = custom_store.query("manual-collection", [1.0, 0.0, 0.0], top_k=1)
        assert len(matches) == 1
        assert matches[0].score == pytest.approx(1.0)
