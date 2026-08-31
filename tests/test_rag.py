"""
Unit Tests for the SEC RAG Subsystem (Phase 2).
Validates section parsing, multi-strategy chunking, hybrid retrieval,
citation verification, and explicit abstention.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.rag.ingest import build_curated_10k_corpus, ParsedFiling
from src.rag.chunk import FixedSizeChunker, SectionAwareChunker, DocumentChunk
from src.rag.retrieve import RAGRetriever, RetrievalResult
from src.rag.generate import GroundedGenerator, verify_citations_programmatically, ABSTENTION_PHRASE


@pytest.fixture
def sample_filings():
    """Fixture providing parsed filings."""
    return build_curated_10k_corpus()


def test_section_parser_identifies_items(sample_filings):
    """
    Test 1: Verifies that 10-K filings have intact section metadata (Item 1, 1A, 7, 8).
    """
    assert len(sample_filings) >= 10
    for filing in sample_filings:
        assert filing.ticker is not None
        assert filing.fiscal_year >= 2023
        assert "ITEM_1" in filing.sections
        assert "ITEM_1A" in filing.sections
        assert "ITEM_7" in filing.sections
        assert "ITEM_8" in filing.sections
        assert len(filing.sections["ITEM_7"]) > 50


def test_chunking_strategies_generate_valid_metadata(sample_filings):
    """
    Test 2: Verifies that both Fixed and Section-Aware chunking produce valid chunks
    with proper metadata and positive token counts.
    """
    sample = sample_filings[0]
    fixed_chunker = FixedSizeChunker(chunk_size=200, overlap=20)
    section_chunker = SectionAwareChunker(max_tokens=300, min_tokens=50)

    fixed_chunks = fixed_chunker.chunk_filing(sample)
    section_chunks = section_chunker.chunk_filing(sample)

    assert len(fixed_chunks) > 0
    assert len(section_chunks) > 0

    for c in section_chunks:
        assert c.ticker == sample.ticker
        assert c.fiscal_year == sample.fiscal_year
        assert c.section_name in sample.sections
        assert c.token_count > 0
        assert len(c.chunk_id) > 5


def test_hybrid_retrieval_executes_successfully():
    """
    Test 3: Verifies that hybrid retrieval (Dense + BM25) returns valid results
    ordered by rank for a financial query.
    """
    retriever = RAGRetriever(strategy="section_aware", use_hybrid=True, use_reranker=False, top_k=5)
    results = retriever.retrieve("What was NVIDIA's Data Center revenue in FY2024?")

    assert len(results) > 0
    assert len(results) <= 5
    assert results[0].rank == 1
    assert results[0].chunk.ticker == "NVDA"
    assert "NVDA" in results[0].chunk.chunk_id


def test_citation_verification_detects_hallucinations():
    """
    Test 4: Verifies that programmatic citation verification correctly approves valid
    retrieved chunk IDs and rejects hallucinated chunk IDs.
    """
    dummy_chunk = DocumentChunk(
        chunk_id="AAPL_10K_2023_ITEM_7_0001",
        doc_id="AAPL_10K_2023",
        ticker="AAPL",
        company_name="Apple Inc.",
        fiscal_year=2023,
        section_name="ITEM_7",
        section_title="MD&A",
        text="R&D expense was $29,915 million in FY2023.",
        strategy="section_aware",
        token_count=10,
    )
    retrieved_result = RetrievalResult(chunk=dummy_chunk, score=0.9, rank=1, retrieval_method="test")

    # Case A: Valid citation matching retrieved ID
    valid_text = "Apple spent $29,915 million on R&D in FY2023 [AAPL_10K_2023_ITEM_7_0001]."
    verif_valid = verify_citations_programmatically(valid_text, [retrieved_result])
    assert verif_valid.is_valid is True
    assert verif_valid.citation_precision == 1.0
    assert len(verif_valid.hallucinated_ids) == 0

    # Case B: Hallucinated citation NOT in retrieved set
    fake_text = "Apple spent $29,915 million on R&D in FY2023 [FAKE_10K_CHUNK_9999]."
    verif_fake = verify_citations_programmatically(fake_text, [retrieved_result])
    assert verif_fake.is_valid is False
    assert verif_fake.citation_precision == 0.0
    assert "FAKE_10K_CHUNK_9999" in verif_fake.hallucinated_ids


def test_abstention_on_unanswerable_question():
    """
    Test 5: Verifies that the GroundedGenerator correctly abstains when context is irrelevant.
    """
    generator = GroundedGenerator()
    dummy_chunk = DocumentChunk(
        chunk_id="TSLA_10K_2023_ITEM_1_0001",
        doc_id="TSLA_10K_2023",
        ticker="TSLA",
        company_name="Tesla, Inc.",
        fiscal_year=2023,
        section_name="ITEM_1",
        section_title="Business",
        text="Tesla manufactures electric vehicles and solar energy products.",
        strategy="section_aware",
        token_count=10,
    )
    retrieved_result = RetrievalResult(chunk=dummy_chunk, score=0.01, rank=1, retrieval_method="test")

    unanswerable_query = "What was Tesla's commercial television advertising spend in 2023?"
    ans = generator.generate_answer(unanswerable_query, [retrieved_result])

    assert ans.abstained is True
    assert ABSTENTION_PHRASE in ans.response_text
