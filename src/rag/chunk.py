"""
Multi-Strategy Document Chunking Module.
Implements:
  1. Fixed-Size Chunking (Baseline: 512 tokens with 50-token overlap)
  2. Section-Aware Recursive Chunking (Candidate: Split on Item boundaries, sub-split preserving metadata headers)
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.config import (
    FIXED_CHUNK_SIZE,
    FIXED_CHUNK_OVERLAP,
    SECTION_CHUNK_MAX_TOKENS,
    SECTION_CHUNK_MIN_TOKENS,
    CHUNKS_DIR,
    ITEM_SECTIONS,
)
from src.rag.ingest import ParsedFiling, build_curated_10k_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents an atomic text chunk with metadata."""
    chunk_id: str
    doc_id: str
    ticker: str
    company_name: str
    fiscal_year: int
    section_name: str
    section_title: str
    text: str
    strategy: str
    token_count: int
    section: Optional[str] = None

    def __post_init__(self):
        if self.section is None:
            self.section = self.section_name

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["section"] = self.section or self.section_name
        return d




def count_tokens(text: str) -> int:
    """Estimates token count via whitespace word splitting."""
    return len(text.split())


class FixedSizeChunker:
    """
    Baseline Strategy: Naive sliding window chunking across entire concatenated document
    with fixed token length and constant overlap.
    """

    def __init__(self, chunk_size: int = FIXED_CHUNK_SIZE, overlap: int = FIXED_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_filing(self, filing: ParsedFiling) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        # Concatenate all sections into one continuous stream (ignoring section boundaries)
        full_text = "\n\n".join([f"{sec_name}:\n{sec_text}" for sec_name, sec_text in filing.sections.items()])
        words = full_text.split()

        step = max(1, self.chunk_size - self.overlap)
        chunk_idx = 0
        for i in range(0, len(words), step):
            window_words = words[i : i + self.chunk_size]
            if len(window_words) < 20:
                continue

            chunk_text = " ".join(window_words)
            chunk_id = f"{filing.doc_id}_fixed_{chunk_idx:04d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=filing.doc_id,
                    ticker=filing.ticker,
                    company_name=filing.company_name,
                    fiscal_year=filing.fiscal_year,
                    section_name="UNSTRUCTURED",
                    section_title="Full Document (Fixed Window)",
                    text=chunk_text,
                    strategy="fixed",
                    token_count=len(window_words),
                )
            )
            chunk_idx += 1

        return chunks


class SectionAwareChunker:
    """
    Candidate Strategy: Section-aware chunking.
    1. Splits strictly on 10-K Item boundaries (Item 1, 1A, 7, 8).
    2. Sub-splits long sections recursively on natural paragraph breaks.
    3. Prepends a structured metadata header to every sub-chunk so the LLM and retriever
       always retain context of the company, fiscal year, and section.
    """

    def __init__(self, max_tokens: int = SECTION_CHUNK_MAX_TOKENS, min_tokens: int = SECTION_CHUNK_MIN_TOKENS):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def chunk_filing(self, filing: ParsedFiling) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for sec_name, sec_text in filing.sections.items():
            sec_title = ITEM_SECTIONS.get(sec_name, sec_name)
            # Prepend contextual header
            header = f"[{filing.ticker} ({filing.company_name}) 10-K FY{filing.fiscal_year} | {sec_title}]\n"

            # Split section by paragraphs or bullet items
            raw_lines = [p.strip() for p in sec_text.split("\n") if p.strip()]
            paragraphs = []
            for line in raw_lines:
                if len(line.split()) > self.max_tokens:
                    # Sub-split long lines
                    words = line.split()
                    for j in range(0, len(words), self.max_tokens):
                        paragraphs.append(" ".join(words[j : j + self.max_tokens]))
                else:
                    paragraphs.append(line)

            current_buffer: List[str] = []
            current_token_count = 0


            for para in paragraphs:
                para_tokens = count_tokens(para)
                if current_token_count + para_tokens > self.max_tokens and current_buffer:
                    # Flush current buffer
                    chunk_body = "\n\n".join(current_buffer)
                    full_chunk_text = header + chunk_body
                    chunk_id = f"{filing.doc_id}_{sec_name}_{chunk_idx:04d}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=filing.doc_id,
                            ticker=filing.ticker,
                            company_name=filing.company_name,
                            fiscal_year=filing.fiscal_year,
                            section_name=sec_name,
                            section_title=sec_title,
                            text=full_chunk_text,
                            strategy="section_aware",
                            token_count=count_tokens(full_chunk_text),
                        )
                    )
                    chunk_idx += 1
                    current_buffer = [para]
                    current_token_count = para_tokens
                else:
                    current_buffer.append(para)
                    current_token_count += para_tokens

            # Flush remaining paragraphs
            if current_buffer:
                chunk_body = "\n\n".join(current_buffer)
                full_chunk_text = header + chunk_body
                chunk_id = f"{filing.doc_id}_{sec_name}_{chunk_idx:04d}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=filing.doc_id,
                        ticker=filing.ticker,
                        company_name=filing.company_name,
                        fiscal_year=filing.fiscal_year,
                        section_name=sec_name,
                        section_title=sec_title,
                        text=full_chunk_text,
                        strategy="section_aware",
                        token_count=count_tokens(full_chunk_text),
                    )
                )
                chunk_idx += 1

        return chunks


def generate_all_chunks() -> Dict[str, List[DocumentChunk]]:
    """
    Loads parsed 10-Ks and generates both Fixed and Section-Aware chunk datasets.
    Persists datasets to `data/processed/chunks/`.
    """
    filings = build_curated_10k_corpus()
    fixed_chunker = FixedSizeChunker()
    section_chunker = SectionAwareChunker()

    fixed_chunks: List[DocumentChunk] = []
    section_chunks: List[DocumentChunk] = []

    for f in filings:
        fixed_chunks.extend(fixed_chunker.chunk_filing(f))
        section_chunks.extend(section_chunker.chunk_filing(f))

    # Save to disk
    fixed_path = CHUNKS_DIR / "fixed_chunks.json"
    section_path = CHUNKS_DIR / "section_chunks.json"

    with open(fixed_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in fixed_chunks], f, indent=2)
    logger.info("Saved %d fixed chunks to %s", len(fixed_chunks), fixed_path)

    with open(section_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in section_chunks], f, indent=2)
    logger.info("Saved %d section-aware chunks to %s", len(section_chunks), section_path)

    return {
        "fixed": fixed_chunks,
        "section_aware": section_chunks,
    }


if __name__ == "__main__":
    result = generate_all_chunks()
    print(f"\nChunking complete:")
    print(f"- Fixed-size strategy: {len(result['fixed'])} chunks")
    print(f"- Section-aware strategy: {len(result['section_aware'])} chunks")
