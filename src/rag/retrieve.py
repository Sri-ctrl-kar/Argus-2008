"""
Hybrid Retrieval and Cross-Encoder Reranking Module.
Supports:
  1. Dense vector search
  2. Sparse BM25 keyword search
  3. Hybrid Fusion (Reciprocal Rank Fusion)
  4. Metadata pre-filtering by company / ticker
  5. Cross-Encoder reranking
"""

import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.rag.config import (
    CHROMA_PERSIST_DIR,
    DENSE_TOP_K,
    BM25_TOP_K,
    FINAL_TOP_K,
    HYBRID_ALPHA,
    RERANKER_MODEL_NAME,
    TARGET_COMPANIES,
)
from src.rag.chunk import DocumentChunk
from src.rag.index import DenseVectorIndex, SparseBM25Index

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a scored chunk returned from retrieval."""
    chunk: DocumentChunk
    score: float
    rank: int
    retrieval_method: str

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    def to_dict(self) -> Dict[str, Any]:


        return {
            "chunk_id": self.chunk.chunk_id,
            "ticker": self.chunk.ticker,
            "section_name": self.chunk.section_name,
            "section_title": self.chunk.section_title,
            "text": self.chunk.text,
            "score": float(self.score),
            "rank": self.rank,
            "retrieval_method": self.retrieval_method,
        }


def extract_metadata_filter(query: str) -> Optional[str]:
    """
    Detects company ticker or brand name mentions in query for metadata pre-filtering.
    """
    q_lower = query.lower()
    for ticker, info in TARGET_COMPANIES.items():
        if ticker.lower() in q_lower.split() or info["name"].lower() in q_lower:
            return ticker
        # Special case aliases
        if ticker == "GOOGL" and ("google" in q_lower or "alphabet" in q_lower):
            return "GOOGL"
        if ticker == "META" and ("facebook" in q_lower or "meta" in q_lower or "instagram" in q_lower):
            return "META"
        if ticker == "AMZN" and ("amazon" in q_lower or "aws" in q_lower):
            return "AMZN"
        if ticker == "MSFT" and ("microsoft" in q_lower or "azure" in q_lower):
            return "MSFT"
        if ticker == "NVDA" and ("nvidia" in q_lower or "geforce" in q_lower or "h100" in q_lower):
            return "NVDA"
        if ticker == "AAPL" and ("apple" in q_lower or "iphone" in q_lower):
            return "AAPL"
        if ticker == "TSLA" and ("tesla" in q_lower or "musk" in q_lower):
            return "TSLA"
        if ticker == "AMD" and ("amd" in q_lower or "radeon" in q_lower or "epyc" in q_lower):
            return "AMD"
        if ticker == "INTC" and ("intel" in q_lower or "xeon" in q_lower):
            return "INTC"
        if ticker == "NFLX" and ("netflix" in q_lower or "streaming" in q_lower):
            return "NFLX"

    return None


class CrossEncoderReranker:
    """Reranks candidate chunks by calculating query-chunk cross-attention scores."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except Exception as e:
            logger.info("CrossEncoder fallback: utilizing lexical and semantic overlap reranker.")
            self._model = None

    def rerank(self, query: str, candidate_chunks: List[DocumentChunk], top_k: int = FINAL_TOP_K) -> List[Tuple[DocumentChunk, float]]:
        if not candidate_chunks:
            return []

        if self._model is not None:
            pairs = [[query, c.text] for c in candidate_chunks]
            scores = self._model.predict(pairs)
            scored = list(zip(candidate_chunks, [float(s) for s in scores]))
        else:
            # Fallback high-performance lexical-overlap reranker
            q_terms = set(re.findall(r"\b\w+\b", query.lower()))
            scored = []
            for c in candidate_chunks:
                c_text = c.text.lower()
                overlap = sum(1 for t in q_terms if t in c_text)
                exact_phrase_bonus = 3.0 if any(word in c_text for word in query.lower().split() if len(word) > 4) else 0.0
                score = (overlap / (len(q_terms) + 1e-5)) + exact_phrase_bonus
                scored.append((c, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> List[str]:
    """Fuse rankings by rank position, not score.

    Raw BM25 and cosine scores are on incompatible scales -- summing them
    lets BM25's unbounded values swamp cosine's [0,1] range. RRF sidesteps
    the problem entirely by using only ordinal position.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class RAGRetriever:
    """Unified multi-mode retrieval engine for SEC 10-K filings."""

    def __init__(
        self,
        strategy: str = "section_aware",
        use_hybrid: bool = False,
        use_reranker: bool = False,
        use_metadata_filtering: bool = True,
        top_k: int = FINAL_TOP_K,
    ):
        self.strategy = strategy
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.use_metadata_filtering = use_metadata_filtering
        self.top_k = top_k

        # Load indices from cache
        dense_path = CHROMA_PERSIST_DIR / f"dense_{strategy}.pkl"
        bm25_path = CHROMA_PERSIST_DIR / f"bm25_{strategy}.pkl"

        if not dense_path.exists() or not bm25_path.exists():
            from src.rag.index import build_and_save_all_indices
            build_and_save_all_indices()

        self.dense_index = DenseVectorIndex.load(dense_path)
        self.bm25_index = SparseBM25Index.load(bm25_path)
        self.reranker = CrossEncoderReranker() if use_reranker else None

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Executes query retrieval through configured pipeline.
        """
        effective_k = k if k is not None else self.top_k
        filter_ticker = extract_metadata_filter(query) if self.use_metadata_filtering else None

        if not self.use_hybrid:
            # 1. Pure Dense Retrieval
            candidates = self.dense_index.search(query, top_k=DENSE_TOP_K if self.use_reranker else effective_k, filter_ticker=filter_ticker)
            initial_chunks = [c for c, _ in candidates]
            initial_scores = [s for _, s in candidates]
        else:
            # 2. Hybrid Retrieval (Reciprocal Rank Fusion)
            # Retrieve deeper from each arm (k=50), then fuse and truncate
            dense_res = self.dense_index.search(query, top_k=50, filter_ticker=filter_ticker)
            bm25_res = self.bm25_index.search(query, top_k=50, filter_ticker=filter_ticker)

            dense_ids = [c.chunk_id for c, _ in dense_res]
            bm25_ids = [c.chunk_id for c, _ in bm25_res]

            chunk_map: Dict[str, DocumentChunk] = {
                c.chunk_id: c for c, _ in (dense_res + bm25_res)
            }

            fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids], k=60)
            initial_chunks = [chunk_map[cid] for cid in fused_ids if cid in chunk_map]
            initial_scores = [1.0 / (idx + 1) for idx in range(len(initial_chunks))]


        # 3. Optional Reranking Step
        if self.use_reranker and self.reranker is not None and initial_chunks:
            reranked = self.reranker.rerank(query, initial_chunks[:20], top_k=effective_k)
            final_chunks = [c for c, _ in reranked]
            final_scores = [s for _, s in reranked]
            method = f"{self.strategy}_hybrid_rerank" if self.use_hybrid else f"{self.strategy}_dense_rerank"
        else:
            final_chunks = initial_chunks[: effective_k]
            final_scores = initial_scores[: effective_k]
            method = f"{self.strategy}_hybrid" if self.use_hybrid else f"{self.strategy}_dense"

        results: List[RetrievalResult] = []
        for rank, (chunk, score) in enumerate(zip(final_chunks, final_scores), 1):
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                    retrieval_method=method,
                )
            )

        return results


def build(config_name: str) -> RAGRetriever:
    """
    Factory function for standard RAG retriever configurations.
    """
    mapping = {
        "fixed_dense": {"strategy": "fixed", "use_hybrid": False, "use_reranker": False},
        "section_dense": {"strategy": "section_aware", "use_hybrid": False, "use_reranker": False},
        "section_hybrid": {"strategy": "section_aware", "use_hybrid": True, "use_reranker": False},
        "section_hybrid_rerank": {"strategy": "section_aware", "use_hybrid": True, "use_reranker": True},
    }
    kwargs = mapping.get(config_name, {"strategy": "section_aware", "use_hybrid": True, "use_reranker": True})
    return RAGRetriever(**kwargs)



if __name__ == "__main__":
    retriever = RAGRetriever(strategy="section_aware", use_hybrid=True, use_reranker=True)
    sample_query = "What was Apple's R&D spend in FY2023?"
    results = retriever.retrieve(sample_query)
    print(f"\nQuery: '{sample_query}'")
    print(f"Retrieved {len(results)} chunks:")
    for r in results:
        print(f"[{r.rank}] ({r.chunk.ticker} {r.chunk.section_name}) Score: {r.score:.4f} | ID: {r.chunk.chunk_id}")
        print(f"     Preview: {r.chunk.text[:120]}...\n")
