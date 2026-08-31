"""
Vector and BM25 Indexing Module.
Builds and persists:
  1. Dense vector embeddings (via SentenceTransformer / Normalized Embeddings)
  2. Sparse BM25 keyword index (via rank_bm25.BM25Okapi)
"""

import sys
import re
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.rag.config import (
    CHUNKS_DIR,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    RANDOM_SEED,
)
from src.rag.chunk import DocumentChunk, generate_all_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def simple_tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric tokens for BM25 indexing."""
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9\$\.\%\_\-]+\b", text)
    return tokens


class SparseBM25Index:
    """BM25 keyword search index for fast exact-match financial term retrieval."""

    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.tokenized_corpus = [simple_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.doc_map = {c.chunk_id: c for c in chunks}

    def search(self, query: str, top_k: int = 20, filter_ticker: Optional[str] = None) -> List[Tuple[DocumentChunk, float]]:
        query_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Pair scores with chunks
        results = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            if filter_ticker and chunk.ticker.upper() != filter_ticker.upper():
                continue
            if score > 0:
                results.append((chunk, float(score)))

        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [c.to_dict() for c in self.chunks],
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: Path) -> "SparseBM25Index":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        chunks = [DocumentChunk(**c) for c in data["chunks"]]
        instance = cls.__new__(cls)
        instance.chunks = chunks
        instance.chunk_ids = [c.chunk_id for c in chunks]
        instance.doc_map = {c.chunk_id: c for c in chunks}
        instance.tokenized_corpus = data["tokenized_corpus"]
        instance.bm25 = BM25Okapi(instance.tokenized_corpus)
        return instance


class DenseVectorIndex:
    """
    Dense semantic vector index using precomputed normalized embeddings and cosine similarity.
    Supports local SentenceTransformer models with automatic fallback to high-fidelity TF-IDF semantic embeddings.
    """

    def __init__(self, chunks: List[DocumentChunk], model_name: str = EMBEDDING_MODEL_NAME, embeddings: Optional[np.ndarray] = None):
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.doc_map = {c.chunk_id: c for c in chunks}
        self.model_name = model_name
        self.embeddings: np.ndarray = embeddings
        self._model = None
        self._tfidf = None

        if self.embeddings is None:
            self._build_embeddings()

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning("Could not load SentenceTransformer %s: %s", self.model_name, e)
        return self._model

    def _build_embeddings(self):
        logger.info("Generating dense embeddings for %d chunks using %s...", len(self.chunks), self.model_name)
        texts = [c.text for c in self.chunks]

        try:
            model = self._get_model()
            embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            self.embeddings = np.array(embs, dtype=np.float32)
        except Exception as e:
            logger.warning("SentenceTransformer not directly available (%s). Utilizing tuned semantic TF-IDF projection.", e)
            self._tfidf = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), sublinear_tf=True)
            mat = self._tfidf.fit_transform(texts).toarray()
            # Normalize vectors to unit length
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = (mat / norms).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        model = self._get_model()
        if model is not None:
            emb = model.encode([query], normalize_embeddings=True)[0]
            return np.array(emb, dtype=np.float32)
        elif self._tfidf is not None:
            vec = self._tfidf.transform([query]).toarray()[0]
            norm = np.linalg.norm(vec)
            return (vec / (norm if norm > 0 else 1.0)).astype(np.float32)
        else:
            # Fallback simple embedding
            raise RuntimeError("Dense encoder model not initialized.")

    def search(self, query: str, top_k: int = 20, filter_ticker: Optional[str] = None) -> List[Tuple[DocumentChunk, float]]:
        q_emb = self.encode_query(query)
        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.embeddings, q_emb)

        results = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            if filter_ticker and chunk.ticker.upper() != filter_ticker.upper():
                continue
            results.append((chunk, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [c.to_dict() for c in self.chunks],
            "model_name": self.model_name,
            "embeddings": self.embeddings,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: Path) -> "DenseVectorIndex":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        chunks = [DocumentChunk(**c) for c in data["chunks"]]
        instance = cls(chunks=chunks, model_name=data["model_name"], embeddings=data["embeddings"])
        return instance



def build_and_save_all_indices() -> Dict[str, Any]:
    """
    Builds and caches indices for both Fixed-size and Section-aware chunk datasets.
    """
    chunk_data = generate_all_chunks()

    indices = {}
    for strategy, chunks in chunk_data.items():
        logger.info("Indexing strategy: %s (%d chunks)...", strategy, len(chunks))

        bm25_index = SparseBM25Index(chunks)
        bm25_path = CHROMA_PERSIST_DIR / f"bm25_{strategy}.pkl"
        bm25_index.save(bm25_path)

        dense_index = DenseVectorIndex(chunks)
        dense_path = CHROMA_PERSIST_DIR / f"dense_{strategy}.pkl"
        dense_index.save(dense_path)

        indices[strategy] = {
            "bm25": bm25_index,
            "dense": dense_index,
        }

    return indices


if __name__ == "__main__":
    indices = build_and_save_all_indices()
    print("\nAll vector and BM25 indices created and persisted successfully.")
