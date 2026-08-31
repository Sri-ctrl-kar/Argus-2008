"""Artifact Lifespan and Dependency Registry for Argus API."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.config import MODELS_DIR
from src.rag.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME

logger = logging.getLogger("argus.api")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file for exact artifact provenance tracking."""
    if not file_path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


class ArtifactRegistry:
    """Singleton container storing preloaded machine learning models and RAG indexes."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.is_loaded: bool = False

        # Phase 1: Fraud Detection
        self.fraud_pipeline: Any = None
        self.fraud_metadata: Dict[str, Any] = {}
        self.fraud_model_version: str = ""
        self.fraud_threshold: float = 0.5
        self.fraud_artifact_hash: str = ""

        # Phase 2: RAG
        self.rag_retrievers: Dict[str, Any] = {}
        self.rag_chunk_count: int = 0
        self.rag_embedding_model: str = EMBEDDING_MODEL_NAME

    def load_all_artifacts(self) -> None:
        """Loads and validates all system artifacts at startup. Fails loudly on missing files."""
        logger.info("Initializing Argus API Artifact Registry...")
        self.start_time = time.time()

        # 1. Load Phase 1 Fraud Model & Metadata
        pipeline_path = MODELS_DIR / "fraud_pipeline.joblib"
        meta_path = MODELS_DIR / "model_metadata.json"

        if not pipeline_path.exists():
            raise RuntimeError(f"Required fraud model artifact missing at {pipeline_path}. Run src/train.py first.")
        if not meta_path.exists():
            raise RuntimeError(f"Required model metadata missing at {meta_path}. Run src/train.py first.")

        try:
            self.fraud_artifact_hash = compute_sha256(pipeline_path)
            self.fraud_pipeline = joblib.load(pipeline_path)
            self.fraud_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            model_name = self.fraud_metadata.get("best_model_name", "model")
            self.fraud_threshold = float(self.fraud_metadata.get("chosen_decision_threshold", 0.5))
            self.fraud_model_version = f"{model_name}_{self.fraud_artifact_hash}"
            logger.info(f"Loaded fraud pipeline {self.fraud_model_version} (threshold={self.fraud_threshold:.4f})")
        except Exception as e:
            raise RuntimeError(f"Failed to load fraud pipeline artifact: {e}") from e

        # 2. Preload Phase 2 RAG Retrievers
        try:
            from src.rag import retrieve

            # Preload default and key configurations
            configs_to_preload = ["section_dense", "section_hybrid", "fixed_dense", "bm25_only"]
            for cfg in configs_to_preload:
                self.rag_retrievers[cfg] = retrieve.build(cfg)

            # Get chunk count from primary index
            primary_retriever = self.rag_retrievers.get("section_dense")
            if primary_retriever and hasattr(primary_retriever, "dense_index"):
                self.rag_chunk_count = len(primary_retriever.dense_index.chunks)

            logger.info(f"Preloaded RAG retrievers across {self.rag_chunk_count} corpus chunks")
        except Exception as e:
            raise RuntimeError(f"Failed to load RAG index artifacts: {e}") from e

        self.is_loaded = True

    def get_uptime_seconds(self) -> float:
        """Return seconds elapsed since service startup."""
        if self.start_time == 0.0:
            return 0.0
        return round(time.time() - self.start_time, 2)


registry = ArtifactRegistry()
