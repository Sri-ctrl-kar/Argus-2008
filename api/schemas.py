"""Pydantic Request and Response Schemas for Argus API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class TransactionScoreRequest(BaseModel):
    """Payload representing one raw credit card transaction for fraud scoring."""

    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset")
    V1: float = Field(...)
    V2: float = Field(...)
    V3: float = Field(...)
    V4: float = Field(...)
    V5: float = Field(...)
    V6: float = Field(...)
    V7: float = Field(...)
    V8: float = Field(...)
    V9: float = Field(...)
    V10: float = Field(...)
    V11: float = Field(...)
    V12: float = Field(...)
    V13: float = Field(...)
    V14: float = Field(...)
    V15: float = Field(...)
    V16: float = Field(...)
    V17: float = Field(...)
    V18: float = Field(...)
    V19: float = Field(...)
    V20: float = Field(...)
    V21: float = Field(...)
    V22: float = Field(...)
    V23: float = Field(...)
    V24: float = Field(...)
    V25: float = Field(...)
    V26: float = Field(...)
    V27: float = Field(...)
    V28: float = Field(...)
    Amount: float = Field(..., description="Transaction amount in currency units")

    def to_feature_dict(self) -> Dict[str, float]:
        """Convert to ordered dictionary matching training column specification."""
        return self.model_dump()


class TransactionScoreResponse(BaseModel):
    """Response model for transaction fraud scoring."""

    fraud_probability: float = Field(..., description="Estimated posterior probability of fraud")
    decision: str = Field(..., description="Decision flag: flag if probability >= threshold, else allow")
    threshold: float = Field(..., description="Optimal decision threshold from Phase 1 cost-utility analysis")
    model_version: str = Field(..., description="Model identifier and provenance hash")


class AskRequest(BaseModel):
    """Request model for regulatory document Q&A."""

    question: str = Field(..., min_length=3, description="Financial or disclosure question over SEC 10-Ks")
    ticker: Optional[str] = Field(None, description="Optional ticker filter (e.g. AAPL, MSFT, NVDA)")
    top_k: int = Field(5, ge=1, le=20, description="Number of context passages to retrieve")
    config: str = Field("section_dense", description="Retriever configuration: fixed_dense, section_dense, section_hybrid, section_hybrid_rerank, bm25_only")


class CitationItem(BaseModel):
    """Structured citation representing a supporting passage in a source 10-K."""

    chunk_id: str = Field(..., description="Deterministic chunk identifier")
    ticker: str = Field(..., description="Company ticker symbol")
    fiscal_year: int = Field(..., description="Reported fiscal year")
    section: str = Field(..., description="10-K Item section (e.g. ITEM_7, ITEM_8)")
    text: str = Field(..., description="Passage text")


class AskResponse(BaseModel):
    """Response model for document intelligence question answering."""

    answer: str = Field(..., description="Synthesized answer grounded in cited regulatory passages")
    abstained: bool = Field(..., description="True if the system abstained due to missing or ungrounded evidence")
    citations: List[CitationItem] = Field(default_factory=list, description="Structured source citations")
    config: str = Field(..., description="Retrieval configuration used to serve the request")


class HealthResponse(BaseModel):
    """System health, uptime, and loaded artifact provenance."""

    status: str = Field(..., description="System operational status")
    fraud_model: Dict[str, Any] = Field(..., description="Fraud detection model metadata and artifact hash")
    rag_index: Dict[str, Any] = Field(..., description="RAG vector store metadata and chunk counts")
    uptime_seconds: float = Field(..., description="Total service uptime in seconds")
