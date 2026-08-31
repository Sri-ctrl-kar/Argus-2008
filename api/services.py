"""Service Layer for Argus API.

THIN WRAPPER OVER src/ -- NO PREPROCESSING OR MODEL LOGIC REIMPLEMENTED HERE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from api.deps import registry
from api.schemas import (
    AskRequest,
    AskResponse,
    CitationItem,
    TransactionScoreRequest,
    TransactionScoreResponse,
)
from src.config import FEATURE_COLS
from src.rag.generate import GroundedGenerator


def score_transaction_service(request: TransactionScoreRequest) -> TransactionScoreResponse:
    """Scores a transaction through the persisted whole pipeline artifact."""
    if not registry.is_loaded or registry.fraud_pipeline is None:
        raise RuntimeError("Fraud pipeline is not initialized.")

    # Convert request to single-row DataFrame with exact training feature order
    feature_dict = request.to_feature_dict()
    df = pd.DataFrame([feature_dict])[FEATURE_COLS]

    # Predict probability via loaded whole pipeline (preprocessor + model together)
    proba = registry.fraud_pipeline.predict_proba(df)
    fraud_prob = float(proba[0, 1])

    decision = "flag" if fraud_prob >= registry.fraud_threshold else "allow"

    return TransactionScoreResponse(
        fraud_probability=fraud_prob,
        decision=decision,
        threshold=registry.fraud_threshold,
        model_version=registry.fraud_model_version,
    )


def ask_filings_service(request: AskRequest) -> AskResponse:
    """Answers a financial question grounded in SEC 10-K regulatory disclosures."""
    if not registry.is_loaded:
        raise RuntimeError("RAG index is not initialized.")

    # Get retriever from preloaded cache or build on demand
    retriever = registry.rag_retrievers.get(request.config)
    if retriever is None:
        from src.rag import retrieve
        retriever = retrieve.build(request.config)
        registry.rag_retrievers[request.config] = retriever

    # Execute retrieval
    results = retriever.retrieve(request.question, k=request.top_k)

    # Generate answer with citation verification and abstention checks
    generator = GroundedGenerator()
    gen = generator.generate_answer(request.question, results)

    # Build structured citations
    citations: List[CitationItem] = []
    for r in results:
        citations.append(
            CitationItem(
                chunk_id=r.chunk.chunk_id,
                ticker=r.chunk.ticker,
                fiscal_year=r.chunk.fiscal_year,
                section=r.chunk.section_name,
                text=r.chunk.text,
            )
        )

    return AskResponse(
        answer=gen.response_text,
        abstained=gen.abstained,
        citations=citations,
        config=request.config,
    )
