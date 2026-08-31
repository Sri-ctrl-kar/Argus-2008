"""FastAPI Application Entry Point for Argus Platform."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.deps import registry
from api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    TransactionScoreRequest,
    TransactionScoreResponse,
)
from api.services import ask_filings_service, score_transaction_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("argus.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager ensuring artifacts are loaded once at startup."""
    logger.info("Starting Argus API service...")
    try:
        registry.load_all_artifacts()
        logger.info("All model and document intelligence artifacts ready.")
    except Exception as e:
        logger.critical(f"FATAL: Service startup aborted due to artifact loading error: {e}")
        raise e
    yield
    logger.info("Shutting down Argus API service.")


app = FastAPI(
    title="Argus Financial Intelligence API",
    description="Unified API serving transaction fraud inference and SEC 10-K RAG intelligence.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format informative 422 validation errors detailing missing or invalid fields."""
    error_messages = []
    for err in exc.errors():
        location = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        error_messages.append(f"{location}: {msg}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request payload validation failed.",
            "errors": error_messages,
        },
    )


@app.post(
    "/score",
    response_model=TransactionScoreResponse,
    summary="Score transaction for fraud probability",
    tags=["Fraud Detection"],
)
def score_transaction(payload: TransactionScoreRequest) -> TransactionScoreResponse:
    """Scores a single transaction and returns estimated fraud probability and decision flag."""
    try:
        return score_transaction_service(payload)
    except Exception as e:
        logger.error(f"Error processing /score request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Answer financial question over SEC 10-K filings",
    tags=["Document Intelligence"],
)
def ask_question(payload: AskRequest) -> AskResponse:
    """Retrieves regulatory filings passages and synthesizes a citation-grounded response."""
    try:
        return ask_filings_service(payload)
    except Exception as e:
        logger.error(f"Error processing /ask request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check and artifact provenance",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """Reports service liveness, uptime, and provenance of all loaded artifacts."""
    return HealthResponse(
        status="ok",
        fraud_model={
            "loaded": registry.fraud_pipeline is not None,
            "version": registry.fraud_model_version,
            "threshold": registry.fraud_threshold,
            "artifact_hash": registry.fraud_artifact_hash,
        },
        rag_index={
            "loaded": bool(registry.rag_retrievers),
            "chunks": registry.rag_chunk_count,
            "embedding_model": registry.rag_embedding_model,
            "preloaded_configs": list(registry.rag_retrievers.keys()),
        },
        uptime_seconds=registry.get_uptime_seconds(),
    )
