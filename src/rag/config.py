"""
Central Configuration for Argus RAG Subsystem.
Defines paths, model identifiers, SEC EDGAR compliance headers, and hyperparameters.
"""

import os
import sys
from pathlib import Path

# Enable offline mode for cached HuggingFace models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# Paths
# ==========================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FILINGS_RAW_DIR = RAW_DATA_DIR / "filings"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_DIR = PROCESSED_DATA_DIR / "chunks"
CHROMA_PERSIST_DIR = PROCESSED_DATA_DIR / "chroma_db"

EVAL_DIR = PROJECT_ROOT / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
ABLATION_RESULTS_FILE = RESULTS_DIR / "ablation.json"
ABLATION_REPORT_MD = RESULTS_DIR / "ablation_table.md"

# Ensure directories exist
FILINGS_RAW_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# SEC EDGAR Ingestion Settings
# ==========================================
# SEC EDGAR requires a User-Agent in the format: Sample Company Name AdminContact@<sample company domain>.com
SEC_USER_AGENT = "ArgusFinancialIntelligence/1.0 (argus-capstone@argus.internal)"
SEC_RATE_LIMIT_DELAY = 0.12  # Seconds between requests (max 10 requests/sec allowed by SEC)

# 10 Benchmark Companies across Tech, Semis, and Retail
TARGET_COMPANIES = {
    "AAPL": {"cik": "0000320193", "name": "Apple Inc."},
    "MSFT": {"cik": "0000789019", "name": "Microsoft Corp"},
    "NVDA": {"cik": "0001045810", "name": "NVIDIA Corp"},
    "AMZN": {"cik": "0001018724", "name": "Amazon.com Inc"},
    "GOOGL": {"cik": "0001652044", "name": "Alphabet Inc."},
    "META": {"cik": "0001326801", "name": "Meta Platforms, Inc."},
    "TSLA": {"cik": "0001318605", "name": "Tesla, Inc."},
    "AMD": {"cik": "0000002488", "name": "Advanced Micro Devices, Inc."},
    "INTC": {"cik": "0000050863", "name": "Intel Corp"},
    "NFLX": {"cik": "0001065280", "name": "Netflix, Inc."},
}

# 10-K Section Definitions
ITEM_SECTIONS = {
    "ITEM_1": "Item 1. Business",
    "ITEM_1A": "Item 1A. Risk Factors",
    "ITEM_7": "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "ITEM_8": "Item 8. Financial Statements and Supplementary Data",
}

# ==========================================
# Chunking Hyperparameters
# ==========================================
FIXED_CHUNK_SIZE = 50  # tokens / approx words
FIXED_CHUNK_OVERLAP = 10
SECTION_CHUNK_MAX_TOKENS = 50
SECTION_CHUNK_MIN_TOKENS = 10



# ==========================================
# Embedding & Retrieval Settings
# ==========================================
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DENSE_TOP_K = 20
BM25_TOP_K = 20
HYBRID_ALPHA = 0.5  # Weight for dense vs BM25 in reciprocal rank fusion
FINAL_TOP_K = 5

# ==========================================
# Reproducibility Seed
# ==========================================
RANDOM_SEED = 42
