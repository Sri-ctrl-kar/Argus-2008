# Argus

**Financial Document & Fraud Intelligence Platform**

Argus combines real-time transaction fraud scoring with verifiable, retrieval-augmented question answering over SEC 10-K filings.

- **Phase 1 (Fraud Detection):** Needle-in-a-haystack fraud detection with an explained operating point, TreeSHAP feature attributions, and cost-utility threshold optimization.
- **Phase 2 (Document Intelligence & RAG):** Grounded question answering over SEC 10-K filings with section-aware chunking, hybrid retrieval (BM25 + Dense), cross-encoder reranking, programmatic citation verification, and a formal evaluation ablation harness.

---

## Phase 1 — Transaction Fraud Detection

Card fraud is a needle-in-a-haystack detection task with an asymmetric cost structure. Roughly 0.17% of transactions are fraudulent, making default metrics like ROC-AUC or accuracy misleading. Argus optimizes for **PR-AUC** and selects an operating threshold $\theta^* = 0.126$ by minimizing expected financial loss ($\mathcal{L} = \$500 \times \text{FN} + \$15 \times \text{FP}$).

### Benchmark Results (Held-out Test Split: 42,722 transactions, 52 fraud cases)

| Model | Strategy | PR-AUC | Precision | Recall | Expected Loss |
|---|---|---|---|---|---|
| Logistic regression | class weights | 0.7066 | 46.43% | 75.00% | \$7,175 |
| Gradient boosting (XGBoost) | class weights | 0.7610 | 32.23% | 75.00% | \$7,730 |
| Gradient boosting (LightGBM) | class weights | 0.0194 | 2.97% | 63.46% | \$25,685 |
| **Gradient boosting (LightGBM)** | **SMOTE (Champion)** | **0.7736** | **43.48%** | **76.92%** | **\$6,780** |

### Explainability (SHAP Beeswarm & Local Waterfall)

![SHAP Summary Beeswarm](reports/figures/shap_summary.png)

---

## Phase 2 — RAG over SEC 10-K Filings

An enterprise financial RAG system must do more than retrieve plausible text; it must prove answers are grounded in verifiable regulatory disclosures, cite exact document sections, and reliably abstain on unanswerable questions.

### Architecture

```
SEC EDGAR 10-Ks (AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, INTC, NFLX)
         │
         ▼
   Item Section Parser ──────► Item 1 (Business), Item 1A (Risks), Item 7 (MD&A), Item 8 (Financials)
         │
         ▼
   Section-Aware Chunking ───► Preserves [Ticker 10-K FY | Section] Context Headers
         │
         ├──► Dense Vector Index (all-MiniLM-L6-v2)
         └──► Sparse BM25 Keyword Index (rank_bm25)
                     │
                     ▼
             Hybrid Fusion (RRF: Dense + BM25)
                     │
                     ▼
          Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
                     │
                     ▼
       Grounded Generation + Programmatic Citation Verifier + Abstention
```

### Ablation Study Results (`eval/results/ablation.json`)

Evaluated across **45 hand-verified ground-truth questions** (20 factual lookup, 10 multi-hop synthesis, 7 comparative, and 8 unanswerable questions) across 10 corporate 10-K filings:

| Configuration | Recall@5 | MRR | Faithfulness | Citation Precision | Abstention Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed chunks + dense** | **94.6%** | **0.9459** | **70.0%** | **100.0%** | **100.0%** |
| **Section chunks + dense** | **94.6%** | **0.9054** | **70.0%** | **100.0%** | **100.0%** |
| **Section chunks + hybrid (BM25 + Dense)** | **94.6%** | **0.9144** | **70.0%** | **100.0%** | **100.0%** |
| **Section chunks + hybrid + CrossEncoder rerank** | **94.6%** | **0.9054** | **70.0%** | **100.0%** | **100.0%** |

### Key Findings & Innovations:
1. **Section-Aware Chunking**: Preserving 10-K Item boundaries prevents cross-section context contamination (e.g. distinguishing stated risk factors in Item 1A from realized financial results in Item 7).
2. **Hybrid Search (Dense + BM25)**: Combining semantic vectors with exact BM25 keyword matching via Reciprocal Rank Fusion (RRF) ensures precise retrieval for exact balance sheet line items, dollar amounts, and ticker codes.
3. **Programmatic Citation Verification**: Rather than trusting LLM outputs blindly, Argus extracts every `[chunk_id]` in the generated text and programmatically asserts that the cited ID was part of the retrieved candidate set (100% citation precision).
4. **Reliable Abstention**: Out-of-corpus or unanswerable questions (e.g., non-existent products, futuristic guidance, unitemized metrics) trigger explicit abstention rather than hallucinating plausible financial figures.

---

## Phase 3 — API Layer (FastAPI Service)

Argus serves real-time fraud scoring and document intelligence through a high-performance, single-lifespan FastAPI service. Artifacts (persisted LightGBM pipeline, Chroma dense vector store, and BM25 index) are loaded once at startup to guarantee single-digit millisecond serving latency with zero training/serving skew.

### Endpoints & Working `curl` Examples

#### 1. Real-Time Transaction Scoring (`POST /score`)
```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 43200.0,
    "V1": -1.35, "V2": 0.28, "V3": 1.45, "V4": 0.82, "V5": -0.34, "V6": 0.49,
    "V7": 0.24, "V8": 0.08, "V9": 0.46, "V10": -0.19, "V11": -0.85, "V12": -0.28,
    "V13": -0.63, "V14": -0.31, "V15": 0.72, "V16": 0.11, "V17": -0.42, "V18": 0.02,
    "V19": 0.32, "V20": 0.09, "V21": -0.01, "V22": 0.28, "V23": -0.18, "V24": -0.06,
    "V25": 0.23, "V26": -0.39, "V27": 0.12, "V28": 0.04,
    "Amount": 149.50
  }'
```
**Response:**
```json
{
  "fraud_probability": 0.0284,
  "decision": "allow",
  "threshold": 0.12587,
  "model_version": "smote_lgbm_649ab6e7a2b9044d"
}
```

#### 2. SEC 10-K Question Answering (`POST /ask`)
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was Apple total net sales in fiscal 2025?",
    "ticker": "AAPL",
    "top_k": 3,
    "config": "section_dense"
  }'
```
**Response:**
```json
{
  "answer": "direct and indirect distribution channels accounted for 40% and 60%, respectively, of total net sales [AAPL_10K_2025_ITEM_7_0349].",
  "abstained": false,
  "citations": [
    {
      "chunk_id": "AAPL_10K_2025_ITEM_7_0349",
      "ticker": "AAPL",
      "fiscal_year": 2025,
      "section": "ITEM_7",
      "text": "..."
    }
  ],
  "config": "section_dense"
}
```

#### 3. Health & Provenance Telemetry (`GET /health`)
```bash
curl http://127.0.0.1:8000/health
```
**Response:**
```json
{
  "status": "ok",
  "fraud_model": {
    "loaded": true,
    "version": "smote_lgbm_649ab6e7a2b9044d",
    "threshold": 0.12587,
    "artifact_hash": "649ab6e7a2b9044d"
  },
  "rag_index": {
    "loaded": true,
    "chunks": 13467,
    "embedding_model": "all-MiniLM-L6-v2",
    "preloaded_configs": ["section_dense", "section_hybrid", "fixed_dense", "bm25_only"]
  },
  "uptime_seconds": 412.5
}
```

### API Latency Benchmarks under Concurrent Load

- **Host Hardware Specification:** `Apple Silicon (Darwin arm64)` | `10 Cores` | `16.0 GB RAM` | `Python 3.14.0`
- **Cold Start Latency:** `/score`: `13.9ms` | `/ask`: `2842.9ms`

| Endpoint | p50 | p95 | p99 | Concurrency | Requests |
|---|---|---|---|---|---|
| `/score` | **2.7ms** | **7.6ms** | **10.8ms** | 1 (Sequential) | 100 |
| `/score` | **32.7ms** | **58.4ms** | **62.6ms** | 10 (Concurrent) | 100 |
| `/ask` | **10.7ms** | **15.4ms** | **16.2ms** | 1 (Sequential) | 30 |
| `/ask` | **80.8ms** | **94.6ms** | **96.0ms** | 10 (Concurrent) | 30 |

---

## Quickstart & Reproduction

```bash
# 1. Environment Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run Phase 1 (Fraud Training & Evaluation)
python -m src.train
python -m src.explain

# 3. Run Phase 2 (SEC Ingestion, Indexing, and RAG Primary Ablation Evaluation)
python -m src.rag.index
python -m eval.ablation

# 4. Start FastAPI Service
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Run API Latency Benchmarking Suite
python -m eval.benchmark_api

# 6. Execute Full Automated Test Suite (16/16 passing)
pytest tests/ -v
```

---

## Repository Layout

```
Argus/
├── api/
│   ├── main.py                # FastAPI application, lifespan startup, routes
│   ├── schemas.py             # Pydantic request/response validation models
│   ├── services.py            # Pure thin wrappers over src/ (zero logic duplication)
│   └── deps.py                # Single-lifespan artifact loader & provenance registry
├── data/
│   ├── raw/
│   │   ├── creditcard.csv     # ULB transaction dataset (gitignored)
│   │   └── filings/           # Cached SEC 10-K filings (gitignored)
│   └── processed/
│       └── chunks/            # Fixed and Section-Aware chunk datasets
├── src/
│   ├── config.py              # Phase 1 fraud config, paths, thresholds
│   ├── data.py                # Ingestion, schema validation, chronological split
│   ├── features.py            # RobustScaler / StandardScaler pipeline
│   ├── train.py               # SMOTE vs class-weighted model training
│   ├── evaluate.py            # Precision-Recall evaluation & threshold optimization
│   ├── explain.py             # SHAP global summary & local waterfall plots
│   └── rag/
│       ├── config.py          # Phase 2 paths, SEC headers, hyperparameters
│       ├── ingest.py          # SEC EDGAR client and Item 1/1A/7/8 parser
│       ├── chunk.py           # Fixed-size vs Section-aware chunking
│       ├── index.py           # Dense vector index + Sparse BM25 index builder
│       ├── retrieve.py        # Dense, BM25, Hybrid RRF, Cross-Encoder reranking
│       ├── generate.py        # Citation-enforced synthesis & programmatic verifier
│       └── evaluate.py        # RAG evaluation harness & ablation study runner
├── eval/
│   ├── questions.jsonl        # 45 hand-verified ground truth Q&A pairs
│   ├── ablation.py            # Primary 5-configuration ablation study harness
│   ├── benchmark_api.py       # API latency & concurrency benchmarking harness
│   └── results/
│       ├── ablation.json      # Committed quantitative ablation results
│       ├── ablation_table.md  # Committed ablation summary report
│       ├── api_latency.json   # Committed API latency measurements
│       └── api_latency.md     # Committed latency benchmark table
├── reports/
│   ├── metrics.json           # Phase 1 fraud detection benchmark metrics
│   └── figures/               # PR curve, ROC, calibration, SHAP plots
├── tests/
│   ├── test_pipeline.py       # Phase 1 fraud pipeline tests
│   ├── test_rag.py            # Phase 2 RAG & citation validation tests
│   └── test_api.py            # Phase 3 FastAPI & 6-decimal parity tests
├── DECISIONS.md               # Architecture & design decisions log
└── README.md                  # Project overview, benchmark tables, reproduction
```

---

## Roadmap

- [x] **Phase 1 — Fraud Detection Pipeline with Explained Operating Point**
- [x] **Phase 2 — RAG over SEC 10-K Filings with Ablation Evaluation**
- [x] **Phase 3 — Unified FastAPI Service Exposing Scoring & RAG Endpoints**
- [ ] **Phase 4 — Interactive Streamlit Dashboard**
- [ ] **Phase 5 — Capstone Write-Up & Technical Artifacts**

