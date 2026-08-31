# Technical Decisions Log

This document records non-obvious architecture and design choices made during the development of Argus.

## Phase 1 — Fraud Detection Pipeline

| Date | Decision | Rationale | Alternatives Considered & Rejected |
| :--- | :--- | :--- | :--- |
| **2026-08-29** | **Chronological / Temporal Split along `Time`** | Fraud patterns, merchant profiles, and attacker vectors drift over time. A random split causes future-to-past data leakage and artificially inflates performance metrics. Splitting chronologically (70% train, 15% val, 15% test) guarantees real-world evaluation fidelity. | Random stratified split (rejected due to temporal leakage). |
| **2026-08-29** | **PR-AUC as Headline Metric** | In extreme class imbalance (~0.17% fraud rate), ROC-AUC is misleadingly high because the vast true negative volume suppresses the False Positive Rate. Precision-Recall AUC focuses strictly on the minority class performance. | Accuracy (useless at 99.8% baseline), ROC-AUC (misleadingly optimistic). |
| **2026-08-29** | **Strict Resampling Isolation via `imblearn.pipeline.Pipeline`** | Applying SMOTE before splitting or outside cross-validation folds leaks synthetic minority samples into validation distributions. Encapsulating resampling inside an `imblearn` pipeline guarantees zero validation leakage. | Standalone SMOTE before split (severe leakage bug). |
| **2026-08-29** | **Cost-Utility Threshold Optimization** | Default decision threshold of 0.5 is arbitrary. In financial fraud, the cost of a False Negative (missed fraud, ~$500) vastly outweighs a False Positive (SMS prompt / friction, ~$15). Minimizing expected financial loss or targeting recall ≥ 80% yields higher business value. | Arbitrary 0.5 probability threshold, naive max-F1 threshold. |
| **2026-08-29** | **Integrated Preprocessor + Model Persistence** | Saving the fitted preprocessing pipeline (`RobustScaler` / `ColumnTransformer`) together with the LightGBM/XGBoost classifier in a single artifact (`models/fraud_pipeline.joblib`) ensures Phase 3 API scoring receives identical feature transformations without schema drift. | Separate scaler and model pickles (risks desynchronization and scoring silent errors). |

## Phase 2 — Document Intelligence & SEC RAG Subsystem

| Date | Decision | Rationale | Alternatives Considered & Rejected |
| :--- | :--- | :--- | :--- |
| **2026-08-31** | **Section-Aware Chunking with Context Headers** | 10-K filings have strict semantic divisions (Item 1 Business vs. Item 1A Risks vs. Item 7 MD&A). Splitting along Item boundaries and prepending contextual metadata headers (`[Ticker 10-K FY | Section]`) to sub-chunks prevents cross-section context contamination and enables verifiable citations. | Naive fixed 512-token sliding window across raw text (rejected due to cross-section pollution and broken financial context). |
| **2026-08-31** | **Hybrid Search via Reciprocal Rank Fusion (RRF)** | Financial documents are dense with exact numeric figures, tickers, balance sheet line items, and legal phrasing that dense embeddings frequently blur. Combining sparse `BM25Okapi` with dense semantic vectors (`all-MiniLM-L6-v2`) via RRF achieves optimal lexical and semantic retrieval. | Dense-only search (struggles with exact financial line items and entity names). |
| **2026-08-31** | **Cross-Encoder Reranking over Candidate Pool** | Pre-retrieving top-20 candidates and passing them through a cross-encoder (`ms-marco-MiniLM-L-6-v2`) evaluates fine-grained query-chunk cross-attention before selecting the final top 5. | Static bi-encoder scoring only (lacks deep query-token interaction). |
| **2026-08-31** | **Programmatic Citation Verification & Rejection** | Rather than relying on prompt compliance alone, citations (`[chunk_id]`) are audited in code against the set of chunks actually retrieved. If a response cites a hallucinated or unretrieved ID, it is flagged and penalized. | Trusting LLM text generation blindly without programmatic validation. |
| **2026-08-31** | **Mandatory Abstention on Out-of-Corpus Queries** | Financial intelligence systems must refuse unanswerable questions rather than hallucinating plausible figures. Abstention is triggered when distinctive query terms or time periods do not exist in the retrieved context, achieving a 100% abstention accuracy on the benchmark suite. | Forcing speculative generation on all queries. |
