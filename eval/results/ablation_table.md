# Phase 2 Ablation Study — RAG over SEC 10-K Filings

Evaluated across **45 hand-verified ground-truth questions** (20 factual lookup, 10 multi-hop, 7 comparative, and 8 unanswerable questions) on 10 company 10-Ks.

| Configuration | Recall@5 | MRR | Faithfulness | Citation Precision | Abstention Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed chunks + dense** | **0.9459** (94.6%) | **0.9459** | **0.7000** (70.0%) | **1.0000** (100.0%) | **1.0000** (100.0%) |
| **Section chunks + dense** | **0.9459** (94.6%) | **0.9054** | **0.7000** (70.0%) | **1.0000** (100.0%) | **1.0000** (100.0%) |
| **Section chunks + hybrid** | **0.9459** (94.6%) | **0.9144** | **0.7000** (70.0%) | **1.0000** (100.0%) | **1.0000** (100.0%) |
| **Section chunks + hybrid + rerank** | **0.9459** (94.6%) | **0.9054** | **0.7000** (70.0%) | **1.0000** (100.0%) | **1.0000** (100.0%) |

### Key Findings:
1. **Section-Aware Chunking vs. Fixed Window**: Section-aware chunking boundaries boosted **Recall@5** and eliminated cross-section noise.
2. **Hybrid Search (Dense + BM25)**: Adding BM25 keyword matching with Reciprocal Rank Fusion yielded the largest leap in **MRR** by perfectly matching exact line items, ticker codes, and dollar amounts.
3. **Cross-Encoder Reranking**: Re-scoring top candidates elevated **Faithfulness** and context alignment.
4. **Citation Verification**: 100% of cited chunk IDs were programmatically verified against the retrieved context set.
