"""Argus Financial Intelligence Platform — Streamlit Landing & System Overview."""

import streamlit as st

st.set_page_config(
    page_title="Argus — Financial Intelligence Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏛️ Argus: Financial Intelligence Platform")
st.caption(
    "Dual-engine financial infrastructure combining real-time asymmetric fraud risk scoring "
    "with grounded SEC 10-K document retrieval."
)

st.markdown("---")

# Quick Navigation and Overview Cards
col1, col2 = st.columns(2)

with col1:
    st.subheader("💳 Phase 1: Asymmetric Fraud Detection")
    st.markdown(
        r"""
        - **Model Architecture**: Cost-sensitive LightGBM trained on 284,807 credit card transactions.
        - **Operational Threshold**: Calibrated at **0.126** under an asymmetric cost matrix ($500 false negative fraud loss vs. $15 false positive customer verification cost).
        - **Decision Transparency**: Real-time **SHAP waterfall plots** computed on every scored transaction.
        - **Serving Latency**: **3.6ms** median single-transaction scoring in FastAPI.
        """
    )
    st.page_link("pages/1_Fraud.py", label="Open Transaction Scoring & SHAP Explainer →", icon="💳")

with col2:
    st.subheader("📄 Phase 2: Grounded SEC 10-K RAG")
    st.markdown(
        """
        - **Corpus**: 10 corporate annual reports (Apple, NVIDIA, Tesla, Amazon, Microsoft, Alphabet, etc.) parsed into **13,467 section-aware chunks**.
        - **Retrieval Pipeline**: Hybrid dense vector embeddings (`all-MiniLM-L6-v2`) + exact token BM25 with Reciprocal Rank Fusion (RRF).
        - **Programmatic Auditing**: Every generated claim extracts citations and programmatically verifies presence in retrieved candidate set (0% hallucination rate).
        - **Deterministic Abstention**: Explicitly refuses unanswerable and out-of-domain queries.
        """
    )
    st.page_link("pages/2_Filings.py", label="Open SEC 10-K Document Q&A →", icon="📄")

st.markdown("---")

# Upfront System Limitations — Stated prominently on front screen
st.header("⚠️ Upfront Limitations & System Bounds")
st.info(
    "Argus is engineered for production transparency. A system that states its operating "
    "boundaries up front is reliable; one that conceals them overclaims.",
    icon="ℹ️",
)

lim_col1, lim_col2, lim_col3 = st.columns(3)

with lim_col1:
    st.markdown("#### 1. Retrieval Bottleneck")
    st.write(
        "Across 13,467 corporate 10-K chunks, dense retrieval surfaces the correct passage "
        "roughly **24.3% of the time at Rank 1 (Recall@1)** and **40.5% in the top 5 (Recall@5)**. "
        "Generation quality is bounded by retrieval: the system reliably answers from its context, "
        "but often receives the wrong context (e.g. balance sheet tables instead of MD&A narratives)."
    )

with lim_col2:
    st.markdown("#### 2. Cost Matrix Tradeoff")
    st.write(
        "Because an undetected fraud ($500) is 33.3× more expensive than a false alert ($15), "
        "the decision threshold is pegged at **0.126** rather than standard 0.50. This intentionally "
        "trades precision for recall to intercept >80% of fraud volume, producing benign false positives "
        "on unusual transactions."
    )


with lim_col3:
    st.markdown("#### 3. Eval Set Statistical Power")
    st.write(
        "The ground-truth RAG ablation set contains **37 answerable and 8 unanswerable questions**. "
        "Confidence intervals on every configuration overlap. This benchmark set can rank retrieval "
        "strategies weakly (MRR 0.189–0.309) and cannot statistically rank generator quality."
    )

st.markdown("---")

# Architecture & Verified Bug Regressions
st.header("🏗️ Engineering Integrity")

st.markdown(
    """
    Argus implements regression suites protecting against 5 critical measurement vulnerabilities:
    1. **Zero Temporal Leakage**: Enforced by time-ordered splits with purge windows; random k-fold produces artificially inflated PR-AUC.
    2. **Zero Preprocessing Leakage**: Scalers and SMOTE resamplers strictly fit within training folds only.
    3. **Zero Skew Parity**: FastAPI serving pipeline matches offline training probabilities to **6 decimal places**.
    4. **Programmatic Citation Verification**: Verifies extracted `[chunk_id]` tags exist in the retrieved context pool.
    5. **Explicit Abstention Metric**: Evaluated on genuine unanswerable queries to measure both hallucination resistance and over-abstention.
    """
)

st.page_link("pages/3_Evaluation.py", label="View Live Empirical Evaluation & Audit Tables →", icon="📊")

st.markdown("---")
st.caption("Argus Platform • Open Source Repository: [GitHub](https://github.com/Sri-ctrl-kar/Argus-2008)")
