"""SEC 10-K Grounded Document Question-Answering Page."""

import app._bootstrap  # noqa: F401
import streamlit as st

from app.components.shared import query_rag_service

st.set_page_config(
    page_title="Argus — SEC 10-K Document Q&A",
    page_icon="📄",
    layout="wide",
)

st.title("📄 SEC 10-K Grounded Intelligence")
st.caption(
    "Query 10 corporate annual reports (13,467 section chunks) with strict verbatim grounding, "
    "programmatic citation auditing, and transparent failure inspection."
)

st.markdown("---")

# Pre-filled example queries representing key operational modes
EXAMPLES = {
    "1. Simple Factual Lookup (Apple FY2025 Net Sales)": {
        "ticker": "AAPL",
        "question": "What was Apple total net sales in fiscal 2025?",
        "note": "Standard factual extraction from Item 7 MD&A narrative.",
    },
    "2. Growth Drivers Synthesis (Netflix FY2024 Operating Income)": {
        "ticker": "NFLX",
        "question": "What was the driver of growth in Netflix's operating income in FY2024?",
        "note": "Complex multi-factor question highlighting BM25 vs Dense retrieval dynamics.",
    },
    "3. Out-of-Domain Abstention (Tesla 2023 TV Ad Spend)": {
        "ticker": "TSLA",
        "question": "What was Tesla's commercial television advertising spend in 2023?",
        "note": "Deliberately unanswerable: Tesla does not itemize commercial TV ads. Demonstrates reliable refusal.",
    },
}

col_query, col_config = st.columns([1.2, 0.8])

with col_query:
    st.subheader("1. Select or Enter Financial Query")
    example_choice = st.selectbox(
        "Choose a Pre-Engineered Example:",
        options=list(EXAMPLES.keys()),
        index=0,
    )
    example_data = EXAMPLES[example_choice]
    st.info(f"💡 **Scenario Context**: {example_data['note']}")

    user_query = st.text_input(
        "Financial Question:",
        value=example_data["question"],
        help="Type any question regarding the 10 corporate SEC 10-K filings.",
    )

with col_config:
    st.subheader("2. Retrieval Settings")
    config_choice = st.selectbox(
        "Retrieval Architecture:",
        options=[
            "section_dense (Section Chunks + Dense Embedding — Recommended)",
            "section_hybrid (Section Chunks + Dense + BM25 RRF)",
            "bm25_only (Diagnostic Exact Keyword Matching)",
            "fixed_dense (Fixed 500-token Chunks + Dense)",
        ],
        index=0,
    )
    config_key = config_choice.split()[0]

    company_filter = st.selectbox(
        "Company Filter (Optional):",
        options=["All Tickers (Cross-corpus search)", "AAPL", "NVDA", "TSLA", "NFLX", "AMZN", "MSFT", "GOOGL", "INTC", "META", "JPM"],
        index=0,
    )

st.markdown("---")

# Execution button
if st.button("Run Grounded Retrieval & Answer Synthesis", type="primary"):
    with st.spinner("Executing dense vector retrieval across 13,467 chunks and synthesizing grounded answer..."):
        # Augment query with company filter hint if specified
        final_query = user_query
        if company_filter != "All Tickers (Cross-corpus search)":
            final_query = f"{company_filter}: {user_query}"

        res = query_rag_service(final_query, config_name=config_key)

    st.subheader("3. Synthesized Answer & Programmatic Citation Audit")

    if res.get("abstained"):
        st.warning(
            f"🛑 **MODEL ABSTAINED**: {res['answer']}\n\n"
            "The system refused to answer because the retrieved chunks did not contain "
            "verbatim figures directly answering the prompt. In production financial intelligence, "
            "abstention is a critical safety feature preventing hallucinations."
        )
    else:
        st.success(f"**Answer**: {res['answer']}")

    # Citation Verification Telemetry
    verif = res.get("verification", {})
    cited_ids = verif.get("cited_ids", [])
    hallucinated_ids = verif.get("hallucinated_ids", [])
    is_valid = verif.get("is_valid", True)

    c_audit1, c_audit2, c_audit3 = st.columns(3)
    with c_audit1:
        st.metric("Citations Found", len(cited_ids))
    with c_audit2:
        st.metric("Hallucinated Chunk IDs", len(hallucinated_ids))
    with c_audit3:
        if is_valid and not hallucinated_ids:
            st.metric("Programmatic Verification", "100% Valid ✅")
        else:
            st.metric("Programmatic Verification", "Failed ❌")

    st.markdown("---")

    # Transparent Context Inspection — ALWAYS shown, even on wrong answer or abstention
    st.subheader("4. Transparent Retrieval Inspection: Retrieved Passage Chunks")
    st.write(
        "Below are the top 5 passages surfaced by the retrieval index. "
        "Examining these excerpts allows you to diagnose whether a failure was due to "
        "retrieving the wrong context (e.g. financial tables instead of narrative) vs. generation error."
    )

    citations = res.get("citations", [])
    if not citations:
        st.write("No candidate chunks returned.")
    else:
        for c in citations:
            rank = c.get("rank", 1)
            cid = c.get("chunk_id", "UNKNOWN")
            score = c.get("score", 0.0)
            ticker = c.get("ticker", "N/A")
            sec = c.get("section", "N/A")
            is_cited = cid in cited_ids

            badge = "🔖 [CITED IN ANSWER]" if is_cited else ""
            with st.expander(f"Rank {rank} | [{cid}] — {ticker} ({sec}) | Score: {score:.4f} {badge}"):
                st.markdown(f"**Chunk ID**: `{cid}` | **Filing**: `{ticker} 10-K ({c.get('fiscal_year', 2024)})` | **Section**: `{sec}`")
                st.code(c.get("text", ""), language="markdown")

st.markdown("---")
st.caption("Argus SEC 10-K RAG • Audited against 13,467 annual report passages.")
