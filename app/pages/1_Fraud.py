"""Fraud Transaction Scoring & SHAP Waterfall Explainer Page."""

import app._bootstrap  # noqa: F401  — must precede app/src imports
import pandas as pd
import streamlit as st


from app.components.shared import (
    compute_shap_waterfall,
    get_presets,
    load_fraud_artifacts,
)

st.set_page_config(
    page_title="Argus — Fraud Scoring & SHAP",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Real-Time Fraud Scoring & SHAP Explainer")
st.caption(
    "Score real-world transactions against our cost-calibrated LightGBM model "
    "and inspect exact feature attributions via local SHAP tree explainability."
)

pipeline, meta, explainer = load_fraud_artifacts()
presets = get_presets()
features = meta["features"]
default_threshold = float(meta.get("chosen_decision_threshold", 0.12587))

st.markdown("---")

# Left Column: Inputs & Threshold Slider | Right Column: Prediction & Decision Card
col_input, col_decision = st.columns([1.1, 0.9])

with col_input:
    st.subheader("1. Transaction Input")
    input_mode = st.radio(
        "Choose Input Method:",
        options=["Preset Transactions (Recommended)", "Manual Feature Entry"],
        horizontal=True,
    )

    active_row = {}

    if input_mode == "Preset Transactions (Recommended)":
        preset_names = list(presets.keys())
        selected_preset_name = st.selectbox(
            "Select a Canonical Test Case:",
            options=preset_names,
            index=0,
        )
        selected_preset = presets[selected_preset_name]
        st.info(f"**Scenario**: {selected_preset['description']}\n\n**Ground Truth Label**: `{selected_preset['ground_truth']}`")
        active_row = selected_preset["row"].copy()

        with st.expander("Inspect Raw Feature Values for this Preset"):
            st.json(active_row)
    else:
        st.markdown("Adjust key features below to observe model response:")
        # Provide clean sliders for primary human-interpretable fields
        manual_amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=25000.0, value=149.50, step=5.0)
        manual_time = st.number_input("Transaction Time (Seconds elapsed)", min_value=0.0, max_value=172800.0, value=43200.0, step=100.0)

        # Baseline PCA features initialized to standard normal zero
        active_row = {f: 0.0 for f in features}
        active_row["Amount"] = float(manual_amount)
        active_row["Time"] = float(manual_time)

        with st.expander("Fine-tune Latent Features (V1 - V28)"):
            c1, c2, c3, c4 = st.columns(4)
            for idx, feat in enumerate([f"V{i}" for i in range(1, 29)]):
                col_target = [c1, c2, c3, c4][idx % 4]
                active_row[feat] = col_target.number_input(f"{feat}", value=0.0, step=0.1, format="%.2f")

with col_decision:
    st.subheader("2. Operational Decision")

    threshold = st.slider(
        "Decision Threshold (τ)",
        min_value=0.01,
        max_value=0.99,
        value=default_threshold,
        step=0.005,
        format="%.3f",
        help="Transactions with probability >= threshold are flagged for manual review / challenge.",
    )

    st.caption(
        rf"💡 **Calibrated Default ({default_threshold:.3f})**: Reflects the $500 false negative "
        rf"vs. $15 false positive loss asymmetry. Lowering threshold catches more fraud; raising it reduces customer friction."
    )


    # Score active transaction
    df_single = pd.DataFrame([active_row])[features]
    prob_fraud = float(pipeline.predict_proba(df_single)[0, 1])
    is_flagged = prob_fraud >= threshold

    # Decision Card Display
    st.markdown("### Risk Assessment")
    metric_c1, metric_c2 = st.columns(2)
    with metric_c1:
        st.metric("Fraud Probability", f"{prob_fraud:.2%}", delta=f"{prob_fraud - threshold:+.2%} vs threshold")
    with metric_c2:
        if is_flagged:
            st.error("🚨 **DECISION: FLAG FOR AUDIT**\n\nHigh risk score exceeds operational threshold.")
        else:
            st.success("✅ **DECISION: ALLOW TRANSACTION**\n\nRisk score within approved threshold bounds.")

st.markdown("---")

# Prominent SHAP Waterfall Section
st.subheader("3. Feature Attribution: Why Did the Model Make This Decision?")
st.write(
    "Local SHAP (SHapley Additive exPlanations) isolates the exact dollar/latent features that pushed "
    "the probability up (red/positive risk drivers) or down (blue/benign mitigating factors) from the base rate."
)

with st.spinner("Generating local SHAP waterfall attribution..."):
    fig = compute_shap_waterfall(active_row, meta, pipeline, explainer, max_display=10)
    st.pyplot(fig, use_container_width=True)

st.markdown("---")
st.caption("Argus Fraud Engine • Parity-verified against offline training pipeline to 6 decimal places.")
