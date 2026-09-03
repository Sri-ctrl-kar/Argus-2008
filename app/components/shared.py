"""Shared utilities, model artifact caching, SHAP explainers, and evaluation loaders for Argus."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Root directories
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
PRESETS_FILE = APP_DIR / "components" / "presets.json"


@st.cache_resource(show_spinner="Loading fraud detection model artifacts...")
def load_fraud_artifacts() -> Tuple[Any, Dict[str, Any], Any]:
    """Load serialized LightGBM fraud pipeline, metadata, and pre-fit SHAP TreeExplainer."""
    pipeline_path = MODELS_DIR / "fraud_pipeline.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"

    if not pipeline_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Model artifacts missing from {MODELS_DIR}")

    pipeline = joblib.load(pipeline_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Extract underlying classifier for SHAP TreeExplainer
    clf = pipeline.named_steps.get("classifier", pipeline)
    explainer = shap.TreeExplainer(clf)
    return pipeline, meta, explainer


@st.cache_data
def get_presets() -> Dict[str, Any]:
    """Load canonical test set presets representing key operational behaviors."""
    if PRESETS_FILE.exists():
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


from imblearn.base import SamplerMixin


def compute_shap_waterfall(
    row_dict: Dict[str, float],
    meta: Dict[str, Any],
    pipeline: Any,
    explainer: Any,
    max_display: int = 10,
) -> plt.Figure:
    """Generate a clean SHAP waterfall figure for a single transaction row."""
    features = meta["features"]
    df_row = pd.DataFrame([row_dict])[features]

    # Preprocess row through pipeline transformers (excluding classifier and samplers)
    preprocessed = df_row
    if hasattr(pipeline, "steps"):
        for name, step in pipeline.steps[:-1]:  # everything but the classifier
            if isinstance(step, SamplerMixin):  # SMOTE and friends: training only
                continue
            preprocessed = step.transform(preprocessed)
    elif hasattr(pipeline, "named_steps"):
        for name, step in pipeline.named_steps.items():
            if name != "classifier":
                if isinstance(step, SamplerMixin):
                    continue
                preprocessed = step.transform(preprocessed)
    else:
        preprocessed = df_row.values

    shap_values = explainer(preprocessed)
    # Binary classification: select positive fraud class index
    if len(shap_values.values.shape) == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    # Create figure with high DPI and dark/light mode compatibility
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    plt.sca(ax)
    shap.plots.waterfall(sv, max_display=max_display, show=False)
    plt.title("SHAP Waterfall: What Drove This Risk Score?", fontsize=13, pad=15, weight="bold")
    plt.tight_layout()
    return fig



@st.cache_data
def load_ablation_metrics() -> List[Dict[str, Any]]:
    """Read committed RAG ablation metrics from eval/results/ablation.json."""
    ablation_path = EVAL_RESULTS_DIR / "ablation.json"
    if ablation_path.exists():
        with open(ablation_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def load_latency_metrics() -> Dict[str, Any]:
    """Read committed API latency benchmark metrics from eval/results/api_latency.json."""
    latency_path = EVAL_RESULTS_DIR / "api_latency.json"
    if latency_path.exists():
        with open(latency_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache_resource(show_spinner="Initializing RAG Retriever & Vector Store...")
def get_cached_retriever(config_name: str):
    from src.rag.retrieve import RAGRetriever
    use_hybrid = "hybrid" in config_name or config_name == "bm25_only"
    use_rerank = "rerank" in config_name
    strategy = "fixed_size" if "fixed" in config_name else "section_aware"
    return RAGRetriever(strategy=strategy, use_hybrid=use_hybrid, use_reranker=use_rerank)

@st.cache_resource
def get_cached_generator():
    from src.rag.generate import GroundedGenerator
    return GroundedGenerator()

def query_rag_service(query: str, config_name: str = "section_dense") -> Dict[str, Any]:
    """Execute SEC 10-K RAG query with in-process fallback and graceful failure degradation."""
    try:
        retriever = get_cached_retriever(config_name)
        results = retriever.retrieve(query, top_k=5)
        generator = get_cached_generator()
        ans = generator.generate_answer(query, results)


        return {
            "answer": ans.response_text,
            "abstained": ans.abstained,
            "citations": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "ticker": r.chunk.ticker,
                    "fiscal_year": r.chunk.fiscal_year,
                    "section": r.chunk.section_name,
                    "text": r.chunk.text,
                    "rank": r.rank,
                    "score": round(r.score, 4),
                }
                for r in results
            ],
            "config": config_name,
            "verification": ans.verification.to_dict(),
        }
    except Exception as e:
        return {
            "answer": f"Retrieval / generation error: {str(e)}",
            "abstained": True,
            "citations": [],
            "config": config_name,
            "verification": {"is_valid": False, "hallucinated_ids": []},
            "error": str(e),
        }
