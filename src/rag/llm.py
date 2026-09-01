"""LLM Interface with LRU caching for grounded generation."""

from __future__ import annotations

import os
import logging
import warnings
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2048)
def _cached(system: str, prompt: str, base_url: str = "", model_name: str = "") -> str:
    """Cached on (system, prompt, base_url, model_name) so ablation re-runs are near-free.

    Note the key includes the prompt, which includes the retrieved contexts.
    Keying on the question alone would make every config return identical
    answers and faithfulness could not move.
    """
    from openai import OpenAI

    if base_url:
        client = OpenAI(base_url=base_url, api_key="ollama")
        model = model_name or "llama3.1:latest"
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = "gpt-4o-mini"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0,          # determinism matters for reproducible evals
    )
    return resp.choices[0].message.content or ""


def _is_ollama_available() -> bool:
    """Check if local Ollama daemon is running."""
    import urllib.request
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def call_llm(system: str, prompt: str) -> str:
    """Public entrypoint that routes to OpenAI, local Ollama (Llama 3.1), or falls back to extractive mode."""
    # 1. Check for OpenAI API Key
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return _cached(system, prompt, base_url="", model_name="gpt-4o-mini")
        except Exception as e:
            logger.warning(f"OpenAI call failed ({e}), checking local Ollama fallback...")

    # 2. Check for local Ollama (Llama 3.1)
    if _is_ollama_available():
        try:
            return _cached(system, prompt, base_url="http://127.0.0.1:11434/v1", model_name="llama3.1:latest")
        except Exception as e:
            logger.warning(f"Local Ollama generation failed ({e}), falling back to deterministic extraction.")

    # 3. Warning on offline extractive fallback
    warnings.warn(
        "No OpenAI API key or Ollama server — falling back to extractive mode. "
        "Metrics from this run are NOT generation metrics.",
        RuntimeWarning,
        stacklevel=2,
    )
    return ""
