"""LLM Interface with LRU caching for grounded generation."""

from __future__ import annotations

import os
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2048)
def _cached(system: str, prompt: str) -> str:
    """Cached on (system, prompt) so ablation re-runs are near-free.

    Note the key includes the prompt, which includes the retrieved contexts.
    Keying on the question alone would make every config return identical
    answers and faithfulness could not move.
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0,          # determinism matters for reproducible evals
    )
    return resp.choices[0].message.content or ""


def call_llm(system: str, prompt: str) -> str:
    """Public entrypoint that checks for OPENAI_API_KEY and calls cached completion."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return _cached(system, prompt)
        except Exception as e:
            logger.warning(f"OpenAI completion call failed ({e}), falling back to deterministic extraction.")
            return ""
    return ""
