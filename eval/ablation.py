"""Ablation harness.

Runs every retrieval configuration against the eval set and writes a
committed results table. This is the primary deliverable of Phase 2 -- the
chatbot is a demo, the table is the evidence.

Adapt the ADAPTER section only.

Run:  python -m eval.ablation
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

# ===========================================================================
# ADAPTER -- point these at your code
# ===========================================================================

CONFIGS = {
    "bm25_only": "BM25 only (diagnostic)",
    "fixed_dense": "Fixed chunks + dense",
    "section_dense": "Section chunks + dense",
    "section_hybrid": "Section chunks + hybrid (BM25 + dense)",
    "section_hybrid_rerank": "Section chunks + hybrid + rerank",
}


K_VALUES = (1, 3, 5)


def build_retriever(config_name: str):
    from src.rag import retrieve

    return retrieve.build(config_name)


def do_retrieve(retriever, question: str, k: int) -> list[dict]:
    """Return [{'id': str, 'text': str}, ...] in rank order."""
    results = retriever.retrieve(question, k=k)
    return [{"id": r.chunk_id, "text": r.text} for r in results]


def do_generate(question: str, contexts: list[dict]) -> dict:
    """Return {'answer': str, 'cited_ids': [str], 'abstained': bool}."""
    from src.rag import generate

    return generate.answer(question, contexts)


# ===========================================================================
# Generic below this line
# ===========================================================================

QUESTIONS = Path("eval/questions.jsonl")
RESULTS_DIR = Path("eval/results")
MAX_K = max(K_VALUES)


def load_questions() -> list[dict]:
    rows = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    missing = [r for r in rows if r.get("answerable", True) and not r.get("gold_span")]
    if missing:
        raise ValueError(
            f"{len(missing)} answerable questions lack a 'gold_span'. Scoring on "
            "document match instead of span containment measures 'found the right "
            "company', not 'found the right passage'. Add gold_span first."
        )
    return rows


def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def span_hit(gold_span: str, contexts: list[dict]) -> bool:
    """True if any retrieved chunk contains the gold answer span.

    Span containment is used rather than chunk-ID matching because chunk IDs
    differ across chunking strategies -- ID-based gold labels would have to
    be re-annotated for every config, which makes a chunking ablation
    impossible to run honestly.
    """
    needle = _norm(gold_span)
    return any(needle in _norm(c["text"]) for c in contexts)


def rank_of_hit(gold_span: str, contexts: list[dict]) -> int | None:
    needle = _norm(gold_span)
    for i, c in enumerate(contexts, start=1):
        if needle in _norm(c["text"]):
            return i
    return None


def run_config(name: str, questions: list[dict]) -> dict:
    print(f"\n--- {name} " + "-" * (60 - len(name)))
    retriever = build_retriever(name)

    answerable = [q for q in questions if q.get("answerable", True)]
    unanswerable = [q for q in questions if not q.get("answerable", True)]

    hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    generations: list[dict] = []

    for q in answerable:
        t0 = time.perf_counter()
        contexts = do_retrieve(retriever, q["question"], MAX_K)
        latencies.append((time.perf_counter() - t0) * 1000)

        for k in K_VALUES:
            if span_hit(q["gold_span"], contexts[:k]):
                hits[k] += 1

        rank = rank_of_hit(q["gold_span"], contexts)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        gen = do_generate(q["question"], contexts)
        retrieved_ids = {c["id"] for c in contexts}
        generations.append(
            {
                "question": q["question"],
                "answer": gen["answer"],
                "contexts": [c["text"] for c in contexts],
                "ground_truth": q.get("answer", ""),
                "abstained": gen.get("abstained", False),
                # A citation is valid only if it points at a chunk actually
                # retrieved. Asking the model to cite and trusting it is not
                # verification.
                "citations_valid": all(
                    cid in retrieved_ids for cid in gen.get("cited_ids", [])
                ),
                "n_citations": len(gen.get("cited_ids", [])),
            }
        )

    # Abstention is measured on both subsets. Reporting only the
    # unanswerable rate hides over-abstention, which is a real cost.
    abstained_correctly = 0
    for q in unanswerable:
        contexts = do_retrieve(retriever, q["question"], MAX_K)
        gen = do_generate(q["question"], contexts)
        abstained_correctly += bool(gen.get("abstained", False))

    n = len(answerable)
    cited = [g for g in generations if g["n_citations"] > 0]

    result = {
        "config": name,
        "label": CONFIGS[name],
        "n_answerable": n,
        "n_unanswerable": len(unanswerable),
        **{f"recall@{k}": hits[k] / n for k in K_VALUES},
        "mrr": statistics.mean(reciprocal_ranks),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_p95": sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "citation_precision": (
            sum(g["citations_valid"] for g in cited) / len(cited) if cited else None
        ),
        "abstention_on_unanswerable": (
            abstained_correctly / len(unanswerable) if unanswerable else None
        ),
        "over_abstention_on_answerable": (
            sum(g["abstained"] for g in generations) / n
        ),
    }

    ragas = score_with_ragas(generations)
    result.update(ragas)

    print(
        f"  R@1 {result['recall@1']:.3f}  R@5 {result['recall@5']:.3f}  "
        f"MRR {result['mrr']:.3f}  {result['latency_ms_median']:.0f}ms"
    )
    return result


def score_with_ragas(generations: list[dict]) -> dict:
    """RAGAS faithfulness and relevancy. Skipped cleanly if unavailable."""
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError:
        print("  (ragas not installed -- generation metrics skipped)")
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

    scored = [g for g in generations if not g["abstained"]]
    if not scored:
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

    try:
        ds = Dataset.from_dict(
            {
                "question": [g["question"] for g in scored],
                "answer": [g["answer"] for g in scored],
                "contexts": [g["contexts"] for g in scored],
                "ground_truth": [g["ground_truth"] for g in scored],
            }
        )
        scores = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
        return {
            "faithfulness": float(scores["faithfulness"]) if "faithfulness" in scores else None,
            "answer_relevancy": float(scores["answer_relevancy"]) if "answer_relevancy" in scores else None,
            "context_precision": float(scores["context_precision"]) if "context_precision" in scores else None,
        }
    except Exception as e:
        print(f"  (ragas evaluation skipped: {e})")
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None}



def to_markdown(rows: list[dict]) -> str:
    def cell(v):
        if v is None:
            return "n/a"
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    header = (
        "| Configuration | R@1 | R@3 | R@5 | MRR | Faithfulness | "
        "Citation prec. | Abstention (unans.) | Median latency |"
    )
    sep = "|" + "---|" * 9
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['label']} | {cell(r['recall@1'])} | {cell(r['recall@3'])} | "
            f"{cell(r['recall@5'])} | {cell(r['mrr'])} | {cell(r.get('faithfulness'))} | "
            f"{cell(r.get('citation_precision'))} | "
            f"{cell(r.get('abstention_on_unanswerable'))} | "
            f"{r['latency_ms_median']:.0f}ms |"
        )
    return "\n".join(lines)


def sanity_check(rows: list[dict]) -> None:
    """Flag the failure modes that produce a valid-looking, meaningless table."""
    print("\n" + "=" * 70)
    warnings = []

    r1 = [r["recall@1"] for r in rows]
    if max(r1) - min(r1) < 0.01:
        warnings.append(
            "All configs have near-identical Recall@1. Either the configs are "
            "not distinct, or the eval set is too easy to separate them."
        )
    if min(r1) > 0.95:
        warnings.append(
            "Recall@1 above 0.95 on a 13k-chunk corpus is implausibly high. "
            "Check that gold_span matching is not trivially satisfied."
        )
    faith = [r.get("faithfulness") for r in rows if r.get("faithfulness") is not None]
    if len(set(faith)) == 1 and len(faith) > 1:
        warnings.append(
            "Identical faithfulness across configs suggests generation results "
            "are cached on question alone. Key the cache on question + contexts."
        )

    if warnings:
        print("SANITY WARNINGS")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("Sanity checks passed -- configs separate and values look plausible.")
    print("=" * 70)


def main() -> None:
    questions = load_questions()
    print(f"{len(questions)} eval questions loaded")

    rows = [run_config(name, questions) for name in CONFIGS]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ablation.json").write_text(json.dumps(rows, indent=2))
    table = to_markdown(rows)
    (RESULTS_DIR / "ablation_table.md").write_text(table + "\n")

    print("\n" + table)
    sanity_check(rows)
    print(f"\nwrote {RESULTS_DIR}/ablation.json and ablation_table.md")


if __name__ == "__main__":
    main()
