"""Ablation diagnostic.

Answers one question: are the four configurations actually doing different
things, or did they all run the same code path?

Adapt ONLY the `retrieve_for_config` function below to call your retriever.
Everything else works as-is.

Run:  python -m eval.diagnose
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# ADAPT THIS FUNCTION ONLY
# ---------------------------------------------------------------------------

CONFIGS = [
    "fixed_dense",
    "section_dense",
    "section_hybrid",
    "section_hybrid_rerank",
]


def retrieve_for_config(config_name: str, question: str, k: int = 5) -> list[str]:
    """Return the ordered chunk IDs your pipeline retrieves for this config.

    Replace the body with however your code builds a retriever. The point is
    that this must go through the SAME entry point your ablation harness
    uses -- if you construct the retriever differently here, the diagnostic
    tests the wrong thing.
    """
    from src.rag import retrieve  # noqa: F401  <-- your module

    retriever = retrieve.build(config_name)          # <-- your builder
    results = retriever.retrieve(question, k=k)      # <-- your call
    return [r.chunk_id for r in results]             # <-- your ID field


# ---------------------------------------------------------------------------
# Everything below is generic
# ---------------------------------------------------------------------------

QUESTIONS_PATH = Path("eval/questions.jsonl")
OUT_PATH = Path("eval/results/diagnostic.json")


def load_questions(n: int = 10) -> list[dict]:
    rows = [json.loads(line) for line in QUESTIONS_PATH.read_text().splitlines() if line.strip()]
    answerable = [r for r in rows if r.get("answerable", True)]
    return answerable[:n]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


# --- Check 1: do configs return different chunks? --------------------------


def check_configs_differ(questions: list[dict]) -> dict:
    print("=" * 70)
    print("CHECK 1  Do the four configs retrieve different chunks?")
    print("=" * 70)

    per_question = []
    for q in questions:
        retrieved = {c: retrieve_for_config(c, q["question"]) for c in CONFIGS}
        base = retrieved[CONFIGS[0]]
        overlaps = {c: jaccard(base, retrieved[c]) for c in CONFIGS[1:]}
        per_question.append({"question": q["question"], "retrieved": retrieved, "overlap_vs_base": overlaps})

    print(f"\n{'config':<32} {'mean Jaccard vs fixed_dense':>28}")
    print("-" * 62)
    identical_count = 0
    for c in CONFIGS[1:]:
        vals = [p["overlap_vs_base"][c] for p in per_question]
        mean = sum(vals) / len(vals)
        flag = "  <-- IDENTICAL" if mean > 0.999 else ""
        if mean > 0.999:
            identical_count += 1
        print(f"{c:<32} {mean:>28.3f}{flag}")

    print("\nExample, first question:")
    print(f"  {per_question[0]['question'][:70]}")
    for c in CONFIGS:
        ids = per_question[0]["retrieved"][c]
        print(f"  {c:<30} {ids}")

    if identical_count == len(CONFIGS) - 1:
        print(
            "\nVERDICT: all configs return the same chunks. Your config flag is\n"
            "not reaching the retriever -- every ablation row ran identical code.\n"
            "Fix the wiring before interpreting any metric."
        )
    elif identical_count:
        print(f"\nVERDICT: {identical_count} config(s) are not distinct. Check those specifically.")
    else:
        print("\nVERDICT: configs are genuinely different. The problem is the eval set -> see Check 2.")

    return {"per_question": per_question, "identical_configs": identical_count}


# --- Check 2: is the eval set too easy? ------------------------------------


def check_question_difficulty(questions: list[dict]) -> dict:
    print("\n" + "=" * 70)
    print("CHECK 2  Are the questions too easy? (lexical overlap with answer)")
    print("=" * 70)

    def tokens(s: str) -> set[str]:
        stop = {"the", "a", "an", "of", "in", "for", "to", "and", "was", "is",
                "what", "how", "did", "does", "which", "on", "at", "by", "with"}
        return {w.strip(".,?()$%").lower() for w in s.split()} - stop

    rows = []
    for q in questions:
        qt, at = tokens(q["question"]), tokens(str(q.get("answer", "")))
        overlap = len(qt & at) / len(qt) if qt else 0.0
        rows.append((overlap, q["question"]))

    rows.sort(reverse=True)
    mean = sum(r[0] for r in rows) / len(rows)

    print(f"\nmean question/answer token overlap: {mean:.2f}")
    print("\nhighest-overlap questions (these are the too-easy ones):")
    for ov, question in rows[:5]:
        print(f"  {ov:.2f}  {question[:66]}")

    if mean > 0.35:
        print(
            "\nVERDICT: high lexical overlap. Your questions reuse the filing's own\n"
            "wording, so any retriever wins and no ablation can show a difference.\n"
            "Rewrite the high-overlap questions using an analyst's phrasing, not\n"
            "the document's."
        )
    else:
        print("\nVERDICT: overlap looks reasonable. Difficulty is probably not the issue.")

    return {"mean_overlap": mean, "ranked": rows}


# --- Check 3: recall at stricter k -----------------------------------------


def check_recall_headroom(questions: list[dict]) -> dict:
    print("\n" + "=" * 70)
    print("CHECK 3  Is Recall@5 saturated? (recompute at k=1 and k=3)")
    print("=" * 70)

    out = {}
    for k in (1, 3, 5):
        hits = {c: 0 for c in CONFIGS}
        for q in questions:
            gold = q.get("source_chunk_id") or q.get("source_doc")
            if gold is None:
                continue
            for c in CONFIGS:
                ids = retrieve_for_config(c, q["question"], k=k)
                if any(gold in str(i) for i in ids):
                    hits[c] += 1
        out[k] = {c: hits[c] / len(questions) for c in CONFIGS}

    print(f"\n{'config':<32} {'R@1':>8} {'R@3':>8} {'R@5':>8}")
    print("-" * 58)
    for c in CONFIGS:
        print(f"{c:<32} {out[1][c]:>8.3f} {out[3][c]:>8.3f} {out[5][c]:>8.3f}")

    spread = max(out[1].values()) - min(out[1].values())
    if spread > 0.02:
        print(
            f"\nVERDICT: configs separate at k=1 (spread {spread:.3f}) even though they\n"
            "tie at k=5. Report Recall@1 and Recall@3 -- k=5 was hiding the signal."
        )
    else:
        print("\nVERDICT: no separation even at k=1. Combine with Check 1 and 2 verdicts.")

    return out


# --- Check 4: corpus size --------------------------------------------------


def check_corpus_size() -> dict:
    print("\n" + "=" * 70)
    print("CHECK 4  How big is the haystack?")
    print("=" * 70)
    try:
        from src.rag import index  # noqa

        store = index.load()               # <-- adapt if named differently
        n = store.count()
        print(f"\ntotal chunks indexed: {n}")
        if n < 500:
            print(
                "\nVERDICT: small corpus. Retrieving 5 from this pool is close to\n"
                "trivial. Add filings or chunk more finely to create real competition\n"
                "between candidates."
            )
        else:
            print("\nVERDICT: corpus size is adequate.")
        return {"n_chunks": n}
    except Exception as exc:
        print(f"\ncould not read index automatically ({exc}).")
        print("Check the chunk count manually -- under ~500 means the task is too easy.")
        return {}


def main() -> None:
    questions = load_questions()
    print(f"running diagnostics on {len(questions)} answerable questions\n")

    results = {
        "configs_differ": check_configs_differ(questions),
        "difficulty": check_question_difficulty(questions),
        "recall_headroom": check_recall_headroom(questions),
        "corpus": check_corpus_size(),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        "identical_configs": results["configs_differ"]["identical_configs"],
        "mean_question_answer_overlap": results["difficulty"]["mean_overlap"],
        "recall_at_k": results["recall_headroom"],
        "corpus": results["corpus"],
    }
    OUT_PATH.write_text(json.dumps(serialisable, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
