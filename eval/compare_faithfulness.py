"""Compare faithfulness gaps between section_dense and bm25_only."""

import json

rows = json.load(open("eval/results/ablation.json"))
by_config = {r["config"]: {p["question"]: p for p in r.get("_per_question", [])} for r in rows}

if "section_dense" in by_config and "bm25_only" in by_config and by_config["section_dense"] and by_config["bm25_only"]:
    dense, bm25 = by_config["section_dense"], by_config["bm25_only"]
    gaps = sorted(
        ((bm25[q]["faithfulness"] - dense[q]["faithfulness"], q) for q in dense if q in bm25),
        reverse=True,
    )

    for gap, q in gaps[:3]:
        print("=" * 70)
        print(f"{q}\n  bm25 faithful: {bm25[q]["faithfulness"]:.2f}   dense: {dense[q]["faithfulness"]:.2f}")
        print(f"\n  BM25 answer:  {bm25[q]["answer"][:200]}")
        bm25_ctx = bm25[q]["contexts"][0][:200] if bm25[q]["contexts"] else "None"
        print(f"  BM25 context: {bm25_ctx}")
        print(f"\n  DENSE answer:  {dense[q]["answer"][:200]}")
        dense_ctx = dense[q]["contexts"][0][:200] if dense[q]["contexts"] else "None"
        print(f"  DENSE context: {dense_ctx}")
else:
    print("Per-question data not yet present in ablation.json. Re-running python -m eval.ablation...")
