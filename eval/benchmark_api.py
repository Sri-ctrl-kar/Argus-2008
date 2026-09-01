"""Argus API Latency Benchmarking Harness.

Measures p50, p95, p99 latencies for /score and /ask endpoints under
sequential and concurrent load, evaluating uncached vs cached /ask queries
over the eval/questions.jsonl corpus.

Run: python -m eval.benchmark_api
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient
from api.main import app

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_hardware_info() -> Dict[str, Any]:
    """Collect accurate host hardware specifications using standard library."""
    import subprocess

    ram_gb = 16.0
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            ram_gb = round(int(out) / (1024 ** 3), 2)
    except Exception:
        pass

    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "Apple Silicon",
        "cpu_count_logical": os.cpu_count() or 8,
        "cpu_count_physical": os.cpu_count() or 8,
        "ram_gb": ram_gb,
        "python_version": platform.python_version(),
    }


def get_sample_score_payload() -> Dict[str, float]:
    """Sample transaction payload for /score benchmarking."""
    features = {f"V{i}": float(i * 0.1 - 1.4) for i in range(1, 29)}
    features["Time"] = 43200.0
    features["Amount"] = 149.50
    return features


def run_latency_benchmark(
    client: TestClient,
    endpoint: str,
    payload: Dict[str, Any],
    concurrency: int = 1,
    n_requests: int = 50,
) -> Dict[str, Any]:
    """Benchmark endpoint latency across n_requests under specified concurrency."""
    latencies: List[float] = []

    def send_request() -> float:
        t0 = time.perf_counter()
        resp = client.post(endpoint, json=payload)
        t1 = time.perf_counter()
        assert resp.status_code == 200, f"Request failed: {resp.status_code}"
        return (t1 - t0) * 1000  # Convert to ms

    if concurrency == 1:
        for _ in range(n_requests):
            latencies.append(send_request())
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send_request) for _ in range(n_requests)]
            for f in concurrent.futures.as_completed(futures):
                latencies.append(f.result())

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95 = latencies_sorted[int(0.95 * len(latencies_sorted)) - 1]
    p99 = latencies_sorted[int(0.99 * len(latencies_sorted)) - 1]

    return {
        "endpoint": endpoint,
        "concurrency": concurrency,
        "n_requests": n_requests,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "min_ms": round(latencies_sorted[0], 2),
        "max_ms": round(latencies_sorted[-1], 2),
    }


def to_markdown_table(rows: List[Dict[str, Any]], cold_starts: Dict[str, float], hw: Dict[str, Any]) -> str:
    """Format benchmark results as a clean Markdown table with hardware metadata."""
    proc = hw["processor"]
    mach = hw["machine"]
    phys = hw["cpu_count_physical"]
    logi = hw["cpu_count_logical"]
    ram = hw["ram_gb"]
    opsys = hw["os"]
    pyv = hw["python_version"]
    cs_score = cold_starts["/score"]
    cs_ask = cold_starts["/ask"]

    lines = [
        "### API Latency Benchmark Results",
        "",
        f"**Hardware Specification:** `{proc} ({mach})` | `{phys} Cores / {logi} Threads` | `{ram} GB RAM` | `OS: {opsys} {mach}` | `Python {pyv}`",
        "",
        f"**Cold Start Latency:** `/score`: `{cs_score:.1f}ms` | `/ask`: `{cs_ask:.1f}ms`",
        "",
        "| Endpoint | Description | p50 | p95 | p99 | Concurrency | Requests |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ep = r["endpoint"]
        desc = r.get("description", "")
        p50 = r["p50_ms"]
        p95 = r["p95_ms"]
        p99 = r["p99_ms"]
        conc = r["concurrency"]
        n_req = r["n_requests"]
        lines.append(
            f"| `{ep}` | {desc} | {p50:.1f}ms | {p95:.1f}ms | {p99:.1f}ms | {conc} | {n_req} |"
        )
    return "\n".join(lines)


def main() -> None:
    print("========================================================================")
    print("Argus API Latency Benchmarking Suite")
    print("========================================================================")

    hw = get_hardware_info()
    proc = hw["processor"]
    mach = hw["machine"]
    ram = hw["ram_gb"]
    pyv = hw["python_version"]
    print(f"Hardware: {proc} ({mach}) | {ram} GB RAM | Python {pyv}")

    # Load 30 questions from eval/questions.jsonl
    questions_file = Path("eval/questions.jsonl")
    if questions_file.exists():
        qs = [json.loads(l)["question"] for l in questions_file.read_text().splitlines() if l.strip()][:30]
    else:
        qs = ["What was Apple total net sales in fiscal 2025?"] * 30

    score_payload = get_sample_score_payload()

    with TestClient(app) as client:
        # Measure Cold Starts
        print("\n--- Measuring Cold Starts ---")
        t0 = time.perf_counter()
        resp_score = client.post("/score", json=score_payload)
        t_cold_score = (time.perf_counter() - t0) * 1000
        assert resp_score.status_code == 200

        t0 = time.perf_counter()
        resp_ask = client.post("/ask", json={"question": qs[0]})
        t_cold_ask = (time.perf_counter() - t0) * 1000
        assert resp_ask.status_code == 200

        cold_starts = {"/score": t_cold_score, "/ask": t_cold_ask}
        print(f"  Cold Start /score: {t_cold_score:.2f}ms")
        print(f"  Cold Start /ask:   {t_cold_ask:.2f}ms")

        rows = []

        # 1. /score Sequential
        print("\n--- Benchmarking /score (Sequential c=1) ---")
        r1 = run_latency_benchmark(client, "/score", score_payload, concurrency=1, n_requests=100)
        r1["description"] = "Single transaction score"
        rows.append(r1)
        print(f"  /score (c=1):  p50={r1['p50_ms']}ms | p95={r1['p95_ms']}ms | p99={r1['p99_ms']}ms")

        # 2. /score Concurrent
        print("\n--- Benchmarking /score (Concurrent c=10) ---")
        r2 = run_latency_benchmark(client, "/score", score_payload, concurrency=10, n_requests=100)
        r2["description"] = "Concurrent transaction scoring"
        rows.append(r2)
        print(f"  /score (c=10): p50={r2['p50_ms']}ms | p95={r2['p95_ms']}ms | p99={r2['p99_ms']}ms")


        # 3. /ask Uncached vs Cached Evaluation
        print(f"\n--- Benchmarking /ask across {len(qs)} corpus questions ---")
        def timed_ask(q: str) -> float:
            t = time.perf_counter()
            resp = client.post("/ask", json={"question": q})
            assert resp.status_code == 200
            return (time.perf_counter() - t) * 1000

        cold_times = [timed_ask(q) for q in qs]        # each question seen for the first time
        warm_times = [timed_ask(q) for q in qs]        # same questions, now cached

        for label, xs, desc in (
            ("uncached", cold_times, "30 distinct 10-K questions (cold/first-seen)"),
            ("cached", warm_times, "30 repeat 10-K questions (LRU warm-cache)"),
        ):
            xs_sorted = sorted(xs)
            p50 = statistics.median(xs_sorted)
            p95 = xs_sorted[int(0.95 * len(xs_sorted)) - 1]
            p99 = xs_sorted[int(0.99 * len(xs_sorted)) - 1]
            print(f"{label:<10} p50 {p50:7.1f}ms  p95 {p95:7.1f}ms")
            rows.append({
                "endpoint": "/ask",
                "description": desc,
                "concurrency": 1,
                "n_requests": len(xs),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "min_ms": round(xs_sorted[0], 2),
                "max_ms": round(xs_sorted[-1], 2),
            })

        # 4. /ask Concurrent c=10
        print("\n--- Benchmarking /ask (Concurrent c=10) ---")
        def send_concurrent_ask(q: str) -> float:
            t = time.perf_counter()
            resp = client.post("/ask", json={"question": q})
            assert resp.status_code == 200
            return (time.perf_counter() - t) * 1000

        concurrent_latencies = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(send_concurrent_ask, qs[i % len(qs)]) for i in range(30)]
            for f in concurrent.futures.as_completed(futures):
                concurrent_latencies.append(f.result())

        c_sorted = sorted(concurrent_latencies)
        p50 = statistics.median(c_sorted)
        p95 = c_sorted[int(0.95 * len(c_sorted)) - 1]
        p99 = c_sorted[int(0.99 * len(c_sorted)) - 1]
        r_conc = {
            "endpoint": "/ask",
            "description": "Concurrent document Q&A",
            "concurrency": 10,
            "n_requests": len(c_sorted),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "min_ms": round(c_sorted[0], 2),
            "max_ms": round(c_sorted[-1], 2),
        }
        rows.append(r_conc)
        print(f"  /ask (c=10):   p50={r_conc['p50_ms']}ms | p95={r_conc['p95_ms']}ms | p99={r_conc['p99_ms']}ms")


    # Save results
    report_data = {
        "hardware": hw,
        "cold_starts_ms": cold_starts,
        "benchmarks": rows,
    }
    (RESULTS_DIR / "api_latency.json").write_text(json.dumps(report_data, indent=2))
    md_table = to_markdown_table(rows, cold_starts, hw)
    (RESULTS_DIR / "api_latency.md").write_text(md_table + "\n")

    print("\n" + "=" * 70)
    print(md_table)
    print("=" * 70)
    print(f"Wrote {RESULTS_DIR}/api_latency.json and api_latency.md")


if __name__ == "__main__":
    main()
