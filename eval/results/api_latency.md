### API Latency Benchmark Results

**Hardware Specification:** `arm (arm64)` | `10 Cores / 10 Threads` | `16.0 GB RAM` | `OS: Darwin arm64` | `Python 3.14.0`

**Cold Start Latency:** `/score`: `46.4ms` | `/ask`: `6293.0ms`

| Endpoint | Description | p50 | p95 | p99 | Concurrency | Requests |
|---|---|---|---|---|---|---|
| `/score` | Single transaction score | 3.6ms | 6.6ms | 9.6ms | 1 | 100 |
| `/score` | Concurrent transaction scoring | 39.0ms | 62.4ms | 70.5ms | 10 | 100 |
| `/ask` | 30 distinct 10-K questions (cold/first-seen) | 1639.0ms | 5132.3ms | 5378.4ms | 1 | 30 |
| `/ask` | 30 repeat 10-K questions (LRU warm-cache) | 12.1ms | 18.0ms | 26.0ms | 1 | 30 |
| `/ask` | Concurrent document Q&A | 148.0ms | 187.3ms | 196.6ms | 10 | 30 |
