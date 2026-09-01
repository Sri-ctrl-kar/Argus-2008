### API Latency Benchmark Results

**Hardware Specification:** `arm (arm64)` | `10 Cores / 10 Threads` | `16.0 GB RAM` | `OS: Darwin arm64` | `Python 3.14.0`

**Cold Start Latency:** `/score`: `46.2ms` | `/ask`: `8202.2ms`

| Endpoint | p50 | p95 | p99 | Concurrency | Requests |
|---|---|---|---|---|---|
| `/score` | 3.0ms | 5.2ms | 12.0ms | 1 | 100 |
| `/score` | 34.7ms | 56.4ms | 59.3ms | 10 | 100 |
| `/ask` | 14.5ms | 22.4ms | 23.2ms | 1 | 30 |
| `/ask` | 94.7ms | 109.9ms | 110.5ms | 10 | 30 |
