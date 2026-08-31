### API Latency Benchmark Results

**Hardware Specification:** `arm (arm64)` | `10 Cores / 10 Threads` | `16.0 GB RAM` | `OS: Darwin arm64` | `Python 3.14.0`

**Cold Start Latency:** `/score`: `13.9ms` | `/ask`: `2842.9ms`

| Endpoint | p50 | p95 | p99 | Concurrency | Requests |
|---|---|---|---|---|---|
| `/score` | 2.7ms | 7.6ms | 10.8ms | 1 | 100 |
| `/score` | 32.7ms | 58.4ms | 62.6ms | 10 | 100 |
| `/ask` | 10.7ms | 15.4ms | 16.2ms | 1 | 30 |
| `/ask` | 80.8ms | 94.6ms | 96.0ms | 10 | 30 |
