| Configuration | R@1 | R@3 | R@5 | MRR | Faithfulness | Citation prec. | Abstention (unans.) | Median latency |
|---|---|---|---|---|---|---|---|---|
| BM25 only (diagnostic) | 0.189 | 0.189 | 0.189 | 0.189 | 0.988 | 1.000 | 0.875 | 15ms |
| Fixed chunks + dense | 0.243 | 0.297 | 0.378 | 0.289 | 0.997 | 1.000 | 1.000 | 8ms |
| Section chunks + dense | 0.243 | 0.378 | 0.405 | 0.309 | 0.995 | 1.000 | 1.000 | 8ms |
| Section chunks + hybrid (BM25 + dense) | 0.135 | 0.270 | 0.297 | 0.199 | 0.960 | 1.000 | 1.000 | 28ms |
| Section chunks + hybrid + rerank | 0.216 | 0.324 | 0.351 | 0.259 | 0.997 | 1.000 | 1.000 | 69ms |
