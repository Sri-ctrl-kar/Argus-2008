| Configuration | R@1 | R@3 | R@5 | MRR | Faithfulness | Citation prec. | Abstention (unans.) | Median latency |
|---|---|---|---|---|---|---|---|---|
| BM25 only (diagnostic) | 0.189 | 0.189 | 0.189 | 0.189 | n/a | 1.000 | 0.875 | 19ms |
| Fixed chunks + dense | 0.243 | 0.297 | 0.378 | 0.289 | n/a | 1.000 | 1.000 | 14ms |
| Section chunks + dense | 0.243 | 0.378 | 0.405 | 0.309 | n/a | 1.000 | 1.000 | 11ms |
| Section chunks + hybrid (BM25 + dense) | 0.135 | 0.270 | 0.297 | 0.199 | n/a | 1.000 | 1.000 | 32ms |
| Section chunks + hybrid + rerank | 0.216 | 0.324 | 0.351 | 0.259 | n/a | 1.000 | 1.000 | 81ms |
