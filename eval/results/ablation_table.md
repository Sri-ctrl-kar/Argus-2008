| Configuration | R@1 | R@3 | R@5 | MRR | Faithfulness | Citation prec. | Abstention (unans.) | Median latency |
|---|---|---|---|---|---|---|---|---|
| Fixed chunks + dense | 0.243 | 0.297 | 0.378 | 0.289 | n/a | 1.000 | 1.000 | 12ms |
| Section chunks + dense | 0.243 | 0.378 | 0.405 | 0.309 | n/a | 1.000 | 1.000 | 12ms |
| Section chunks + hybrid (BM25 + dense) | 0.135 | 0.216 | 0.324 | 0.197 | n/a | 1.000 | 1.000 | 30ms |
| Section chunks + hybrid + rerank | 0.216 | 0.297 | 0.324 | 0.253 | n/a | 1.000 | 1.000 | 76ms |
