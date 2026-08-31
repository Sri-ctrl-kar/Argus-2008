| Configuration | R@1 | R@3 | R@5 | MRR | Faithfulness | Citation prec. | Abstention (unans.) | Median latency |
|---|---|---|---|---|---|---|---|---|
| Fixed chunks + dense | 0.243 | 0.324 | 0.432 | 0.302 | n/a | 1.000 | 1.000 | 14ms |
| Section chunks + dense | 0.270 | 0.378 | 0.405 | 0.312 | n/a | 1.000 | 1.000 | 11ms |
| Section chunks + hybrid (BM25 + dense) | 0.135 | 0.243 | 0.297 | 0.192 | n/a | 1.000 | 1.000 | 31ms |
| Section chunks + hybrid + rerank | 0.216 | 0.270 | 0.297 | 0.244 | n/a | 1.000 | 1.000 | 79ms |
