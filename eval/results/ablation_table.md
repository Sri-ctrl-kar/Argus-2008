| Configuration | R@1 | R@3 | R@5 | MRR | Faithfulness | Citation prec. | Abstention (unans.) | Median latency |
|---|---|---|---|---|---|---|---|---|
| Fixed chunks + dense | 0.243 | 0.324 | 0.432 | 0.302 | n/a | 1.000 | 1.000 | 15ms |
| Section chunks + dense | 0.270 | 0.378 | 0.405 | 0.312 | n/a | 1.000 | 1.000 | 14ms |
| Section chunks + hybrid (BM25 + dense) | 0.135 | 0.270 | 0.324 | 0.201 | n/a | 1.000 | 1.000 | 30ms |
| Section chunks + hybrid + rerank | 0.216 | 0.324 | 0.351 | 0.264 | n/a | 1.000 | 1.000 | 92ms |
