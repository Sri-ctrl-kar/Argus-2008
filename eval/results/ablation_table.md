| Configuration | R@1 | R@3 | R@5 | MRR | Faithfulness | Citation prec. | Abstention (unans.) | Median latency |
|---|---|---|---|---|---|---|---|---|
| BM25 only (diagnostic) | 0.189 | 0.189 | 0.189 | 0.189 | 0.983 | 1.000 | 1.000 | 18ms |
| Fixed chunks + dense | 0.243 | 0.297 | 0.378 | 0.289 | 0.980 | 1.000 | 1.000 | 9ms |
| Section chunks + dense | 0.243 | 0.378 | 0.405 | 0.309 | 0.976 | 1.000 | 1.000 | 11ms |
| Section chunks + hybrid (BM25 + dense) | 0.135 | 0.270 | 0.297 | 0.199 | 0.979 | 1.000 | 1.000 | 31ms |
| Section chunks + hybrid + rerank | 0.216 | 0.324 | 0.351 | 0.259 | 0.982 | 1.000 | 1.000 | 75ms |
