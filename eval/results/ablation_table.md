| Configuration | R@1 | R@3 | R@5 | MRR | Faithfulness | Citation prec. | Abstention (unans.) | Median latency |
|---|---|---|---|---|---|---|---|---|
| Fixed chunks + dense | 0.108 | 0.189 | 0.297 | 0.167 | n/a | 1.000 | 1.000 | 15ms |
| Section chunks + dense | 0.135 | 0.243 | 0.270 | 0.177 | n/a | 1.000 | 1.000 | 10ms |
| Section chunks + hybrid (BM25 + dense) | 0.027 | 0.135 | 0.189 | 0.080 | n/a | 1.000 | 1.000 | 33ms |
| Section chunks + hybrid + rerank | 0.081 | 0.189 | 0.216 | 0.128 | n/a | 1.000 | 1.000 | 84ms |
