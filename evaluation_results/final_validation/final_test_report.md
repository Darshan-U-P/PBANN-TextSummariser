# PB-ANN Text Summariser — Final Test Report

- Evaluation time (UTC): 2026-10-11T21:58:06.484324+00:00
- Device: cuda
- Test articles: 500
- Seeds: [42, 123, 2026]
- Sample checks passed: 5/5

## ROUGE results

| Metric | Lead-3 | PB-ANN mean across seeds | Difference |
|---|---:|---:|---:|
| rouge1 | 0.3027 | 0.3040 | +0.0013 |
| rouge2 | 0.1194 | 0.1192 | -0.0001 |
| rougeL | 0.2068 | 0.2094 | +0.0025 |

## Conclusion

Results are mixed across ROUGE metrics. The evaluation does not show a consistent overall winner over Lead-3.

## Sample output checks

| Article ID | Source words | Summary words | Compression | Copied source sentences | Shorter |
|---|---:|---:|---:|---|---|
| f001ec5c4704938247d27a44948eebb37ae98d01 | 567 | 79 | 86.07% | True | True |
| c27cf1b136cc270023de959e7ab24638021bc43f | 364 | 72 | 80.22% | True | True |
| 9f051fe922a8f45b315a366a5af1aab58d93f050 | 992 | 81 | 91.83% | True | True |
| 05cf349b0acd480c600238ba1c40b59a8aafc8b7 | 243 | 60 | 75.31% | True | True |
| d360e4a4a1c26cb01e1955a725688d330a43c971 | 317 | 42 | 86.75% | True | True |

## Limitations

- ROUGE measures lexical overlap with reference summaries, not factual correctness.
- Training sentence labels were heuristic, not manually annotated.
- Lead-3 is a baseline; implementation of extractive summarization does not require beating it.
- Do not use the held-out test results to tune the model.
