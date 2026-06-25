# Bootstrap Confidence Intervals for VLM Evaluation

Image-level bootstrap resampling was applied to the 50-image evaluation set. In each bootstrap iteration, 50 images were sampled with replacement and evaluation metrics were recalculated. The 2.5th and 97.5th percentiles of the bootstrap distribution are reported as 95% confidence intervals.

| Metric | Original Value | Bootstrap Mean | 95% CI Lower | 95% CI Upper |
|---|---:|---:|---:|---:|
| micro_precision | 0.5311 | 0.5320 | 0.4796 | 0.5877 |
| micro_recall | 0.6240 | 0.6241 | 0.5784 | 0.6700 |
| micro_f1 | 0.5738 | 0.5740 | 0.5333 | 0.6154 |
| macro_precision | 0.5807 | 0.5804 | 0.5308 | 0.6309 |
| macro_recall | 0.6558 | 0.6554 | 0.6101 | 0.7002 |
| macro_f1 | 0.5960 | 0.5957 | 0.5564 | 0.6347 |
| mean_jaccard | 0.4398 | 0.4395 | 0.3984 | 0.4826 |
| exact_match_accuracy | 0.0200 | 0.0198 | 0.0000 | 0.0600 |
