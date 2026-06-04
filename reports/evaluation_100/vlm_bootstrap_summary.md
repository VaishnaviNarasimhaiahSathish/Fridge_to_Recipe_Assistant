# Bootstrap Confidence Intervals for VLM Evaluation

Image-level bootstrap resampling was applied to the 100-image evaluation set. In each bootstrap iteration, 100 images were sampled with replacement and evaluation metrics were recalculated. The 2.5th and 97.5th percentiles of the bootstrap distribution are reported as 95% confidence intervals.

| Metric | Original Value | Bootstrap Mean | 95% CI Lower | 95% CI Upper |
|---|---:|---:|---:|---:|
| micro_precision | 0.4789 | 0.4789 | 0.4379 | 0.5204 |
| micro_recall | 0.6069 | 0.6072 | 0.5666 | 0.6475 |
| micro_f1 | 0.5354 | 0.5352 | 0.5000 | 0.5703 |
| macro_precision | 0.5007 | 0.5003 | 0.4566 | 0.5444 |
| macro_recall | 0.6324 | 0.6327 | 0.5878 | 0.6763 |
| macro_f1 | 0.5357 | 0.5356 | 0.4961 | 0.5738 |
| mean_jaccard | 0.3910 | 0.3908 | 0.3537 | 0.4284 |
| exact_match_accuracy | 0.0200 | 0.0200 | 0.0000 | 0.0500 |
