# Bootstrap Confidence Intervals for VLM Evaluation

Image-level bootstrap resampling was applied to the 100-image evaluation set. In each bootstrap iteration, 100 images were sampled with replacement and evaluation metrics were recalculated. The 2.5th and 97.5th percentiles of the bootstrap distribution are reported as 95% confidence intervals.

| Metric | Original Value | Bootstrap Mean | 95% CI Lower | 95% CI Upper |
|---|---:|---:|---:|---:|
| micro_precision | 0.4779 | 0.4778 | 0.4368 | 0.5195 |
| micro_recall | 0.6056 | 0.6059 | 0.5651 | 0.6467 |
| micro_f1 | 0.5342 | 0.5341 | 0.4984 | 0.5694 |
| macro_precision | 0.4992 | 0.4988 | 0.4549 | 0.5431 |
| macro_recall | 0.6308 | 0.6310 | 0.5854 | 0.6748 |
| macro_f1 | 0.5342 | 0.5340 | 0.4941 | 0.5728 |
| mean_jaccard | 0.3900 | 0.3898 | 0.3522 | 0.4276 |
| exact_match_accuracy | 0.0200 | 0.0200 | 0.0000 | 0.0500 |
