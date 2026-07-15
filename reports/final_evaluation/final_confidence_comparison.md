# Final Confidence Filtering Comparison

| Strategy | TP | FP | FN | Precision | Recall | Micro F1 |
|---|---:|---:|---:|---:|---:|---:|
| High + medium | 460 | 479 | 299 | 0.4899 | 0.6061 | 0.5418 |
| High only | 421 | 338 | 338 | 0.5547 | 0.5547 | 0.5547 |

## Key message

High-only filtering usually reduces false positives and improves trustworthiness for recipe recommendation, but it can also reduce recall by dropping correct medium-confidence predictions.
