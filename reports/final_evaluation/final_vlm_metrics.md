# Final VLM Evaluation Metrics

## Inputs

- Ground truth: `data/annotations/manual_ground_truth_100/manual_ground_truth_100.csv`
- VLM predictions: `reports/vlm_predictions_100.jsonl`
- Normalization: `configs/ingredient_normalization.json`

Skipped images because the image file was not found locally: **0**

## Final Metrics

| Strategy | Images | TP | FP | FN | Precision | Recall | Micro F1 | Mean Jaccard | Exact Match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| High + medium predictions | 100 | 460 | 479 | 299 | 0.4899 | 0.6061 | 0.5418 | 0.3960 | 0.0200 |
| High confidence only | 100 | 421 | 338 | 338 | 0.5547 | 0.5547 | 0.5547 | 0.4072 | 0.0300 |

## Interpretation

- **High + medium predictions** measures the full saved VLM output.
- **High confidence only** matches the app operating point used in the React + FastAPI demo.
- For recipe recommendation, high precision is preferred because wrong ingredients can lead to misleading recipe suggestions.

## Figures

- `figures/final_main_metrics.png`
- `figures/final_confidence_comparison.png`
- `figures/final_precision_vs_recall.png`
- `figures/final_top_false_positives.png`
- `figures/final_top_false_negatives.png`
- `figures/final_error_categories.png`
