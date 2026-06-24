# VLM Ingredient Extraction Evaluation Summary

## Input Files

- Ground truth: `data/annotations/manual_ground_truth_100/manual_ground_truth_100.csv`
- VLM predictions: `reports/vlm_predictions_100.jsonl`
- Normalization file: `configs/ingredient_normalization.json`

## Dataset

- Number of evaluated images: 100
- Missing VLM prediction rows: 0

## Micro-Averaged Metrics

Micro metrics aggregate true positives, false positives, and false negatives across all images.

| Metric | Value |
|---|---:|
| True Positives | 477 |
| False Positives | 519 |
| False Negatives | 309 |
| Precision | 0.4789 |
| Recall | 0.6069 |
| F1-score | 0.5354 |

## Macro-Averaged Metrics

Macro metrics calculate the metric per image and average across images.

| Metric | Value |
|---|---:|
| Precision | 0.5007 |
| Recall | 0.6324 |
| F1-score | 0.5357 |

## Accuracy-Like Metrics

These are reported because standard classification accuracy is not suitable for open-vocabulary multi-label ingredient extraction.

| Metric | Value |
|---|---:|
| Exact Match Accuracy | 0.0200 |
| Mean Jaccard Similarity | 0.3910 |

## Output Files

- Per-image results: `reports/evaluation_100/vlm_per_image_evaluation.csv`
- False positives: `reports/evaluation_100/vlm_false_positives.csv`
- False negatives: `reports/evaluation_100/vlm_false_negatives.csv`

## Notes

- Evaluation is based on normalized ingredient names.
- Generic uncertain items such as `unknown bottle`, `unknown jar`, `beverage`, `condiment`, `prepared food`, and `leftover food` are excluded from the main ingredient metrics.
- Exact match accuracy checks whether the complete predicted ingredient set exactly matches the ground truth set for an image.
- Mean Jaccard similarity measures average set overlap between predicted and ground-truth ingredients.
