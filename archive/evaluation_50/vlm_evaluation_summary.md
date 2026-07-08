# VLM Ingredient Extraction Evaluation Summary

## Input Files

- Ground truth: `reports/manual_ground_truth_50.csv`
- VLM predictions: `reports/vlm_predictions_v1.jsonl`
- Normalization file: `configs/ingredient_normalization.json`

## Dataset

- Number of evaluated images: 50

## Micro-Averaged Metrics

Micro metrics aggregate true positives, false positives, and false negatives across all images.

| Metric | Value |
|---|---:|
| True Positives | 307 |
| False Positives | 271 |
| False Negatives | 185 |
| Precision | 0.5311 |
| Recall | 0.6240 |
| F1-score | 0.5738 |

## Macro-Averaged Metrics

Macro metrics calculate the metric per image and average across images.

| Metric | Value |
|---|---:|
| Precision | 0.5807 |
| Recall | 0.6558 |
| F1-score | 0.5960 |

## Accuracy-Like Metrics

These are reported because standard classification accuracy is not suitable for open-vocabulary multi-label ingredient extraction.

| Metric | Value |
|---|---:|
| Exact Match Accuracy | 0.0200 |
| Mean Jaccard Similarity | 0.4398 |

## Output Files

- Per-image results: `reports/evaluation/vlm_per_image_evaluation.csv`
- False positives: `reports/evaluation/vlm_false_positives.csv`
- False negatives: `reports/evaluation/vlm_false_negatives.csv`

## Notes

- Evaluation is based on normalized ingredient names.
- Generic uncertain items such as `unknown bottle`, `unknown jar`, `beverage`, `condiment`, and `leftover food` are excluded from the main ingredient metrics.
- Exact match accuracy checks whether the complete predicted ingredient set exactly matches the ground truth set for an image.
- Mean Jaccard similarity measures average set overlap between predicted and ground-truth ingredients.
