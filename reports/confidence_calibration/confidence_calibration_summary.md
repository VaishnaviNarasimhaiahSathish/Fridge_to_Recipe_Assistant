# Confidence Calibration Analysis

## Inputs

- VLM predictions: `reports/vlm_predictions_100.jsonl`
- Ground truth: `data/annotations/manual_ground_truth_100/manual_ground_truth_100.csv`
- Normalization: `configs/ingredient_normalization.json`

This analysis checks whether the VLM's own confidence labels are meaningful by comparing ingredient-level precision for each confidence group.

## Confidence Group Metrics

| Confidence | Predictions | True Positives | False Positives | Precision | False Positive Rate |
|---|---:|---:|---:|---:|---:|
| high | 759 | 421 | 338 | 0.5547 | 0.4453 |
| medium | 186 | 44 | 142 | 0.2366 | 0.7634 |

## Interpretation

High-confidence predictions are more reliable than medium-confidence predictions, which supports the app's high-confidence filtering choice.

## Top Medium-Confidence False Positives

- `cheese` (12x)
- `lettuce` (10x)
- `yogurt` (10x)
- `butter` (8x)
- `pickle` (5x)
- `meat` (5x)
- `spinach` (4x)
- `bell pepper` (4x)
- `ham` (4x)
- `jam` (4x)

## Generated Files

| File | Description |
|---|---|
| `confidence_group_metrics.csv` | Precision and false-positive rate by confidence group |
| `confidence_per_prediction.csv` | Per-prediction TP/FP correctness |
| `figures/confidence_precision_comparison.png` | Precision by confidence label |
| `figures/confidence_tp_fp_counts.png` | True/false positives by confidence label |
| `figures/confidence_prediction_volume.png` | Number of predictions by confidence label |
