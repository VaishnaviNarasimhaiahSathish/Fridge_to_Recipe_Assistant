# VLM Confidence Filtering Summary

## Method

The VLM returns a confidence field per predicted ingredient: `high` or `medium`.

Two strategies are compared:

- **Baseline** — keep all predictions (high + medium confidence)
- **High only** — discard medium confidence predictions before evaluation

The expanded `GENERIC_IGNORE_TERMS` from Phase 1 error analysis is applied
in both strategies (context guesses: water, juice, soda, broth, beer, wine,
cider, lemonade, ice, orange juice, lime juice).

## Results

| Strategy | TP | FP | FN | Precision | Recall | Micro F1 |
|---|---:|---:|---:|---:|---:|---:|
| high only | 419 | 340 | 337 | 0.5520 | 0.5542 | 0.5531 |
| high + medium (baseline) | 458 | 482 | 298 | 0.4872 | 0.6058 | 0.5401 |

## Input Files

- VLM predictions: `reports/vlm_predictions_100.jsonl`
- Ground truth: `data/annotations/manual_ground_truth_100/manual_ground_truth_100.csv`
- Normalization: `configs/ingredient_normalization.json`

## Output Files

- This summary: `reports/confidence_filtering_100/vlm_confidence_filtering_summary.md`
- Per-image results: `reports/confidence_filtering_100/vlm_confidence_per_image.csv`

## Notes

- Filtering medium confidence predictions reduces false positives but also
  increases false negatives since some correct predictions are medium confidence.
- The best operating point depends on whether the downstream recipe module
  prefers precision (fewer wrong ingredients) or recall (fewer missed ingredients).
- For recipe recommendation, higher precision is preferable — a recipe suggested
  from confirmed ingredients is more trustworthy than one based on guesses.
