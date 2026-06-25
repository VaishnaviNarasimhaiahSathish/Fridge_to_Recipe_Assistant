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
| high only | 152 | 154 | 130 | 0.4967 | 0.5390 | 0.5170 |
| high + medium (baseline) | 166 | 239 | 116 | 0.4099 | 0.5887 | 0.4833 |

## Input Files

- VLM predictions: `reports/vlm_predictions_100.jsonl`
- Ground truth: `data/annotations/manual_ground_truth_100/manual_ground_truth_100.csv`
- Normalization: `configs/ingredient_normalization.json`
- Skipped (image not found locally, teammate images pending upload): 50

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
