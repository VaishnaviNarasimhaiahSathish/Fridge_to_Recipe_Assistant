# VLM Error Analysis Summary

## Input Files

- False positives: `reports/evaluation_100/vlm_false_positives.csv`
- False negatives: `reports/evaluation_100/vlm_false_negatives.csv`
- Per-image evaluation: `reports/evaluation_100/vlm_per_image_evaluation.csv`

---

## False Positive Analysis

**Total false positives: 519**

### Error category breakdown

| Category | Count | Share |
|---|---:|---:|
| Common fridge default | 164 | 32% |
| Ambiguous visual | 173 | 33% |
| Other | 140 | 27% |
| Context guess | 42 | 8% |

### Top false positives by category

**Common fridge defaults** — model predicts staples without visual confirmation:
- `cheese` (28x)
- `yogurt` (27x)
- `butter` (23x)
- `mayonnaise` (14x)
- `milk` (14x)
- `mustard` (12x)
- `salad dressing` (11x)
- `egg` (8x)

**Ambiguous visual** — hard to confirm from image alone:
- `lettuce` (27x)
- `bread` (15x)
- `pickle` (12x)
- `carrot` (11x)
- `lime` (10x)
- `hot sauce` (9x)
- `tomato` (9x)
- `meat` (8x)

**Context guesses** — inferred from fridge context, not visual evidence:
- `water` (10x)
- `juice` (9x)
- `orange juice` (6x)
- `beer` (3x)
- `soda` (3x)
- `broth` (3x)
- `wine` (2x)
- `lime juice` (2x)

### Key finding

The largest false positive driver is **common fridge defaults** (164 FPs, 32%).
The model predicts cheese, butter, and yogurt regardless of visual confirmation.
This is the primary target for confidence filtering in Phase 2.

---

## False Negative Analysis

**Total false negatives: 309**

### Top 15 most frequently missed ingredients

| Ingredient | Missed in N images |
|---|---:|
| `egg` | 14 |
| `hot sauce` | 13 |
| `lemon` | 12 |
| `ketchup` | 11 |
| `apple` | 11 |
| `milk` | 10 |
| `bell pepper` | 9 |
| `blueberry` | 7 |
| `chicken` | 7 |
| `meat` | 7 |
| `spinach` | 7 |
| `butter` | 7 |
| `barbecue sauce` | 7 |
| `carrot` | 6 |
| `whipped cream` | 6 |

### Key finding

The most missed ingredients (egg, hot sauce, lemon, ketchup, apple) are
typically small, stored in door compartments, inside opaque packaging, or
visually similar to other items.

---

## Per-Image F1 Analysis

| Metric | Value |
|---|---|
| Mean F1 | 0.536 |
| Median F1 | 0.539 |
| Min F1 | 0.000 |
| Max F1 | 1.000 |
| Images with F1 = 0.0 | 3 |
| Images with F1 < 0.3 | 8 |

---

## Recommendations for Phase 2

### 1. Confidence threshold filtering
Apply a minimum confidence filter (try 0.5, 0.6, 0.7) to VLM predictions.
Common fridge defaults are often predicted with lower confidence — filtering
these should reduce false positives with minimal false negative increase.

### 2. Expand GENERIC_IGNORE_TERMS
Add the following high-frequency context guesses to the ignore list in
`src/evaluation/evaluate_vlm_predictions.py`:

```python
"water", "juice", "orange juice", "lime juice", "soda",
"broth", "beer", "wine", "cider", "lemonade", "ice",
```

### 3. Prompt refinement
Add the following instruction to `configs/vlm_prompt_with_counts.txt`:

> Only predict ingredients you can visually confirm. Do not guess based on
> what is commonly found in fridges.

---

## Output Files

| File | Description |
|---|---|
| `reports/error_analysis_100/vlm_error_analysis_summary.md` | This report |
| `reports/error_analysis_100/vlm_fp_categorized.csv` | All 519 FPs with error category label |
| `reports/error_analysis_100/vlm_worst_images.csv` | 15 lowest F1 images |
| `reports/error_analysis_100/figures/vlm_fp_error_categories.png` | FP category breakdown |
| `reports/error_analysis_100/figures/vlm_top_false_positives.png` | Top 20 FPs |
| `reports/error_analysis_100/figures/vlm_top_false_negatives.png` | Top 20 FNs |
| `reports/error_analysis_100/figures/vlm_f1_distribution.png` | F1 distribution histogram |
