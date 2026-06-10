import csv
import collections
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FALSE_POSITIVES_PATH = Path("reports/evaluation_100/vlm_false_positives.csv")
FALSE_NEGATIVES_PATH = Path("reports/evaluation_100/vlm_false_negatives.csv")
PER_IMAGE_PATH       = Path("reports/evaluation_100/vlm_per_image_evaluation.csv")

OUTPUT_DIR  = Path("reports/error_analysis_100")
FIGURES_DIR = OUTPUT_DIR / "figures"

TOP_N_ITEMS = 20


# Ingredients the model predicts from fridge context, not visual evidence
CONTEXT_GUESS_TERMS = {
    "water", "juice", "orange juice", "lime juice", "lemon juice",
    "apple juice", "soda", "beer", "wine", "broth", "cider",
    "lemonade", "ice", "ice water", "sparkling water",
}

# Common staples the model over-predicts regardless of visibility
COMMON_FRIDGE_DEFAULT_TERMS = {
    "butter", "milk", "cheese", "yogurt", "egg", "mayonnaise",
    "ketchup", "mustard", "salad dressing", "cream cheese",
    "sour cream", "margarine", "almond milk", "whipped cream",
    "half & half", "cottage cheese",
}

# Visually ambiguous items easily confused with others in a cluttered fridge
AMBIGUOUS_VISUAL_TERMS = {
    "meat", "salad", "bread", "lettuce", "carrot", "tomato", "pickle",
    "hot sauce", "jam", "soy sauce", "lemon", "lime", "mushroom",
    "strawberry", "avocado", "blueberry", "spinach", "onion",
    "celery", "zucchini", "pepper", "bell pepper",
}


def categorize_false_positive(ingredient: str) -> str:
    ing = ingredient.strip().lower()
    if ing in CONTEXT_GUESS_TERMS:
        return "context_guess"
    if ing in COMMON_FRIDGE_DEFAULT_TERMS:
        return "common_fridge_default"
    if ing in AMBIGUOUS_VISUAL_TERMS:
        return "ambiguous_visual"
    return "other"


def load_data():
    fp_df  = pd.read_csv(FALSE_POSITIVES_PATH)
    fn_df  = pd.read_csv(FALSE_NEGATIVES_PATH)
    img_df = pd.read_csv(PER_IMAGE_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    return fp_df, fn_df, img_df


def write_fp_categorized(fp_df: pd.DataFrame):
    fp_df = fp_df.copy()
    fp_df["error_category"] = fp_df["false_positive"].apply(categorize_false_positive)

    out_path = OUTPUT_DIR / "vlm_fp_categorized.csv"
    fp_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return fp_df


def write_worst_images(img_df: pd.DataFrame):
    worst = (
        img_df
        .sort_values("f1")
        .head(15)[["image_id", "f1", "precision", "recall", "tp", "fp", "fn",
                   "normalized_ground_truth", "normalized_vlm_predictions"]]
    )
    out_path = OUTPUT_DIR / "vlm_worst_images.csv"
    worst.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return worst


def plot_fp_error_categories(fp_df: pd.DataFrame):
    counts = fp_df["error_category"].value_counts()

    labels = {
        "common_fridge_default": "Common fridge default",
        "ambiguous_visual":      "Ambiguous visual",
        "other":                 "Other",
        "context_guess":         "Context guess",
    }

    plt.figure(figsize=(7, 5))
    plt.pie(
        counts.values,
        labels=[labels.get(c, c) for c in counts.index],
        autopct="%1.0f%%",
        startangle=140,
    )
    plt.title("False positive error categories (n=519)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "vlm_fp_error_categories.png", dpi=300)
    plt.close()
    print("Saved: reports/error_analysis_100/figures/vlm_fp_error_categories.png")


def plot_top_false_positives(fp_df: pd.DataFrame):
    counts = (
        fp_df["false_positive"]
        .value_counts()
        .head(TOP_N_ITEMS)
        .sort_values()
    )

    plt.figure(figsize=(10, 7))
    plt.barh(counts.index, counts.values)
    plt.xlabel("Frequency")
    plt.ylabel("False Positive Ingredient")
    plt.title(f"Top {TOP_N_ITEMS} False Positive Ingredients")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "vlm_top_false_positives.png", dpi=300)
    plt.close()
    print("Saved: reports/error_analysis_100/figures/vlm_top_false_positives.png")


def plot_top_false_negatives(fn_df: pd.DataFrame):
    counts = (
        fn_df["false_negative"]
        .value_counts()
        .head(TOP_N_ITEMS)
        .sort_values()
    )

    plt.figure(figsize=(10, 7))
    plt.barh(counts.index, counts.values)
    plt.xlabel("Frequency")
    plt.ylabel("False Negative Ingredient")
    plt.title(f"Top {TOP_N_ITEMS} False Negative Ingredients")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "vlm_top_false_negatives.png", dpi=300)
    plt.close()
    print("Saved: reports/error_analysis_100/figures/vlm_top_false_negatives.png")


def plot_f1_distribution(img_df: pd.DataFrame):
    f1_scores = img_df["f1"]
    mean_f1   = f1_scores.mean()
    median_f1 = f1_scores.median()

    plt.figure(figsize=(8, 4))
    plt.hist(f1_scores, bins=20, edgecolor="white", linewidth=0.6)
    plt.axvline(mean_f1,   linestyle="--", linewidth=1.5, label=f"Mean F1 = {mean_f1:.3f}")
    plt.axvline(median_f1, linestyle="--", linewidth=1.5, label=f"Median F1 = {median_f1:.3f}")
    plt.xlabel("F1 score per image")
    plt.ylabel("Number of images")
    plt.title("F1 score distribution across 100 images")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "vlm_f1_distribution.png", dpi=300)
    plt.close()
    print("Saved: reports/error_analysis_100/figures/vlm_f1_distribution.png")


def write_error_analysis_summary(fp_df: pd.DataFrame, fn_df: pd.DataFrame, img_df: pd.DataFrame):
    category_counts = fp_df["error_category"].value_counts()
    fn_counts       = fn_df["false_negative"].value_counts()
    f1_scores       = img_df["f1"]

    total_fp  = len(fp_df)
    total_fn  = len(fn_df)
    zero_f1   = int((f1_scores == 0.0).sum())
    below_03  = int(((f1_scores > 0) & (f1_scores < 0.3)).sum())

    def top_items(series, n=8):
        return "\n".join(
            f"- `{item}` ({count}x)"
            for item, count in series.head(n).items()
        )

    summary = f"""# VLM Error Analysis Summary

## Input Files

- False positives: `reports/evaluation_100/vlm_false_positives.csv`
- False negatives: `reports/evaluation_100/vlm_false_negatives.csv`
- Per-image evaluation: `reports/evaluation_100/vlm_per_image_evaluation.csv`

---

## False Positive Analysis

**Total false positives: {total_fp}**

### Error category breakdown

| Category | Count | Share |
|---|---:|---:|
| Common fridge default | {category_counts.get('common_fridge_default', 0)} | {category_counts.get('common_fridge_default', 0) / total_fp * 100:.0f}% |
| Ambiguous visual | {category_counts.get('ambiguous_visual', 0)} | {category_counts.get('ambiguous_visual', 0) / total_fp * 100:.0f}% |
| Other | {category_counts.get('other', 0)} | {category_counts.get('other', 0) / total_fp * 100:.0f}% |
| Context guess | {category_counts.get('context_guess', 0)} | {category_counts.get('context_guess', 0) / total_fp * 100:.0f}% |

### Top false positives by category

**Common fridge defaults** — model predicts staples without visual confirmation:
{top_items(fp_df[fp_df['error_category'] == 'common_fridge_default']['false_positive'].value_counts())}

**Ambiguous visual** — hard to confirm from image alone:
{top_items(fp_df[fp_df['error_category'] == 'ambiguous_visual']['false_positive'].value_counts())}

**Context guesses** — inferred from fridge context, not visual evidence:
{top_items(fp_df[fp_df['error_category'] == 'context_guess']['false_positive'].value_counts())}

### Key finding

The largest false positive driver is **common fridge defaults** ({category_counts.get('common_fridge_default', 0)} FPs, {category_counts.get('common_fridge_default', 0) / total_fp * 100:.0f}%).
The model predicts cheese, butter, and yogurt regardless of visual confirmation.
This is the primary target for confidence filtering in Phase 2.

---

## False Negative Analysis

**Total false negatives: {total_fn}**

### Top 15 most frequently missed ingredients

| Ingredient | Missed in N images |
|---|---:|
{chr(10).join(f"| `{item}` | {count} |" for item, count in fn_counts.head(15).items())}

### Key finding

The most missed ingredients (egg, hot sauce, lemon, ketchup, apple) are
typically small, stored in door compartments, inside opaque packaging, or
visually similar to other items.

---

## Per-Image F1 Analysis

| Metric | Value |
|---|---|
| Mean F1 | {f1_scores.mean():.3f} |
| Median F1 | {f1_scores.median():.3f} |
| Min F1 | {f1_scores.min():.3f} |
| Max F1 | {f1_scores.max():.3f} |
| Images with F1 = 0.0 | {zero_f1} |
| Images with F1 < 0.3 | {below_03} |

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
| `reports/error_analysis_100/vlm_fp_categorized.csv` | All {total_fp} FPs with error category label |
| `reports/error_analysis_100/vlm_worst_images.csv` | 15 lowest F1 images |
| `reports/error_analysis_100/figures/vlm_fp_error_categories.png` | FP category breakdown |
| `reports/error_analysis_100/figures/vlm_top_false_positives.png` | Top {TOP_N_ITEMS} FPs |
| `reports/error_analysis_100/figures/vlm_top_false_negatives.png` | Top {TOP_N_ITEMS} FNs |
| `reports/error_analysis_100/figures/vlm_f1_distribution.png` | F1 distribution histogram |
"""

    out_path = OUTPUT_DIR / "vlm_error_analysis_summary.md"
    out_path.write_text(summary, encoding="utf-8")
    print(f"Saved: {out_path}")


def main():
    fp_df, fn_df, img_df = load_data()

    fp_df = write_fp_categorized(fp_df)
    write_worst_images(img_df)

    plot_fp_error_categories(fp_df)
    plot_top_false_positives(fp_df)
    plot_top_false_negatives(fn_df)
    plot_f1_distribution(img_df)

    write_error_analysis_summary(fp_df, fn_df, img_df)

    print()
    print("Error analysis complete.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()