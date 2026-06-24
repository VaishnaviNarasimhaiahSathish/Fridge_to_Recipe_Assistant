"""Compare Gemma 4's raw ingredient predictions against Qwen-reviewed corrections.

The 100-image Gemma batch has no independent manual ground truth, so the
Qwen-reviewed corrected_visible_ingredients column is used as a ground-truth
proxy: Gemma's raw visible_ingredients are evaluated against it the same way
the main pipeline evaluates VLM predictions against manual ground truth.
"""

import csv
import re
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

INPUT_CSV = Path("data/annotations/gemma4_batch_100/gemma4_annotations_reviewed.csv")
NORMALIZATION_PATH = Path("configs/ingredient_normalization.json")

OUTPUT_DIR = Path("reports/gemma4_batch_100")
FIGURES_DIR = OUTPUT_DIR / "figures"
PER_IMAGE_OUTPUT = OUTPUT_DIR / "gemma_vs_qwen_per_image.csv"
MISSED_INGREDIENTS_OUTPUT = OUTPUT_DIR / "gemma_missed_ingredients.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "gemma_vs_qwen_summary.md"

GENERIC_IGNORE_TERMS = {
    "unknown jar", "unknown bottle", "unknown packaged item",
    "unknown container", "unknown item", "unknown food item",
    "food", "drink", "beverage", "condiment", "condiments", "container",
    "package", "packaged item", "prepared food", "prepared meal",
    "prepared salad", "leftover food", "frozen food", "canned food",
    "canned fruit", "sauce", "bottle", "jar", "grocery", "item",
    "green", "greens", "liquid", "leftover", "fruit", "vegetable", "vegetables",
    "chopped vegetables", "frozen vegetable", "leafy green vegetable",
    "dressing", "dips", "snack", "dessert", "spread", "preserve",
}


def load_normalization_map(path: Path) -> dict:
    import json
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def basic_clean_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    return name.strip(" .,-")


def split_ingredient_list(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in re.split(r"[;,]", str(value)) if part.strip()]


def normalize_items(items: list[str], normalization_map: dict) -> list[str]:
    normalized = []
    for raw in items:
        name = basic_clean_name(raw)
        if not name:
            continue
        mapped = normalization_map.get(name, name)
        mapped_list = mapped if isinstance(mapped, list) else [mapped]
        for mapped_name in mapped_list:
            cleaned = basic_clean_name(mapped_name)
            if cleaned and cleaned not in GENERIC_IGNORE_TERMS:
                normalized.append(cleaned)
    seen = set()
    unique = []
    for item in normalized:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_image(gemma_items: list[str], qwen_items: list[str]) -> dict:
    gemma_set = set(gemma_items)
    qwen_set = set(qwen_items)

    true_positives = sorted(gemma_set & qwen_set)
    false_positives = sorted(gemma_set - qwen_set)
    false_negatives = sorted(qwen_set - gemma_set)

    tp, fp, fn = len(true_positives), len(false_positives), len(false_negatives)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    exact_match = int(gemma_set == qwen_set)
    jaccard = safe_divide(len(gemma_set & qwen_set), len(gemma_set | qwen_set))

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "exact_match": exact_match, "jaccard": jaccard,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def list_to_string(items: list[str]) -> str:
    return "; ".join(items)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    normalization_map = load_normalization_map(NORMALIZATION_PATH)

    with INPUT_CSV.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    per_image_rows = []
    missed_counter = Counter()
    total_tp = total_fp = total_fn = 0
    gemma_ingredient_counts = []
    qwen_ingredient_counts = []

    for row in rows:
        gemma_raw = split_ingredient_list(row.get("visible_ingredients", ""))
        qwen_raw = split_ingredient_list(row.get("corrected_visible_ingredients", ""))

        gemma_items = normalize_items(gemma_raw, normalization_map)
        qwen_items = normalize_items(qwen_raw, normalization_map)

        gemma_ingredient_counts.append(len(gemma_items))
        qwen_ingredient_counts.append(len(qwen_items))

        result = evaluate_image(gemma_items, qwen_items)
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

        for item in result["false_negatives"]:
            missed_counter[item] += 1

        per_image_rows.append({
            "image_id": row["image_id"],
            "image_path": row.get("image_path", ""),
            "gemma_ingredients": list_to_string(gemma_items),
            "qwen_corrected_ingredients": list_to_string(qwen_items),
            "gemma_ingredient_count": len(gemma_items),
            "qwen_ingredient_count": len(qwen_items),
            "tp": result["tp"],
            "fp": result["fp"],
            "fn": result["fn"],
            "precision": round(result["precision"], 4),
            "recall": round(result["recall"], 4),
            "f1": round(result["f1"], 4),
            "exact_match": result["exact_match"],
            "jaccard": round(result["jaccard"], 4),
            "false_negatives": list_to_string(result["false_negatives"]),
            "false_positives": list_to_string(result["false_positives"]),
        })

    with PER_IMAGE_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    top_missed = missed_counter.most_common(20)
    with MISSED_INGREDIENTS_OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ingredient", "missed_count"])
        writer.writerows(top_missed)

    micro_precision = safe_divide(total_tp, total_tp + total_fp)
    micro_recall = safe_divide(total_tp, total_tp + total_fn)
    micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    macro_precision = statistics.mean(r["precision"] for r in per_image_rows)
    macro_recall = statistics.mean(r["recall"] for r in per_image_rows)
    macro_f1 = statistics.mean(r["f1"] for r in per_image_rows)
    exact_match_accuracy = statistics.mean(r["exact_match"] for r in per_image_rows)
    mean_jaccard = statistics.mean(r["jaccard"] for r in per_image_rows)

    avg_gemma_count = statistics.mean(gemma_ingredient_counts)
    avg_qwen_count = statistics.mean(qwen_ingredient_counts)

    # Top missed ingredients chart
    if top_missed:
        labels = [name for name, _ in top_missed[:15]][::-1]
        values = [count for _, count in top_missed[:15]][::-1]
        plt.figure(figsize=(10, 7))
        plt.barh(labels, values)
        plt.title("Top ingredients Gemma missed (vs Qwen-corrected reference)")
        plt.xlabel("Times missed")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "gemma_vs_qwen_top_missed.png", dpi=300)
        plt.close()

    accuracy_rows = sorted(per_image_rows, key=lambda r: r["f1"])
    worst5 = accuracy_rows[:5]
    best5 = accuracy_rows[-5:][::-1]

    def fmt_image_rows(rows):
        return "\n".join(
            f"| {r['image_id']} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |"
            for r in rows
        )

    summary = f"""# Gemma 4 vs Qwen Ingredient Comparison

## Input File

- `{INPUT_CSV}`

Gemma's raw `visible_ingredients` predictions are evaluated against Qwen's
`corrected_visible_ingredients` review, used here as a ground-truth proxy
since this 100-image batch has no independent manual ground truth.

## Dataset

- Images compared: {len(per_image_rows)}

## Micro-Averaged Metrics

| Metric | Value |
|---|---:|
| True Positives | {total_tp} |
| False Positives | {total_fp} |
| False Negatives | {total_fn} |
| Precision | {micro_precision:.4f} |
| Recall | {micro_recall:.4f} |
| F1-score | {micro_f1:.4f} |

## Macro-Averaged Metrics

| Metric | Value |
|---|---:|
| Precision | {macro_precision:.4f} |
| Recall | {macro_recall:.4f} |
| F1-score | {macro_f1:.4f} |

## Accuracy-Like Metrics

| Metric | Value |
|---|---:|
| Exact Match Accuracy | {exact_match_accuracy:.4f} |
| Mean Jaccard Similarity | {mean_jaccard:.4f} |

## Average Ingredients Detected

| Source | Average per image |
|---|---:|
| Gemma (raw) | {avg_gemma_count:.2f} |
| Qwen (corrected) | {avg_qwen_count:.2f} |

Gemma detects {avg_gemma_count / avg_qwen_count * 100:.0f}% of the ingredient count Qwen confirms per image on average.

## Most Missed Ingredients

Top ingredients present in the Qwen-corrected reference but missed by Gemma:

| Ingredient | Times missed |
|---|---:|
{chr(10).join(f"| {name} | {count} |" for name, count in top_missed[:10])}

![Top missed ingredients](figures/gemma_vs_qwen_top_missed.png)

## Best / Worst Per-Image Agreement

Worst 5 (lowest F1):

| Image | F1 | Precision | Recall |
|---|---:|---:|---:|
{fmt_image_rows(worst5)}

Best 5 (highest F1):

| Image | F1 | Precision | Recall |
|---|---:|---:|---:|
{fmt_image_rows(best5)}

## Output Files

- Per-image results: `{PER_IMAGE_OUTPUT}`
- Most missed ingredients: `{MISSED_INGREDIENTS_OUTPUT}`
- Figure: `{FIGURES_DIR / "gemma_vs_qwen_top_missed.png"}`

## Notes

- Ingredient names are normalized using `{NORMALIZATION_PATH}`, the same map used for the main Qwen-vs-manual-ground-truth evaluation.
- Generic/uncertain terms (e.g. `unknown bottle`, `prepared food`) are excluded from matching, consistent with `src/evaluation/evaluate_vlm_predictions.py`.
- This comparison measures agreement between Gemma and Qwen, not absolute correctness against a human-verified label set.
"""

    SUMMARY_OUTPUT.write_text(summary, encoding="utf-8")

    print("Gemma vs Qwen comparison complete.")
    print(f"Per-image results: {PER_IMAGE_OUTPUT}")
    print(f"Missed ingredients: {MISSED_INGREDIENTS_OUTPUT}")
    print(f"Summary report: {SUMMARY_OUTPUT}")
    print()
    print(f"Micro F1: {micro_f1:.4f}  Macro F1: {macro_f1:.4f}")
    print(f"Avg Gemma ingredients/image: {avg_gemma_count:.2f}  Avg Qwen ingredients/image: {avg_qwen_count:.2f}")


if __name__ == "__main__":
    main()
