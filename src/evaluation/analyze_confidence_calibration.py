import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.image_resolver import resolve_image_path


GROUND_TRUTH_PATH = Path("data/annotations/manual_ground_truth_100/manual_ground_truth_100.csv")
PREDICTIONS_PATH = Path("reports/vlm_predictions_100.jsonl")
NORMALIZATION_PATH = Path("configs/ingredient_normalization.json")

OUTPUT_DIR = Path("reports/confidence_calibration")
FIGURES_DIR = OUTPUT_DIR / "figures"

GROUP_METRICS_OUTPUT = OUTPUT_DIR / "confidence_group_metrics.csv"
PER_PREDICTION_OUTPUT = OUTPUT_DIR / "confidence_per_prediction.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "confidence_calibration_summary.md"


GENERIC_IGNORE_TERMS = {
    "unknown jar",
    "unknown bottle",
    "unknown packaged item",
    "unknown container",
    "unknown item",
    "unknown food item",
    "food",
    "drink",
    "beverage",
    "condiment",
    "condiments",
    "container",
    "package",
    "packaged item",
    "prepared food",
    "prepared meal",
    "prepared salad",
    "leftover food",
    "frozen food",
    "canned food",
    "canned fruit",
    "sauce",
    "bottle",
    "jar",
    "grocery",
    "item",
    "green",
    "greens",
    "liquid",
    "leftover",
    "fruit",
    "vegetable",
    "vegetables",
    "chopped vegetables",
    "frozen vegetable",
    "leafy green vegetable",
    "dressing",
    "dips",
    "snack",
    "dessert",
    "spread",
    "preserve",
    "water",
    "juice",
    "orange juice",
    "lime juice",
    "lemon juice",
    "apple juice",
    "soda",
    "broth",
    "beer",
    "wine",
    "cider",
    "lemonade",
    "ice",
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def basic_clean_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    return name.strip(" .,-")


def split_ingredient_list(value: str) -> list[str]:
    if not value:
        return []

    return [
        part.strip()
        for part in re.split(r"[;,]", str(value))
        if part.strip()
    ]


def load_normalization_map(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Normalization file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_item(name: str, normalization_map: dict) -> list[str]:
    cleaned = basic_clean_name(name)

    if not cleaned:
        return []

    mapped = normalization_map.get(cleaned, cleaned)

    if isinstance(mapped, list):
        return [
            basic_clean_name(item)
            for item in mapped
            if basic_clean_name(item)
        ]

    return [basic_clean_name(mapped)]


def normalize_items(items: list[str], normalization_map: dict) -> list[str]:
    normalized = []

    for item in items:
        for mapped_item in normalize_item(item, normalization_map):
            if not mapped_item:
                continue

            if mapped_item in GENERIC_IGNORE_TERMS:
                continue

            normalized.append(mapped_item)

    seen = set()
    unique = []

    for item in normalized:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return unique


def load_ground_truth(normalization_map: dict) -> dict[str, set[str]]:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ground truth file not found: {GROUND_TRUTH_PATH}")

    ground_truth = {}

    with GROUND_TRUTH_PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_id = row.get("image_id", "").strip()

            if not image_id:
                continue

            if resolve_image_path(row.get("image_path", "")) is None:
                raise FileNotFoundError(
                    f"Image not resolved for {image_id}. "
                    "Run the final_images_200 matching fix first."
                )

            raw_items = split_ingredient_list(row.get("visible_ingredients", ""))
            normalized_items = normalize_items(raw_items, normalization_map)

            ground_truth[image_id] = set(normalized_items)

    return ground_truth


def extract_json_from_text(text: str):
    if not isinstance(text, str):
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL,
    )

    if fenced_match:
        try:
            return json.loads(fenced_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    object_match = re.search(r"\{.*\}", text, re.DOTALL)

    if object_match:
        try:
            return json.loads(object_match.group(0).strip())
        except json.JSONDecodeError:
            pass

    return None


def get_parsed_response(row: dict):
    parsed = row.get("parsed_response")

    if isinstance(parsed, dict):
        return parsed

    return extract_json_from_text(row.get("raw_response", ""))


def load_prediction_rows() -> list[dict]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Prediction file not found: {PREDICTIONS_PATH}")

    rows = []

    with PREDICTIONS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return rows


def extract_prediction_items(row: dict, normalization_map: dict) -> list[dict]:
    parsed = get_parsed_response(row)

    if not isinstance(parsed, dict):
        return []

    ingredients = parsed.get("ingredients", [])

    if not isinstance(ingredients, list):
        return []

    extracted = []

    for item in ingredients:
        if not isinstance(item, dict):
            continue

        raw_name = str(item.get("name", "")).strip()
        confidence = str(item.get("confidence", "")).strip().lower()

        if not raw_name:
            continue

        if confidence not in {"high", "medium"}:
            confidence = "unknown"

        normalized_names = normalize_items([raw_name], normalization_map)

        for normalized_name in normalized_names:
            extracted.append({
                "raw_name": raw_name,
                "normalized_name": normalized_name,
                "confidence": confidence,
            })

    return extracted


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_confidence_groups() -> tuple[pd.DataFrame, pd.DataFrame]:
    normalization_map = load_normalization_map(NORMALIZATION_PATH)
    ground_truth = load_ground_truth(normalization_map)
    prediction_rows = load_prediction_rows()

    per_prediction_rows = []
    group_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "total": 0})
    ingredient_tp_counter = Counter()
    ingredient_fp_counter = Counter()

    latest_by_image = {}

    for row in prediction_rows:
        image_id = row.get("image_id", "")

        if image_id:
            latest_by_image[image_id] = row

    for image_id, gt_set in ground_truth.items():
        row = latest_by_image.get(image_id)

        if row is None:
            continue

        prediction_items = extract_prediction_items(row, normalization_map)

        seen_prediction_pairs = set()

        for item in prediction_items:
            normalized_name = item["normalized_name"]
            confidence = item["confidence"]

            pair = (confidence, normalized_name)

            if pair in seen_prediction_pairs:
                continue

            seen_prediction_pairs.add(pair)

            is_true_positive = normalized_name in gt_set
            result = "tp" if is_true_positive else "fp"

            group_counts[confidence]["total"] += 1

            if is_true_positive:
                group_counts[confidence]["tp"] += 1
                ingredient_tp_counter[(confidence, normalized_name)] += 1
            else:
                group_counts[confidence]["fp"] += 1
                ingredient_fp_counter[(confidence, normalized_name)] += 1

            per_prediction_rows.append({
                "image_id": image_id,
                "confidence": confidence,
                "raw_name": item["raw_name"],
                "normalized_name": normalized_name,
                "result": result,
                "is_true_positive": int(is_true_positive),
            })

    group_rows = []

    for confidence in ["high", "medium", "unknown"]:
        counts = group_counts[confidence]
        total = counts["total"]
        tp = counts["tp"]
        fp = counts["fp"]

        if total == 0:
            continue

        precision = safe_divide(tp, total)
        false_positive_rate = safe_divide(fp, total)

        group_rows.append({
            "confidence": confidence,
            "total_predictions": total,
            "true_positives": tp,
            "false_positives": fp,
            "precision": round(precision, 4),
            "false_positive_rate": round(false_positive_rate, 4),
        })

    group_df = pd.DataFrame(group_rows)
    per_prediction_df = pd.DataFrame(per_prediction_rows)

    return group_df, per_prediction_df


def plot_precision_comparison(group_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.bar(group_df["confidence"], group_df["precision"])
    plt.ylim(0, 1)
    plt.xlabel("VLM confidence label")
    plt.ylabel("Precision")
    plt.title("Precision by VLM confidence label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confidence_precision_comparison.png", dpi=300)
    plt.close()


def plot_tp_fp_counts(group_df: pd.DataFrame) -> None:
    labels = group_df["confidence"].tolist()
    tp = group_df["true_positives"].tolist()
    fp = group_df["false_positives"].tolist()

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], tp, width, label="True positives")
    plt.bar([i + width / 2 for i in x], fp, width, label="False positives")
    plt.xticks(list(x), labels)
    plt.ylabel("Prediction count")
    plt.title("TP/FP counts by confidence label")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confidence_tp_fp_counts.png", dpi=300)
    plt.close()


def plot_prediction_volume(group_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.bar(group_df["confidence"], group_df["total_predictions"])
    plt.xlabel("VLM confidence label")
    plt.ylabel("Number of predictions")
    plt.title("Prediction volume by confidence label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confidence_prediction_volume.png", dpi=300)
    plt.close()


def write_summary(group_df: pd.DataFrame, per_prediction_df: pd.DataFrame) -> None:
    table_rows = ""

    for _, row in group_df.iterrows():
        table_rows += (
            f"| {row['confidence']} "
            f"| {int(row['total_predictions'])} "
            f"| {int(row['true_positives'])} "
            f"| {int(row['false_positives'])} "
            f"| {row['precision']:.4f} "
            f"| {row['false_positive_rate']:.4f} |\n"
        )

    high_precision = None
    medium_precision = None

    if "high" in set(group_df["confidence"]):
        high_precision = float(group_df[group_df["confidence"] == "high"]["precision"].iloc[0])

    if "medium" in set(group_df["confidence"]):
        medium_precision = float(group_df[group_df["confidence"] == "medium"]["precision"].iloc[0])

    if high_precision is not None and medium_precision is not None:
        if high_precision > medium_precision:
            interpretation = (
                "High-confidence predictions are more reliable than medium-confidence "
                "predictions, which supports the app's high-confidence filtering choice."
            )
        elif high_precision == medium_precision:
            interpretation = (
                "High- and medium-confidence predictions have similar precision. "
                "The confidence labels are only weakly informative."
            )
        else:
            interpretation = (
                "Medium-confidence predictions are more precise than high-confidence "
                "predictions in this evaluation, so the confidence labels should be treated cautiously."
            )
    else:
        interpretation = (
            "The confidence groups could not both be evaluated because at least one group is missing."
        )

    top_medium_fp = ""

    if not per_prediction_df.empty:
        medium_fp = per_prediction_df[
            (per_prediction_df["confidence"] == "medium")
            & (per_prediction_df["result"] == "fp")
        ]

        if not medium_fp.empty:
            counts = medium_fp["normalized_name"].value_counts().head(10)

            top_medium_fp = "\n".join(
                f"- `{ingredient}` ({count}x)"
                for ingredient, count in counts.items()
            )

    if not top_medium_fp:
        top_medium_fp = "- No medium-confidence false positives found."

    markdown = f"""# Confidence Calibration Analysis

## Inputs

- VLM predictions: `{PREDICTIONS_PATH}`
- Ground truth: `{GROUND_TRUTH_PATH}`
- Normalization: `{NORMALIZATION_PATH}`

This analysis checks whether the VLM's own confidence labels are meaningful by comparing ingredient-level precision for each confidence group.

## Confidence Group Metrics

| Confidence | Predictions | True Positives | False Positives | Precision | False Positive Rate |
|---|---:|---:|---:|---:|---:|
{table_rows}
## Interpretation

{interpretation}

## Top Medium-Confidence False Positives

{top_medium_fp}

## Generated Files

| File | Description |
|---|---|
| `confidence_group_metrics.csv` | Precision and false-positive rate by confidence group |
| `confidence_per_prediction.csv` | Per-prediction TP/FP correctness |
| `figures/confidence_precision_comparison.png` | Precision by confidence label |
| `figures/confidence_tp_fp_counts.png` | True/false positives by confidence label |
| `figures/confidence_prediction_volume.png` | Number of predictions by confidence label |
"""

    SUMMARY_OUTPUT.write_text(markdown, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    group_df, per_prediction_df = evaluate_confidence_groups()

    group_df.to_csv(GROUP_METRICS_OUTPUT, index=False)
    per_prediction_df.to_csv(PER_PREDICTION_OUTPUT, index=False)

    plot_precision_comparison(group_df)
    plot_tp_fp_counts(group_df)
    plot_prediction_volume(group_df)

    write_summary(group_df, per_prediction_df)

    print("Confidence calibration analysis complete.")
    print(f"Group metrics: {GROUP_METRICS_OUTPUT}")
    print(f"Per-prediction results: {PER_PREDICTION_OUTPUT}")
    print(f"Summary report: {SUMMARY_OUTPUT}")
    print()
    print(group_df.to_string(index=False))


if __name__ == "__main__":
    main()
