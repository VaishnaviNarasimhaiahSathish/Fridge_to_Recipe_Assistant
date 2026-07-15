import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.image_resolver import resolve_image_path


GROUND_TRUTH_PATH = Path("data/annotations/manual_ground_truth_100/manual_ground_truth_100.csv")
VLM_OUTPUT_PATH = Path("reports/vlm_predictions_100.jsonl")
NORMALIZATION_PATH = Path("configs/ingredient_normalization.json")

GEMMA_REVIEWED_PATH = Path("data/annotations/gemma4_batch_100/gemma4_annotations_reviewed.csv")
GEMMA_RAW_PATH = Path("data/annotations/gemma4_batch_100/gemma4_annotations_raw.csv")

OUTPUT_DIR = Path("reports/final_evaluation")
FIGURES_DIR = OUTPUT_DIR / "figures"

FINAL_METRICS_CSV = OUTPUT_DIR / "final_vlm_metrics.csv"
FINAL_CONFIDENCE_CSV = OUTPUT_DIR / "final_confidence_comparison.csv"
FINAL_PER_IMAGE_HIGH_CSV = OUTPUT_DIR / "final_per_image_high_only.csv"
FINAL_FP_HIGH_CSV = OUTPUT_DIR / "final_false_positives_high_only.csv"
FINAL_FN_HIGH_CSV = OUTPUT_DIR / "final_false_negatives_high_only.csv"

FINAL_METRICS_MD = OUTPUT_DIR / "final_vlm_metrics.md"
FINAL_CONFIDENCE_MD = OUTPUT_DIR / "final_confidence_comparison.md"
FINAL_README_MD = OUTPUT_DIR / "final_readme_metrics.md"


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


CONTEXT_GUESS_TERMS = {
    "water",
    "juice",
    "orange juice",
    "lime juice",
    "lemon juice",
    "apple juice",
    "soda",
    "beer",
    "wine",
    "broth",
    "cider",
    "lemonade",
    "ice",
    "ice water",
    "sparkling water",
}


COMMON_FRIDGE_DEFAULT_TERMS = {
    "butter",
    "milk",
    "cheese",
    "yogurt",
    "egg",
    "mayonnaise",
    "ketchup",
    "mustard",
    "salad dressing",
    "cream cheese",
    "sour cream",
    "margarine",
    "almond milk",
    "whipped cream",
    "half & half",
    "cottage cheese",
}


AMBIGUOUS_VISUAL_TERMS = {
    "meat",
    "salad",
    "bread",
    "lettuce",
    "carrot",
    "tomato",
    "pickle",
    "hot sauce",
    "jam",
    "soy sauce",
    "lemon",
    "lime",
    "mushroom",
    "strawberry",
    "avocado",
    "blueberry",
    "spinach",
    "onion",
    "celery",
    "zucchini",
    "pepper",
    "bell pepper",
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
        mapped_items = normalize_item(item, normalization_map)

        for mapped_item in mapped_items:
            if not mapped_item:
                continue

            if mapped_item in GENERIC_IGNORE_TERMS:
                continue

            normalized.append(mapped_item)

    seen = set()
    unique_items = []

    for item in normalized:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    return unique_items


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


def extract_vlm_ingredient_names(row: dict, strategy: str) -> list[str]:
    parsed = get_parsed_response(row)

    if not isinstance(parsed, dict):
        return []

    ingredients = parsed.get("ingredients", [])

    if not isinstance(ingredients, list):
        return []

    accepted_confidences = {"high"} if strategy == "high_only" else {"high", "medium"}

    names = []

    for item in ingredients:
        if isinstance(item, dict):
            confidence = str(item.get("confidence", "")).strip().lower()
            name = str(item.get("name", "")).strip()

            if name and confidence in accepted_confidences:
                names.append(name)

        elif isinstance(item, str) and strategy == "full_predictions":
            names.append(item)

    return names


def load_ground_truth(normalization_map: dict) -> tuple[dict, int]:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ground truth file not found: {GROUND_TRUTH_PATH}")

    ground_truth = {}
    skipped_missing_image = 0

    with GROUND_TRUTH_PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_id = row.get("image_id", "").strip()

            if not image_id:
                continue

            if resolve_image_path(row.get("image_path", "")) is None:
                skipped_missing_image += 1
                continue

            raw_items = split_ingredient_list(row.get("visible_ingredients", ""))
            normalized_items = normalize_items(raw_items, normalization_map)

            ground_truth[image_id] = {
                "image_id": image_id,
                "image_path": row.get("image_path", ""),
                "raw_ground_truth": raw_items,
                "normalized_ground_truth": normalized_items,
            }

    return ground_truth, skipped_missing_image


def load_vlm_predictions() -> dict:
    if not VLM_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"VLM prediction file not found: {VLM_OUTPUT_PATH}")

    latest_by_image = {}

    with VLM_OUTPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_id = row.get("image_id")

            if image_id:
                latest_by_image[image_id] = row

    return latest_by_image


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_sets(gt_items: list[str], pred_items: list[str]) -> dict:
    gt_set = set(gt_items)
    pred_set = set(pred_items)

    true_positives = sorted(gt_set & pred_set)
    false_positives = sorted(pred_set - gt_set)
    false_negatives = sorted(gt_set - pred_set)

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    exact_match = int(gt_set == pred_set)
    jaccard = safe_divide(len(gt_set & pred_set), len(gt_set | pred_set))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
        "jaccard": jaccard,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def list_to_string(items: list[str]) -> str:
    return "; ".join(items)


def compute_strategy_metrics(
    strategy: str,
    ground_truth: dict,
    predictions: dict,
    normalization_map: dict,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    per_image_rows = []
    false_positive_rows = []
    false_negative_rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    missing_prediction_count = 0

    for image_id, gt_info in ground_truth.items():
        pred_row = predictions.get(image_id)

        if pred_row is None:
            missing_prediction_count += 1
            raw_pred_items = []
            model = ""
            status = "missing"
            elapsed_seconds = ""
        else:
            raw_pred_items = extract_vlm_ingredient_names(pred_row, strategy)
            model = pred_row.get("model", "")
            status = pred_row.get("status", "")
            elapsed_seconds = pred_row.get("elapsed_seconds", "")

        pred_items = normalize_items(raw_pred_items, normalization_map)
        gt_items = gt_info["normalized_ground_truth"]

        result = evaluate_sets(gt_items, pred_items)

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

        per_image_rows.append({
            "image_id": image_id,
            "image_path": gt_info["image_path"],
            "strategy": strategy,
            "status": status,
            "model": model,
            "elapsed_seconds": elapsed_seconds,
            "normalized_ground_truth": list_to_string(gt_items),
            "normalized_vlm_predictions": list_to_string(pred_items),
            "tp": result["tp"],
            "fp": result["fp"],
            "fn": result["fn"],
            "precision": round(result["precision"], 4),
            "recall": round(result["recall"], 4),
            "f1": round(result["f1"], 4),
            "exact_match": result["exact_match"],
            "jaccard": round(result["jaccard"], 4),
            "true_positives": list_to_string(result["true_positives"]),
            "false_positives": list_to_string(result["false_positives"]),
            "false_negatives": list_to_string(result["false_negatives"]),
        })

        for item in result["false_positives"]:
            false_positive_rows.append({
                "image_id": image_id,
                "image_path": gt_info["image_path"],
                "strategy": strategy,
                "false_positive": item,
            })

        for item in result["false_negatives"]:
            false_negative_rows.append({
                "image_id": image_id,
                "image_path": gt_info["image_path"],
                "strategy": strategy,
                "false_negative": item,
            })

    micro_precision = safe_divide(total_tp, total_tp + total_fp)
    micro_recall = safe_divide(total_tp, total_tp + total_fn)
    micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    macro_precision = safe_divide(
        sum(row["precision"] for row in per_image_rows),
        len(per_image_rows),
    )
    macro_recall = safe_divide(
        sum(row["recall"] for row in per_image_rows),
        len(per_image_rows),
    )
    macro_f1 = safe_divide(
        sum(row["f1"] for row in per_image_rows),
        len(per_image_rows),
    )

    exact_match_accuracy = safe_divide(
        sum(row["exact_match"] for row in per_image_rows),
        len(per_image_rows),
    )
    mean_jaccard = safe_divide(
        sum(row["jaccard"] for row in per_image_rows),
        len(per_image_rows),
    )

    metrics = {
        "strategy": strategy,
        "images_evaluated": len(per_image_rows),
        "missing_prediction_count": missing_prediction_count,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "exact_match_accuracy": round(exact_match_accuracy, 4),
        "mean_jaccard": round(mean_jaccard, 4),
    }

    return metrics, per_image_rows, false_positive_rows, false_negative_rows


def categorize_false_positive(ingredient: str) -> str:
    ing = basic_clean_name(ingredient)

    if ing in CONTEXT_GUESS_TERMS:
        return "context_guess"

    if ing in COMMON_FRIDGE_DEFAULT_TERMS:
        return "common_fridge_default"

    if ing in AMBIGUOUS_VISUAL_TERMS:
        return "ambiguous_visual"

    return "other"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_main_metrics(metrics: dict) -> None:
    labels = [
        "Precision",
        "Recall",
        "F1",
        "Jaccard",
        "Exact match",
    ]
    values = [
        metrics["micro_precision"],
        metrics["micro_recall"],
        metrics["micro_f1"],
        metrics["mean_jaccard"],
        metrics["exact_match_accuracy"],
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Final VLM ingredient extraction metrics")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_main_metrics.png", dpi=300)
    plt.close()


def plot_confidence_comparison(metrics_rows: list[dict]) -> None:
    labels = [
        "Precision",
        "Recall",
        "F1",
    ]

    x = range(len(labels))
    width = 0.35

    full = next(row for row in metrics_rows if row["strategy"] == "full_predictions")
    high = next(row for row in metrics_rows if row["strategy"] == "high_only")

    full_values = [
        full["micro_precision"],
        full["micro_recall"],
        full["micro_f1"],
    ]
    high_values = [
        high["micro_precision"],
        high["micro_recall"],
        high["micro_f1"],
    ]

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], full_values, width, label="High + medium")
    plt.bar([i + width / 2 for i in x], high_values, width, label="High only")
    plt.xticks(list(x), labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Confidence filtering comparison")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_confidence_comparison.png", dpi=300)
    plt.close()


def plot_precision_vs_recall(per_image_rows: list[dict]) -> None:
    df = pd.DataFrame(per_image_rows)

    plt.figure(figsize=(7, 6))
    plt.scatter(df["recall"], df["precision"], alpha=0.75)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs recall per image")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_precision_vs_recall.png", dpi=300)
    plt.close()


def plot_top_errors(fp_rows: list[dict], fn_rows: list[dict]) -> None:
    fp_counts = Counter(row["false_positive"] for row in fp_rows)
    fn_counts = Counter(row["false_negative"] for row in fn_rows)

    if fp_counts:
        top_fp = fp_counts.most_common(20)
        labels = [item for item, _ in top_fp][::-1]
        values = [count for _, count in top_fp][::-1]

        plt.figure(figsize=(10, 7))
        plt.barh(labels, values)
        plt.xlabel("Frequency")
        plt.title("Final top false positive ingredients")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "final_top_false_positives.png", dpi=300)
        plt.close()

    if fn_counts:
        top_fn = fn_counts.most_common(20)
        labels = [item for item, _ in top_fn][::-1]
        values = [count for _, count in top_fn][::-1]

        plt.figure(figsize=(10, 7))
        plt.barh(labels, values)
        plt.xlabel("Frequency")
        plt.title("Final top false negative ingredients")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "final_top_false_negatives.png", dpi=300)
        plt.close()


def plot_error_categories(fp_rows: list[dict]) -> dict:
    categorized = []

    for row in fp_rows:
        category = categorize_false_positive(row["false_positive"])
        categorized.append({
            **row,
            "error_category": category,
        })

    category_counts = Counter(row["error_category"] for row in categorized)

    if category_counts:
        labels = []
        values = []

        display_names = {
            "common_fridge_default": "Common fridge default",
            "ambiguous_visual": "Ambiguous visual",
            "context_guess": "Context guess",
            "other": "Other",
        }

        for category, count in category_counts.most_common():
            labels.append(display_names.get(category, category))
            values.append(count)

        plt.figure(figsize=(7, 5))
        plt.pie(values, labels=labels, autopct="%1.0f%%", startangle=140)
        plt.title("Final false positive error categories")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "final_error_categories.png", dpi=300)
        plt.close()

    write_csv(OUTPUT_DIR / "final_false_positives_high_only_categorized.csv", categorized)

    return dict(category_counts)


def compute_gemma_metrics(normalization_map: dict) -> dict | None:
    if not GEMMA_REVIEWED_PATH.exists():
        return None

    with GEMMA_REVIEWED_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    total_tp = total_fp = total_fn = 0
    per_image = []
    gemma_counts = []
    qwen_counts = []

    for row in rows:
        gemma_raw = split_ingredient_list(row.get("visible_ingredients", ""))
        qwen_raw = split_ingredient_list(row.get("corrected_visible_ingredients", ""))

        gemma_items = normalize_items(gemma_raw, normalization_map)
        qwen_items = normalize_items(qwen_raw, normalization_map)

        gemma_counts.append(len(gemma_items))
        qwen_counts.append(len(qwen_items))

        result = evaluate_sets(qwen_items, gemma_items)

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

        per_image.append(result)

    micro_precision = safe_divide(total_tp, total_tp + total_fp)
    micro_recall = safe_divide(total_tp, total_tp + total_fn)
    micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    return {
        "images": len(per_image),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "mean_jaccard": round(sum(r["jaccard"] for r in per_image) / len(per_image), 4),
        "avg_gemma_ingredients": round(sum(gemma_counts) / len(gemma_counts), 2),
        "avg_qwen_reference_ingredients": round(sum(qwen_counts) / len(qwen_counts), 2),
    }


def compute_latency_summary(per_image_rows: list[dict]) -> tuple[dict | None, dict | None]:
    qwen_times = []

    for row in per_image_rows:
        value = str(row.get("elapsed_seconds", "")).strip()

        if not value:
            continue

        try:
            qwen_times.append(float(value))
        except ValueError:
            continue

    qwen_summary = None

    if qwen_times:
        qwen_summary = {
            "model": "Qwen",
            "images": len(qwen_times),
            "min_seconds": round(min(qwen_times), 2),
            "max_seconds": round(max(qwen_times), 2),
            "mean_seconds": round(sum(qwen_times) / len(qwen_times), 2),
            "median_seconds": round(float(pd.Series(qwen_times).median()), 2),
        }

    gemma_summary = None

    if GEMMA_RAW_PATH.exists():
        gemma_times = []

        with GEMMA_RAW_PATH.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                value = str(row.get("elapsed_seconds", "")).strip()

                if not value:
                    continue

                try:
                    gemma_times.append(float(value))
                except ValueError:
                    continue

        if gemma_times:
            gemma_summary = {
                "model": "Gemma",
                "images": len(gemma_times),
                "min_seconds": round(min(gemma_times), 2),
                "max_seconds": round(max(gemma_times), 2),
                "mean_seconds": round(sum(gemma_times) / len(gemma_times), 2),
                "median_seconds": round(float(pd.Series(gemma_times).median()), 2),
            }

    return qwen_summary, gemma_summary


def plot_model_comparisons(
    high_metrics: dict,
    gemma_metrics: dict | None,
    qwen_latency: dict | None,
    gemma_latency: dict | None,
) -> None:
    if gemma_metrics:
        labels = ["Qwen app setting", "Gemma raw"]
        precision = [high_metrics["micro_precision"], gemma_metrics["micro_precision"]]
        recall = [high_metrics["micro_recall"], gemma_metrics["micro_recall"]]
        f1 = [high_metrics["micro_f1"], gemma_metrics["micro_f1"]]

        x = range(len(labels))
        width = 0.25

        plt.figure(figsize=(8, 5))
        plt.bar([i - width for i in x], precision, width, label="Precision")
        plt.bar(list(x), recall, width, label="Recall")
        plt.bar([i + width for i in x], f1, width, label="F1")
        plt.xticks(list(x), labels)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("Final model comparison")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "final_gemma_vs_qwen.png", dpi=300)
        plt.close()

    latency_rows = [row for row in [qwen_latency, gemma_latency] if row]

    if latency_rows:
        labels = [row["model"] for row in latency_rows]
        means = [row["mean_seconds"] for row in latency_rows]
        medians = [row["median_seconds"] for row in latency_rows]

        x = range(len(labels))
        width = 0.35

        plt.figure(figsize=(7, 5))
        plt.bar([i - width / 2 for i in x], means, width, label="Mean")
        plt.bar([i + width / 2 for i in x], medians, width, label="Median")
        plt.xticks(list(x), labels)
        plt.ylabel("Seconds")
        plt.title("Final latency comparison")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "final_latency_comparison.png", dpi=300)
        plt.close()


def markdown_metric_table(metrics_rows: list[dict]) -> str:
    rows = ""

    for row in metrics_rows:
        label = "High + medium predictions" if row["strategy"] == "full_predictions" else "High confidence only"

        rows += (
            f"| {label} "
            f"| {row['images_evaluated']} "
            f"| {row['tp']} "
            f"| {row['fp']} "
            f"| {row['fn']} "
            f"| {row['micro_precision']:.4f} "
            f"| {row['micro_recall']:.4f} "
            f"| {row['micro_f1']:.4f} "
            f"| {row['mean_jaccard']:.4f} "
            f"| {row['exact_match_accuracy']:.4f} |\n"
        )

    return rows


def write_markdown_reports(
    metrics_rows: list[dict],
    high_metrics: dict,
    category_counts: dict,
    gemma_metrics: dict | None,
    qwen_latency: dict | None,
    gemma_latency: dict | None,
    skipped_missing_image: int,
) -> None:
    final_metrics = f"""# Final VLM Evaluation Metrics

## Inputs

- Ground truth: `{GROUND_TRUTH_PATH}`
- VLM predictions: `{VLM_OUTPUT_PATH}`
- Normalization: `{NORMALIZATION_PATH}`

Skipped images because the image file was not found locally: **{skipped_missing_image}**

## Final Metrics

| Strategy | Images | TP | FP | FN | Precision | Recall | Micro F1 | Mean Jaccard | Exact Match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{markdown_metric_table(metrics_rows)}
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
"""

    FINAL_METRICS_MD.write_text(final_metrics, encoding="utf-8")

    confidence_md = f"""# Final Confidence Filtering Comparison

| Strategy | TP | FP | FN | Precision | Recall | Micro F1 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(
    f"| {'High + medium' if row['strategy'] == 'full_predictions' else 'High only'} "
    f"| {row['tp']} | {row['fp']} | {row['fn']} "
    f"| {row['micro_precision']:.4f} | {row['micro_recall']:.4f} | {row['micro_f1']:.4f} |"
    for row in metrics_rows
)}

## Key message

High-only filtering usually reduces false positives and improves trustworthiness for recipe recommendation, but it can also reduce recall by dropping correct medium-confidence predictions.
"""

    FINAL_CONFIDENCE_MD.write_text(confidence_md, encoding="utf-8")

    category_total = sum(category_counts.values()) if category_counts else 0

    category_lines = ""

    display_names = {
        "common_fridge_default": "Common fridge default",
        "ambiguous_visual": "Ambiguous visual",
        "context_guess": "Context guess",
        "other": "Other",
    }

    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        share = safe_divide(count, category_total) * 100
        category_lines += f"- {display_names.get(category, category)}: {count} ({share:.0f}%)\n"

    gemma_section = ""

    if gemma_metrics:
        gemma_section = f"""
## Gemma Comparison

Gemma was evaluated against the Qwen-reviewed correction set as a proxy reference.

| Metric | Value |
|---|---:|
| Images | {gemma_metrics['images']} |
| Precision | {gemma_metrics['micro_precision']:.4f} |
| Recall | {gemma_metrics['micro_recall']:.4f} |
| Micro F1 | {gemma_metrics['micro_f1']:.4f} |
| Mean Jaccard | {gemma_metrics['mean_jaccard']:.4f} |
| Avg Gemma ingredients/image | {gemma_metrics['avg_gemma_ingredients']:.2f} |
| Avg Qwen-reference ingredients/image | {gemma_metrics['avg_qwen_reference_ingredients']:.2f} |
"""

    latency_section = ""

    if qwen_latency or gemma_latency:
        rows = ""

        for item in [qwen_latency, gemma_latency]:
            if not item:
                continue

            rows += (
                f"| {item['model']} "
                f"| {item['images']} "
                f"| {item['mean_seconds']:.2f}s "
                f"| {item['median_seconds']:.2f}s "
                f"| {item['min_seconds']:.2f}s "
                f"| {item['max_seconds']:.2f}s |\n"
            )

        latency_section = f"""
## Latency Summary

| Model | Images | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
{rows}
"""

    readme_md = f"""# Final README Metrics Block

Use this block to update the README after reviewing the final outputs.

## Final Ingredient Extraction Evaluation

The final 100-image evaluation was computed against manually reviewed fridge-image ground truth using the updated normalization rules.

| Strategy | Precision | Recall | Micro F1 | Mean Jaccard | Exact Match |
|---|---:|---:|---:|---:|---:|
| High + medium predictions | {next(row for row in metrics_rows if row['strategy'] == 'full_predictions')['micro_precision']:.4f} | {next(row for row in metrics_rows if row['strategy'] == 'full_predictions')['micro_recall']:.4f} | {next(row for row in metrics_rows if row['strategy'] == 'full_predictions')['micro_f1']:.4f} | {next(row for row in metrics_rows if row['strategy'] == 'full_predictions')['mean_jaccard']:.4f} | {next(row for row in metrics_rows if row['strategy'] == 'full_predictions')['exact_match_accuracy']:.4f} |
| High confidence only app setting | {high_metrics['micro_precision']:.4f} | {high_metrics['micro_recall']:.4f} | {high_metrics['micro_f1']:.4f} | {high_metrics['mean_jaccard']:.4f} | {high_metrics['exact_match_accuracy']:.4f} |

The deployed app uses the **high confidence only** setting to reduce unreliable ingredient guesses before recipe recommendation.

## Error Analysis

False positives in the high-confidence setting were grouped into practical categories:

{category_lines or "- No false positives found.\n"}
{gemma_section}
{latency_section}
## Recommended README Figures

Include only these figures in the README:

1. `reports/final_evaluation/figures/final_main_metrics.png`
2. `reports/final_evaluation/figures/final_confidence_comparison.png`
3. `reports/final_evaluation/figures/final_error_categories.png`
4. `reports/final_evaluation/figures/final_gemma_vs_qwen.png`
5. `reports/final_evaluation/figures/final_latency_comparison.png`

Keep the larger false-positive and false-negative charts in `reports/final_evaluation/figures/` for presentation backup.
"""

    FINAL_README_MD.write_text(readme_md, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    normalization_map = load_normalization_map(NORMALIZATION_PATH)
    ground_truth, skipped_missing_image = load_ground_truth(normalization_map)
    predictions = load_vlm_predictions()

    full_metrics, full_per_image, full_fp, full_fn = compute_strategy_metrics(
        strategy="full_predictions",
        ground_truth=ground_truth,
        predictions=predictions,
        normalization_map=normalization_map,
    )

    high_metrics, high_per_image, high_fp, high_fn = compute_strategy_metrics(
        strategy="high_only",
        ground_truth=ground_truth,
        predictions=predictions,
        normalization_map=normalization_map,
    )

    metrics_rows = [full_metrics, high_metrics]

    write_csv(FINAL_METRICS_CSV, metrics_rows)
    write_csv(FINAL_CONFIDENCE_CSV, metrics_rows)
    write_csv(FINAL_PER_IMAGE_HIGH_CSV, high_per_image)
    write_csv(FINAL_FP_HIGH_CSV, high_fp)
    write_csv(FINAL_FN_HIGH_CSV, high_fn)

    plot_main_metrics(high_metrics)
    plot_confidence_comparison(metrics_rows)
    plot_precision_vs_recall(high_per_image)
    plot_top_errors(high_fp, high_fn)
    category_counts = plot_error_categories(high_fp)

    gemma_metrics = compute_gemma_metrics(normalization_map)
    qwen_latency, gemma_latency = compute_latency_summary(high_per_image)

    plot_model_comparisons(
        high_metrics=high_metrics,
        gemma_metrics=gemma_metrics,
        qwen_latency=qwen_latency,
        gemma_latency=gemma_latency,
    )

    write_markdown_reports(
        metrics_rows=metrics_rows,
        high_metrics=high_metrics,
        category_counts=category_counts,
        gemma_metrics=gemma_metrics,
        qwen_latency=qwen_latency,
        gemma_latency=gemma_latency,
        skipped_missing_image=skipped_missing_image,
    )

    print("Final evaluation summary created.")
    print(f"Output folder: {OUTPUT_DIR}")
    print()
    print("Key outputs:")
    print(f"- {FINAL_METRICS_MD}")
    print(f"- {FINAL_CONFIDENCE_MD}")
    print(f"- {FINAL_README_MD}")
    print(f"- {FIGURES_DIR}")


if __name__ == "__main__":
    main()