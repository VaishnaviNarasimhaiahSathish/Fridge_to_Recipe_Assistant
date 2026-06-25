import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.image_resolver import resolve_image_path


GROUND_TRUTH_PATH = Path("data/annotations/manual_ground_truth_100/manual_ground_truth_100.csv")
VLM_OUTPUT_PATH = Path("reports/vlm_predictions_100.jsonl")
NORMALIZATION_PATH = Path("configs/ingredient_normalization.json")

OUTPUT_DIR = Path("reports/evaluation_100")
PER_IMAGE_OUTPUT = OUTPUT_DIR / "vlm_per_image_evaluation.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "vlm_evaluation_summary.md"
FALSE_POSITIVES_OUTPUT = OUTPUT_DIR / "vlm_false_positives.csv"
FALSE_NEGATIVES_OUTPUT = OUTPUT_DIR / "vlm_false_negatives.csv"


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
    "preserve"
}


def load_normalization_map(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Normalization file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def split_ingredient_list(value):
    if value is None:
        return []

    value = str(value).strip()

    if not value:
        return []

    parts = re.split(r"[;,]", value)

    return [part.strip() for part in parts if part.strip()]


def basic_clean_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" .,-")

    return name


def normalize_item(name, normalization_map):
    """
    Returns a list because one raw label can map to multiple labels.

    Example:
        lemon/lime -> ["lemon", "lime"]
    """
    name = basic_clean_name(name)

    if not name:
        return []

    mapped = normalization_map.get(name, name)

    if isinstance(mapped, list):
        return [basic_clean_name(item) for item in mapped if basic_clean_name(item)]

    return [basic_clean_name(mapped)]


def normalize_items(items, normalization_map, ignore_generic=True):
    normalized = []

    for item in items:
        mapped_items = normalize_item(item, normalization_map)

        for mapped_item in mapped_items:
            if not mapped_item:
                continue

            if ignore_generic and mapped_item in GENERIC_IGNORE_TERMS:
                continue

            normalized.append(mapped_item)

    # Remove duplicates while preserving order
    seen = set()
    unique_items = []

    for item in normalized:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    return unique_items


def load_ground_truth(path: Path, normalization_map):
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    ground_truth = {}
    skipped_missing_image = 0

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required_columns = {"image_id", "image_path", "visible_ingredients"}
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(f"Ground truth file is missing columns: {missing_columns}")

        for row in reader:
            image_id = row["image_id"].strip()

            if not image_id:
                continue

            # Some ground truth images live only on a teammate's machine and
            # haven't been pushed to the repo yet — skip them instead of failing.
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


def extract_json_from_text(text):
    if text is None:
        return None

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


def get_parsed_response(row):
    parsed = row.get("parsed_response")

    if isinstance(parsed, dict):
        return parsed

    return extract_json_from_text(row.get("raw_response"))


def extract_vlm_ingredient_names(parsed_response):
    if not isinstance(parsed_response, dict):
        return []

    ingredients = parsed_response.get("ingredients", [])

    if not isinstance(ingredients, list):
        return []

    names = []

    for item in ingredients:
        if isinstance(item, dict):
            name = item.get("name", "")
            if name:
                names.append(name)
        elif isinstance(item, str):
            names.append(item)

    return names


def load_vlm_predictions(path: Path, normalization_map):
    if not path.exists():
        raise FileNotFoundError(f"VLM output file not found: {path}")

    latest_by_image = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_id = row.get("image_id")

            if not image_id:
                continue

            latest_by_image[image_id] = row

    predictions = {}

    for image_id, row in latest_by_image.items():
        parsed_response = get_parsed_response(row)
        raw_vlm_items = extract_vlm_ingredient_names(parsed_response)
        normalized_vlm_items = normalize_items(raw_vlm_items, normalization_map)

        predictions[image_id] = {
            "image_id": image_id,
            "status": row.get("status", ""),
            "model": row.get("model", ""),
            "elapsed_seconds": row.get("elapsed_seconds", ""),
            "finish_reason": row.get("finish_reason", ""),
            "raw_vlm_predictions": raw_vlm_items,
            "normalized_vlm_predictions": normalized_vlm_items,
        }

    return predictions


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_image(image_id, gt_items, vlm_items):
    gt_set = set(gt_items)
    vlm_set = set(vlm_items)

    true_positives = sorted(gt_set & vlm_set)
    false_negatives = sorted(gt_set - vlm_set)
    false_positives = sorted(vlm_set - gt_set)

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    exact_match = int(gt_set == vlm_set)
    jaccard = safe_divide(len(gt_set & vlm_set), len(gt_set | vlm_set))

    return {
        "image_id": image_id,
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


def list_to_string(items):
    return "; ".join(items)


def write_csv(path: Path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normalization_map = load_normalization_map(NORMALIZATION_PATH)
    ground_truth, skipped_missing_image = load_ground_truth(GROUND_TRUTH_PATH, normalization_map)
    vlm_predictions = load_vlm_predictions(VLM_OUTPUT_PATH, normalization_map)

    if skipped_missing_image:
        print(
            f"Skipping {skipped_missing_image} images not found locally "
            "(teammate images pending upload)"
        )

    per_image_rows = []
    false_positive_rows = []
    false_negative_rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    missing_prediction_count = 0

    for image_id, gt_info in ground_truth.items():
        pred_info = vlm_predictions.get(
            image_id,
            {
                "raw_vlm_predictions": [],
                "normalized_vlm_predictions": [],
                "status": "missing",
                "model": "",
                "elapsed_seconds": "",
                "finish_reason": "",
            },
        )

        if pred_info["status"] == "missing":
            missing_prediction_count += 1

        gt_items = gt_info["normalized_ground_truth"]
        vlm_items = pred_info["normalized_vlm_predictions"]

        result = evaluate_image(image_id, gt_items, vlm_items)

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

        per_image_rows.append(
            {
                "image_id": image_id,
                "image_path": gt_info["image_path"],
                "status": pred_info["status"],
                "model": pred_info["model"],
                "elapsed_seconds": pred_info["elapsed_seconds"],
                "finish_reason": pred_info["finish_reason"],
                "raw_ground_truth": list_to_string(gt_info["raw_ground_truth"]),
                "normalized_ground_truth": list_to_string(gt_items),
                "raw_vlm_predictions": list_to_string(pred_info["raw_vlm_predictions"]),
                "normalized_vlm_predictions": list_to_string(vlm_items),
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
            }
        )

        for item in result["false_positives"]:
            false_positive_rows.append(
                {
                    "image_id": image_id,
                    "image_path": gt_info["image_path"],
                    "false_positive": item,
                    "normalized_ground_truth": list_to_string(gt_items),
                    "normalized_vlm_predictions": list_to_string(vlm_items),
                }
            )

        for item in result["false_negatives"]:
            false_negative_rows.append(
                {
                    "image_id": image_id,
                    "image_path": gt_info["image_path"],
                    "false_negative": item,
                    "normalized_ground_truth": list_to_string(gt_items),
                    "normalized_vlm_predictions": list_to_string(vlm_items),
                }
            )

    if not per_image_rows:
        raise ValueError("No images were evaluated. Check your ground truth file.")

    micro_precision = safe_divide(total_tp, total_tp + total_fp)
    micro_recall = safe_divide(total_tp, total_tp + total_fn)
    micro_f1 = safe_divide(
        2 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    )

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

    write_csv(
        PER_IMAGE_OUTPUT,
        per_image_rows,
        fieldnames=list(per_image_rows[0].keys()),
    )

    write_csv(
        FALSE_POSITIVES_OUTPUT,
        false_positive_rows,
        fieldnames=[
            "image_id",
            "image_path",
            "false_positive",
            "normalized_ground_truth",
            "normalized_vlm_predictions",
        ],
    )

    write_csv(
        FALSE_NEGATIVES_OUTPUT,
        false_negative_rows,
        fieldnames=[
            "image_id",
            "image_path",
            "false_negative",
            "normalized_ground_truth",
            "normalized_vlm_predictions",
        ],
    )

    summary = f"""# VLM Ingredient Extraction Evaluation Summary

## Input Files

- Ground truth: `{GROUND_TRUTH_PATH}`
- VLM predictions: `{VLM_OUTPUT_PATH}`
- Normalization file: `{NORMALIZATION_PATH}`

## Dataset

- Number of evaluated images: {len(per_image_rows)}
- Missing VLM prediction rows: {missing_prediction_count}
- Skipped (image not found locally, teammate images pending upload): {skipped_missing_image}

## Micro-Averaged Metrics

Micro metrics aggregate true positives, false positives, and false negatives across all images.

| Metric | Value |
|---|---:|
| True Positives | {total_tp} |
| False Positives | {total_fp} |
| False Negatives | {total_fn} |
| Precision | {micro_precision:.4f} |
| Recall | {micro_recall:.4f} |
| F1-score | {micro_f1:.4f} |

## Macro-Averaged Metrics

Macro metrics calculate the metric per image and average across images.

| Metric | Value |
|---|---:|
| Precision | {macro_precision:.4f} |
| Recall | {macro_recall:.4f} |
| F1-score | {macro_f1:.4f} |

## Accuracy-Like Metrics

These are reported because standard classification accuracy is not suitable for open-vocabulary multi-label ingredient extraction.

| Metric | Value |
|---|---:|
| Exact Match Accuracy | {exact_match_accuracy:.4f} |
| Mean Jaccard Similarity | {mean_jaccard:.4f} |

## Output Files

- Per-image results: `{PER_IMAGE_OUTPUT}`
- False positives: `{FALSE_POSITIVES_OUTPUT}`
- False negatives: `{FALSE_NEGATIVES_OUTPUT}`

## Notes

- Evaluation is based on normalized ingredient names.
- Generic uncertain items such as `unknown bottle`, `unknown jar`, `beverage`, `condiment`, `prepared food`, and `leftover food` are excluded from the main ingredient metrics.
- Exact match accuracy checks whether the complete predicted ingredient set exactly matches the ground truth set for an image.
- Mean Jaccard similarity measures average set overlap between predicted and ground-truth ingredients.
"""

    SUMMARY_OUTPUT.write_text(summary, encoding="utf-8")

    print("Evaluation complete.")
    print(f"Per-image results: {PER_IMAGE_OUTPUT}")
    print(f"Summary report: {SUMMARY_OUTPUT}")
    print(f"False positives: {FALSE_POSITIVES_OUTPUT}")
    print(f"False negatives: {FALSE_NEGATIVES_OUTPUT}")
    print()
    print(f"Evaluated images: {len(per_image_rows)}")
    print(f"Missing VLM prediction rows: {missing_prediction_count}")
    print(f"Skipped (image not found locally): {skipped_missing_image}")
    print()
    print("Micro metrics:")
    print(f"Precision: {micro_precision:.4f}")
    print(f"Recall:    {micro_recall:.4f}")
    print(f"F1-score:  {micro_f1:.4f}")
    print()
    print("Accuracy-like metrics:")
    print(f"Exact Match Accuracy:    {exact_match_accuracy:.4f}")
    print(f"Mean Jaccard Similarity: {mean_jaccard:.4f}")


if __name__ == "__main__":
    main()