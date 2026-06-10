import csv
import json
import re
from pathlib import Path


VLM_OUTPUT_PATH    = Path("reports/vlm_predictions_100.jsonl")
GROUND_TRUTH_PATH  = Path("reports/manual_ground_truth_100.csv")
NORMALIZATION_PATH = Path("configs/ingredient_normalization.json")

OUTPUT_DIR = Path("reports/confidence_filtering_100")

# We test filtering out "medium" confidence predictions
# and keeping only "high" confidence predictions
CONFIDENCE_LEVELS = ["high", "medium"]


GENERIC_IGNORE_TERMS = {
    "unknown jar", "unknown bottle", "unknown packaged item",
    "unknown container", "unknown item", "unknown food item",
    "food", "drink", "beverage", "condiment", "container",
    "package", "packaged item", "prepared food", "prepared meal",
    "prepared salad", "leftover food", "frozen food", "canned food",
    "canned fruit", "sauce", "bottle", "jar", "grocery", "item",
    "green", "liquid", "leftover", "fruit", "vegetable", "vegetables",
    "chopped vegetables", "frozen vegetable", "leafy green vegetable",
    "dressing", "dips", "snack", "dessert", "spread", "preserve",
    # Added from Phase 1 error analysis — high-frequency context guesses
    "water", "juice", "orange juice", "lime juice", "soda",
    "broth", "beer", "wine", "cider", "lemonade", "ice",
}


def load_normalization_map(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth(path: Path) -> dict:
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            raw = row.get("visible_ingredients", "")
            gt[image_id] = raw
    return gt


def split_ingredients(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[;,]", value)
    return [p.strip().lower() for p in parts if p.strip()]


def normalize(name: str, norm_map: dict) -> str:
    result = norm_map.get(name.strip().lower(), name.strip().lower())
    # Some normalization values are lists (e.g. lemon/lime → ['lemon', 'lime'])
    # Take the first element in that case
    if isinstance(result, list):
        return result[0] if result else name.strip().lower()
    return result


def load_vlm_predictions(path: Path) -> dict:
    predictions = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = row.get("image_id")
            if image_id:
                predictions[image_id] = row
    return predictions


def extract_ingredients_at_confidence(parsed_response: dict, min_confidence: str) -> list[str]:
    """
    Returns ingredient names at or above the given minimum confidence level.
    Confidence hierarchy: high > medium
    min_confidence="high"   → keep only high
    min_confidence="medium" → keep high and medium (baseline, no filtering)
    """
    accepted = {"high"} if min_confidence == "high" else {"high", "medium"}
    names = []
    for ing in parsed_response.get("ingredients", []):
        conf = str(ing.get("confidence", "")).strip().lower()
        if conf in accepted:
            names.append(ing.get("name", "").strip().lower())
    return names


def evaluate(gt: dict, predictions: dict, norm_map: dict, min_confidence: str) -> dict:
    total_tp = total_fp = total_fn = 0
    per_image = []

    for image_id, raw_gt in gt.items():
        pred_row = predictions.get(image_id)

        # Normalize ground truth
        gt_ingredients = split_ingredients(raw_gt)
        gt_normalized = {
            normalize(i, norm_map) for i in gt_ingredients
            if normalize(i, norm_map) not in GENERIC_IGNORE_TERMS
        }

        # Get VLM predictions at this confidence level
        if pred_row and pred_row.get("status") == "success" and pred_row.get("parsed_response"):
            raw_preds = extract_ingredients_at_confidence(
                pred_row["parsed_response"], min_confidence
            )
        else:
            raw_preds = []

        pred_normalized = {
            normalize(i, norm_map) for i in raw_preds
            if normalize(i, norm_map) not in GENERIC_IGNORE_TERMS
        }

        tp = gt_normalized & pred_normalized
        fp = pred_normalized - gt_normalized
        fn = gt_normalized - pred_normalized

        precision = len(tp) / len(pred_normalized) if pred_normalized else 0.0
        recall    = len(tp) / len(gt_normalized)   if gt_normalized   else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        per_image.append({
            "image_id":          image_id,
            "min_confidence":    min_confidence,
            "tp":                len(tp),
            "fp":                len(fp),
            "fn":                len(fn),
            "precision":         round(precision, 4),
            "recall":            round(recall, 4),
            "f1":                round(f1, 4),
        })

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1        = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                       if (micro_precision + micro_recall) > 0 else 0.0)

    return {
        "min_confidence": min_confidence,
        "total_tp":       total_tp,
        "total_fp":       total_fp,
        "total_fn":       total_fn,
        "micro_precision": round(micro_precision, 4),
        "micro_recall":    round(micro_recall, 4),
        "micro_f1":        round(micro_f1, 4),
        "per_image":       per_image,
    }


def write_summary(results: list[dict], output_dir: Path):
    out_path = output_dir / "vlm_confidence_filtering_summary.md"

    rows = ""
    for r in results:
        label = "high only" if r["min_confidence"] == "high" else "high + medium (baseline)"
        rows += (
            f"| {label} | {r['total_tp']} | {r['total_fp']} | {r['total_fn']} "
            f"| {r['micro_precision']:.4f} | {r['micro_recall']:.4f} | {r['micro_f1']:.4f} |\n"
        )

    summary = f"""# VLM Confidence Filtering Summary

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
{rows}
## Input Files

- VLM predictions: `reports/vlm_predictions_100.jsonl`
- Ground truth: `reports/manual_ground_truth_100.csv`
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
"""

    out_path.write_text(summary, encoding="utf-8")
    print(f"Saved: {out_path}")


def write_per_image_csv(results: list[dict], output_dir: Path):
    all_rows = []
    for r in results:
        all_rows.extend(r["per_image"])

    out_path = output_dir / "vlm_confidence_per_image.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    norm_map    = load_normalization_map(NORMALIZATION_PATH)
    gt          = load_ground_truth(GROUND_TRUTH_PATH)
    predictions = load_vlm_predictions(VLM_OUTPUT_PATH)

    results = []
    for level in CONFIDENCE_LEVELS:
        result = evaluate(gt, predictions, norm_map, level)
        results.append(result)

        label = "high only" if level == "high" else "baseline (high + medium)"
        print(f"\nStrategy: {label}")
        print(f"  TP={result['total_tp']}  FP={result['total_fp']}  FN={result['total_fn']}")
        print(f"  Precision : {result['micro_precision']:.4f}")
        print(f"  Recall    : {result['micro_recall']:.4f}")
        print(f"  Micro F1  : {result['micro_f1']:.4f}")

    write_per_image_csv(results, OUTPUT_DIR)
    write_summary(results, OUTPUT_DIR)

    print()
    print("Confidence filtering complete.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()