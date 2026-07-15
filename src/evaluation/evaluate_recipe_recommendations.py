import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingredients.prediction_processing import (
    extract_detected_ingredients,
    load_normalization_map,
)
from src.recipe.retrieve_recipes_hybrid import (
    load_recipes,
    retrieve_recipes_hybrid,
)


PREDICTIONS_PATH = Path("reports/vlm_predictions_100.jsonl")
NORMALIZATION_PATH = Path("configs/ingredient_normalization.json")
RECIPES_PATH = Path("data/recipes.json")

OUTPUT_DIR = Path("reports/recipe_recommendation_evaluation")
FIGURES_DIR = OUTPUT_DIR / "figures"

PER_IMAGE_OUTPUT = OUTPUT_DIR / "recipe_recommendation_per_image.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "recipe_recommendation_summary.md"

TOP_K = 5
MATCH_THRESHOLDS = [0.5, 0.7]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_prediction_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return rows


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def get_recipe_coverage(recipe: dict) -> float:
    return float(recipe.get("coverage", 0.0))


def get_missing_count(recipe: dict) -> int:
    return len(recipe.get("missing", []))


def get_matched_count(recipe: dict) -> int:
    return len(recipe.get("matched", []))


def evaluate_predictions() -> tuple[pd.DataFrame, dict]:
    normalization_map = load_normalization_map(NORMALIZATION_PATH)
    prediction_rows = load_prediction_rows(PREDICTIONS_PATH)
    recipes = load_recipes(RECIPES_PATH)

    per_image_rows = []

    for row in prediction_rows:
        image_id = row.get("image_id", "")

        detected_ingredients = extract_detected_ingredients(
            prediction_row=row,
            normalization_map=normalization_map,
            allowed_confidences={"high"},
        )

        recommendations = retrieve_recipes_hybrid(
            available_ingredients=detected_ingredients,
            top_n=TOP_K,
            candidate_limit=250,
            recipes=recipes,
            ranking_mode="all",
        )

        top_recipe = recommendations[0] if recommendations else {}

        top_coverages = [
            get_recipe_coverage(recipe)
            for recipe in recommendations
        ]

        top_missing_counts = [
            get_missing_count(recipe)
            for recipe in recommendations
        ]

        top_matched_counts = [
            get_matched_count(recipe)
            for recipe in recommendations
        ]

        best_top_3_coverage = max(top_coverages[:3]) if top_coverages[:3] else 0.0
        best_top_5_coverage = max(top_coverages[:5]) if top_coverages[:5] else 0.0

        has_recipe_above_50 = int(any(score >= 0.5 for score in top_coverages))
        has_recipe_above_70 = int(any(score >= 0.7 for score in top_coverages))

        per_image_rows.append({
            "image_id": image_id,
            "detected_ingredient_count": len(detected_ingredients),
            "detected_ingredients": "; ".join(detected_ingredients),
            "recommendation_count": len(recommendations),
            "top1_title": top_recipe.get("title", ""),
            "top1_coverage": round(get_recipe_coverage(top_recipe), 4),
            "top1_matched_count": get_matched_count(top_recipe),
            "top1_missing_count": get_missing_count(top_recipe),
            "top1_missing_difficulty": top_recipe.get("missing_difficulty", ""),
            "top1_cookability_score": round(float(top_recipe.get("cookability_score", 0.0)), 4),
            "top1_prep_time": top_recipe.get("prep_time", ""),
            "best_top3_coverage": round(best_top_3_coverage, 4),
            "best_top5_coverage": round(best_top_5_coverage, 4),
            "mean_top5_coverage": round(safe_mean(top_coverages), 4),
            "mean_top5_missing_count": round(safe_mean(top_missing_counts), 4),
            "mean_top5_matched_count": round(safe_mean(top_matched_counts), 4),
            "has_recipe_above_50": has_recipe_above_50,
            "has_recipe_above_70": has_recipe_above_70,
        })

    df = pd.DataFrame(per_image_rows)

    summary = {
        "images_evaluated": len(df),
        "mean_detected_ingredients": round(float(df["detected_ingredient_count"].mean()), 4),
        "median_detected_ingredients": round(float(df["detected_ingredient_count"].median()), 4),
        "mean_top1_coverage": round(float(df["top1_coverage"].mean()), 4),
        "median_top1_coverage": round(float(df["top1_coverage"].median()), 4),
        "mean_best_top3_coverage": round(float(df["best_top3_coverage"].mean()), 4),
        "mean_best_top5_coverage": round(float(df["best_top5_coverage"].mean()), 4),
        "mean_top1_missing_count": round(float(df["top1_missing_count"].mean()), 4),
        "median_top1_missing_count": round(float(df["top1_missing_count"].median()), 4),
        "images_with_recipe_above_50": int(df["has_recipe_above_50"].sum()),
        "images_with_recipe_above_70": int(df["has_recipe_above_70"].sum()),
        "share_with_recipe_above_50": round(float(df["has_recipe_above_50"].mean()), 4),
        "share_with_recipe_above_70": round(float(df["has_recipe_above_70"].mean()), 4),
    }

    difficulty_counts = Counter(df["top1_missing_difficulty"].fillna("unknown"))

    for difficulty in ["easy", "medium", "hard", "unknown"]:
        summary[f"top1_{difficulty}_count"] = int(difficulty_counts.get(difficulty, 0))

    return df, summary


def plot_coverage_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["top1_coverage"], bins=15, edgecolor="white", linewidth=0.6)
    plt.xlabel("Top-1 recipe coverage")
    plt.ylabel("Number of images")
    plt.title("Top-1 recipe coverage distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "recipe_top1_coverage_distribution.png", dpi=300)
    plt.close()


def plot_topk_coverage(summary: dict) -> None:
    labels = ["Top-1", "Best Top-3", "Best Top-5"]
    values = [
        summary["mean_top1_coverage"],
        summary["mean_best_top3_coverage"],
        summary["mean_best_top5_coverage"],
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Mean coverage")
    plt.title("Recipe recommendation coverage by rank")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "recipe_topk_coverage.png", dpi=300)
    plt.close()


def plot_threshold_success(summary: dict) -> None:
    labels = [">= 50% match", ">= 70% match"]
    values = [
        summary["share_with_recipe_above_50"],
        summary["share_with_recipe_above_70"],
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Share of images")
    plt.title("Images with a usable recipe match")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "recipe_match_thresholds.png", dpi=300)
    plt.close()


def plot_difficulty_distribution(summary: dict) -> None:
    labels = ["Easy", "Medium", "Hard"]
    values = [
        summary["top1_easy_count"],
        summary["top1_medium_count"],
        summary["top1_hard_count"],
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel("Number of images")
    plt.title("Top recipe missing-ingredient difficulty")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "recipe_top1_difficulty_distribution.png", dpi=300)
    plt.close()


def write_summary(summary: dict) -> None:
    markdown = f"""# Recipe Recommendation Evaluation

## Inputs

- VLM predictions: `{PREDICTIONS_PATH}`
- Normalization: `{NORMALIZATION_PATH}`
- Recipe database: `{RECIPES_PATH}`

The evaluation uses high-confidence VLM ingredients because this matches the React + FastAPI demo app operating point.

## Summary Metrics

| Metric | Value |
|---|---:|
| Images evaluated | {summary["images_evaluated"]} |
| Mean detected ingredients per image | {summary["mean_detected_ingredients"]:.2f} |
| Median detected ingredients per image | {summary["median_detected_ingredients"]:.2f} |
| Mean top-1 recipe coverage | {summary["mean_top1_coverage"]:.4f} |
| Median top-1 recipe coverage | {summary["median_top1_coverage"]:.4f} |
| Mean best top-3 recipe coverage | {summary["mean_best_top3_coverage"]:.4f} |
| Mean best top-5 recipe coverage | {summary["mean_best_top5_coverage"]:.4f} |
| Mean missing ingredients in top recipe | {summary["mean_top1_missing_count"]:.2f} |
| Median missing ingredients in top recipe | {summary["median_top1_missing_count"]:.2f} |
| Images with at least one recipe >= 50% match | {summary["images_with_recipe_above_50"]} ({summary["share_with_recipe_above_50"] * 100:.0f}%) |
| Images with at least one recipe >= 70% match | {summary["images_with_recipe_above_70"]} ({summary["share_with_recipe_above_70"] * 100:.0f}%) |

## Top Recipe Difficulty

| Difficulty | Images |
|---|---:|
| Easy | {summary["top1_easy_count"]} |
| Medium | {summary["top1_medium_count"]} |
| Hard | {summary["top1_hard_count"]} |

## Interpretation

This evaluation measures whether the detected fridge ingredients can retrieve feasible recipes from the local recipe database. It does not measure taste preference, user satisfaction, or nutritional quality.

A higher top-k coverage means that the recommender is finding recipes whose ingredients overlap well with what the VLM detected. Missing-ingredient difficulty indicates whether the remaining ingredients are basic staples, common groceries, or more specific items.

## Generated Files

| File | Description |
|---|---|
| `recipe_recommendation_per_image.csv` | Per-image recommendation metrics |
| `figures/recipe_top1_coverage_distribution.png` | Top-1 coverage distribution |
| `figures/recipe_topk_coverage.png` | Mean top-k coverage comparison |
| `figures/recipe_match_thresholds.png` | Share of images with usable recipe matches |
| `figures/recipe_top1_difficulty_distribution.png` | Difficulty distribution for top recipe |
"""

    SUMMARY_OUTPUT.write_text(markdown, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    df, summary = evaluate_predictions()
    df.to_csv(PER_IMAGE_OUTPUT, index=False)

    plot_coverage_distribution(df)
    plot_topk_coverage(summary)
    plot_threshold_success(summary)
    plot_difficulty_distribution(summary)

    write_summary(summary)

    print("Recipe recommendation evaluation complete.")
    print(f"Per-image results: {PER_IMAGE_OUTPUT}")
    print(f"Summary report: {SUMMARY_OUTPUT}")
    print()
    print("Key metrics:")
    print(f"Images evaluated: {summary['images_evaluated']}")
    print(f"Mean top-1 coverage: {summary['mean_top1_coverage']:.4f}")
    print(f"Mean best top-3 coverage: {summary['mean_best_top3_coverage']:.4f}")
    print(f"Images with >=50% recipe match: {summary['images_with_recipe_above_50']}")
    print(f"Images with >=70% recipe match: {summary['images_with_recipe_above_70']}")


if __name__ == "__main__":
    main()