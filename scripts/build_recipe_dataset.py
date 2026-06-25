"""
Build data/recipes.json (500+ recipes) from the RecipeNLG Lite dataset on Hugging Face,
filtered to recipes that overlap with common fridge ingredients seen in VLM predictions.
"""

import json
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

# The dataset repo only ships a legacy loading script, which modern `datasets`
# versions no longer execute. Load from its auto-converted parquet branch instead.
PARQUET_DATA_FILES = {
    "train": "https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite/resolve/refs%2Fconvert%2Fparquet/1.0.0/recipe_nlg_lite-train.parquet",
    "test": "https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite/resolve/refs%2Fconvert%2Fparquet/1.0.0/recipe_nlg_lite-test.parquet",
}

DATA_DIR = Path(__file__).parent.parent / "data"
RECIPES_PATH = DATA_DIR / "recipes.json"
MANUAL_BACKUP_PATH = DATA_DIR / "recipes_manual_40.json"

MIN_COMMON_MATCHES = 3
TOP_N = 500

COMMON_FRIDGE_ITEMS = [
    "egg", "milk", "butter", "cheese", "tomato", "carrot", "apple",
    "lemon", "cucumber", "yogurt", "chicken", "onion", "garlic", "pepper",
]


def parse_ner_field(value) -> list[str]:
    """ner is a comma-separated string of ingredient names, e.g. "salt, pepper, carrots"."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def parse_steps_field(value) -> list[str]:
    """steps is a " . "-separated string of instruction sentences."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(" . ") if part.strip()]
    return []


def normalize_ingredients(raw_ner: list[str]) -> list[str]:
    seen = []
    for item in raw_ner:
        name = item.strip().lower()
        if name and name not in seen:
            seen.append(name)
    return seen


def count_common_matches(ingredients: list[str]) -> int:
    matched = set()
    for common in COMMON_FRIDGE_ITEMS:
        for ing in ingredients:
            if common in ing or ing in common:
                matched.add(common)
                break
    return len(matched)


def estimate_prep_time(steps: list[str]) -> int:
    return max(10, min(90, len(steps) * 5))


def build_recipe_record(row: dict) -> dict | None:
    ingredients = normalize_ingredients(parse_ner_field(row.get("ner")))
    steps = parse_steps_field(row.get("steps"))
    title = str(row.get("name", "")).strip()

    if not ingredients or not steps or not title:
        return None

    return {
        "title": title,
        "ingredients": ingredients,
        "instructions": steps,
        "cuisine": "unknown",
        "meal_type": "unknown",
        "prep_time": estimate_prep_time(steps),
        "_match_count": count_common_matches(ingredients),
    }


def main():
    dataset = load_dataset("parquet", data_files=PARQUET_DATA_FILES)
    combined = concatenate_datasets([dataset["train"], dataset["test"]])
    print(f"Loaded {len(combined)} recipes from m3hrdadfi/recipe_nlg_lite (train+test)")

    candidates = []
    for row in combined:
        record = build_recipe_record(row)
        if record is not None and record["_match_count"] >= MIN_COMMON_MATCHES:
            candidates.append(record)

    print(f"{len(candidates)} recipes matched at least {MIN_COMMON_MATCHES} common fridge ingredients")

    candidates.sort(key=lambda r: (-r["_match_count"], len(r["ingredients"])))
    top_recipes = candidates[:TOP_N]
    for r in top_recipes:
        del r["_match_count"]

    if not MANUAL_BACKUP_PATH.exists() and RECIPES_PATH.exists():
        with open(RECIPES_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        with open(MANUAL_BACKUP_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        print(f"Backed up existing {len(existing)} manual recipes to {MANUAL_BACKUP_PATH}")

    with open(RECIPES_PATH, "w", encoding="utf-8") as f:
        json.dump(top_recipes, f, indent=2)

    print(f"Saved {len(top_recipes)} recipes to {RECIPES_PATH}")


if __name__ == "__main__":
    main()
