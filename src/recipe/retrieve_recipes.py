import json
from pathlib import Path


RECIPES_PATH = Path(__file__).parent.parent.parent / "data" / "recipes.json"

TOP_N = 5


def load_recipes(path: Path = RECIPES_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_recipe(recipe: dict, available_ingredients: set[str]) -> dict:
    """
    Score a recipe against available ingredients.

    Coverage score  = matched / total recipe ingredients  (primary rank key)
    Missing list    = recipe ingredients not in the fridge
    Matched list    = recipe ingredients confirmed in the fridge
    """
    recipe_ingredients = set(recipe["ingredients"])
    matched  = recipe_ingredients & available_ingredients
    missing  = recipe_ingredients - available_ingredients

    coverage = len(matched) / len(recipe_ingredients) if recipe_ingredients else 0.0

    return {
        "id":               recipe["id"],
        "title":            recipe["title"],
        "cuisine":          recipe["cuisine"],
        "meal_type":        recipe["meal_type"],
        "prep_minutes":     recipe["prep_minutes"],
        "instructions":     recipe["instructions"],
        "coverage":         round(coverage, 4),
        "matched_count":    len(matched),
        "missing_count":    len(missing),
        "total_ingredients":len(recipe_ingredients),
        "matched":          sorted(matched),
        "missing":          sorted(missing),
    }


def retrieve_recipes(
    available_ingredients: list[str],
    top_n: int = TOP_N,
    recipes: list[dict] | None = None,
) -> list[dict]:
    """
    Given a list of available ingredients, return the top N recipes
    ranked by ingredient coverage (descending).
    Ties are broken by number of missing ingredients (ascending),
    then by prep time (ascending).
    """
    if recipes is None:
        recipes = load_recipes()

    ingredient_set = {i.strip().lower() for i in available_ingredients if i.strip()}

    scored = [score_recipe(r, ingredient_set) for r in recipes]

    scored.sort(key=lambda r: (
        -r["coverage"],
         r["missing_count"],
         r["prep_minutes"],
    ))

    return scored[:top_n]


def format_results(results: list[dict]) -> str:
    lines = []
    for rank, r in enumerate(results, start=1):
        coverage_pct = f"{r['coverage'] * 100:.0f}%"
        lines.append(f"{rank}. {r['title']} ({r['cuisine']}, {r['meal_type']})")
        lines.append(f"   Coverage    : {coverage_pct} ({r['matched_count']}/{r['total_ingredients']} ingredients)")
        lines.append(f"   Have        : {', '.join(r['matched']) or 'none'}")
        lines.append(f"   Missing     : {', '.join(r['missing']) or 'none'}")
        lines.append(f"   Prep time   : {r['prep_minutes']} min")
        lines.append(f"   Instructions: {r['instructions']}")
        lines.append("")
    return "\n".join(lines)


def main():
    # Example: run with a sample ingredient list for quick testing
    sample_ingredients = [
        "egg", "butter", "cheese", "milk", "tomato", "lettuce", "mustard"
    ]

    print("Available ingredients:", ", ".join(sample_ingredients))
    print()

    results = retrieve_recipes(sample_ingredients)
    print(f"Top {len(results)} recipe recommendations:\n")
    print(format_results(results))


if __name__ == "__main__":
    main()