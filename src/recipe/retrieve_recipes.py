import json
import re
from pathlib import Path


RECIPES_PATH = Path(__file__).parent.parent.parent / "data" / "recipes.json"

TOP_N = 5


def load_recipes(path: Path = RECIPES_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _singularize(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def ingredient_in_text(ingredient: str, text: str) -> bool:
    """
    Whole-word match: true if every token of ``ingredient`` appears as a
    contiguous run inside ``text``'s tokens (after naive singularization).

    Recipe ingredient lines are raw descriptive phrases (e.g. "medium
    tomato", "crumbled feta cheese") rather than canonical names, so exact
    string equality against a normalized fridge ingredient like "tomato"
    almost never matches. Token containment catches these cases while
    avoiding false positives like "egg" inside "eggplant" or "milk" inside
    "buttermilk", since those don't tokenize into a matching word.
    """
    ingredient_tokens = [_singularize(t) for t in _tokenize(ingredient)]
    text_tokens = [_singularize(t) for t in _tokenize(text)]

    n = len(ingredient_tokens)

    if n == 0:
        return False

    return any(
        text_tokens[i:i + n] == ingredient_tokens
        for i in range(len(text_tokens) - n + 1)
    )


def score_recipe(recipe: dict, available_ingredients: set[str], idx: int) -> dict:
    """
    Score a recipe against available ingredients.

    Coverage score  = matched / total recipe ingredients  (primary rank key)
    Missing list    = recipe ingredients not in the fridge
    Matched list    = recipe ingredients confirmed in the fridge
    """
    recipe_ingredients = set(recipe["ingredients"])
    matched = {
        ing for ing in recipe_ingredients
        if any(ingredient_in_text(available, ing) for available in available_ingredients)
    }
    missing = recipe_ingredients - matched

    coverage = len(matched) / len(recipe_ingredients) if recipe_ingredients else 0.0

    return {
        "id":               recipe.get("id", f"r{idx:04d}"),
        "title":            recipe["title"],
        "cuisine":          recipe["cuisine"],
        "meal_type":        recipe["meal_type"],
        "prep_time":        recipe["prep_time"],
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

    scored = [score_recipe(r, ingredient_set, idx) for idx, r in enumerate(recipes)]

    scored.sort(key=lambda r: (
        -r["coverage"],
         r["missing_count"],
         r["prep_time"],
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
        lines.append(f"   Prep time   : {r['prep_time']} min")
        lines.append(f"   Instructions:")
        for step_num, step in enumerate(r["instructions"], start=1):
            lines.append(f"     {step_num}. {step}")
        lines.append("")
    return "\n".join(lines)


def main():
    # Example: run with a sample ingredient list for quick testing
    sample_ingredients = [
        "egg", "milk", "butter", "tomato", "cheese", "carrot"
    ]

    print("Available ingredients:", ", ".join(sample_ingredients))
    print()

    results = retrieve_recipes(sample_ingredients)
    print(f"Top {len(results)} recipe recommendations:\n")
    print(format_results(results))


if __name__ == "__main__":
    main()