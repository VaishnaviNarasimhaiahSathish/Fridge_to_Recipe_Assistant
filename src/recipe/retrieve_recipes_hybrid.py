import json
import math
import re
from collections import Counter
from pathlib import Path


RECIPES_PATH = Path(__file__).parent.parent.parent / "data" / "recipes.json"

TOP_N = 5
CANDIDATE_LIMIT = 50


PASSAU_GROCERY_STORES = [
    "REWE",
    "EDEKA",
    "Netto",
    "Penny",
    "Kaufland",
]


PRICE_ESTIMATES_EUR = {
    "egg": "€2–4",
    "milk": "€1–2",
    "butter": "€2–4",
    "cheese": "€2–5",
    "yogurt": "€1–3",
    "tomato": "€1–3",
    "cucumber": "€1–2",
    "carrot": "€1–2",
    "onion": "€1–2",
    "garlic": "€1–2",
    "potato": "€1–3",
    "chicken": "€4–8",
    "salmon": "€5–10",
    "lettuce": "€1–3",
    "pepper": "€1–3",
    "lemon": "€1–2",
    "apple": "€1–3",
    "bread": "€1–3",
    "pasta": "€1–3",
    "rice": "€1–3",
}


def load_recipes(path: Path = RECIPES_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase word tokens.

    Example:
    "medium tomatoes and feta cheese"
    becomes:
    ["medium", "tomatoes", "and", "feta", "cheese"]
    """
    return re.findall(r"[a-z]+", text.lower())


def simple_normalize_token(token: str) -> str:
    """
    Light normalization for common plural forms.

    This is intentionally simple and explainable.
    It helps with:
    eggs -> egg
    carrots -> carrot
    tomatoes -> tomato

    It is not a full lemmatizer. A better lemmatizer or synonym table
    remains future work.
    """
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    if token.endswith("oes") and len(token) > 4:
        return token[:-2]

    if token.endswith("es") and len(token) > 4:
        return token[:-2]

    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]

    return token


def normalize_tokens(tokens: list[str]) -> list[str]:
    return [simple_normalize_token(token) for token in tokens]


def normalize_text(text: str) -> list[str]:
    return normalize_tokens(tokenize(text))


def recipe_to_search_text(recipe: dict) -> str:
    """
    Build searchable recipe text.

    We mainly use title + ingredients because the query comes from
    detected fridge ingredients.
    """
    title = recipe.get("title", "")
    ingredients = " ".join(recipe.get("ingredients", []))
    return f"{title} {ingredients}"


def ingredient_in_text(ingredient: str, text: str) -> bool:
    """
    Check whether a fridge ingredient appears inside a recipe ingredient phrase.

    This uses whole-token matching, so:
    egg matches "large eggs"
    tomato matches "medium tomato"

    But:
    egg does not match "eggplant"
    milk does not match "buttermilk"
    """
    ingredient_tokens = normalize_text(ingredient)
    text_tokens = normalize_text(text)

    n = len(ingredient_tokens)

    if n == 0:
        return False

    return any(
        text_tokens[i:i + n] == ingredient_tokens
        for i in range(len(text_tokens) - n + 1)
    )


def build_bm25_index(recipes: list[dict]) -> dict:
    """
    Build a lightweight local BM25-style search index.

    This is a small local alternative to a full Elasticsearch setup.
    It is useful for finding candidate recipes that are textually related
    to the detected fridge ingredients.
    """
    documents = []

    for idx, recipe in enumerate(recipes):
        text = recipe_to_search_text(recipe)
        tokens = normalize_text(text)
        token_counts = Counter(tokens)

        documents.append({
            "idx": idx,
            "recipe": recipe,
            "tokens": tokens,
            "token_counts": token_counts,
            "length": len(tokens),
        })

    doc_count = len(documents)
    avg_doc_len = (
        sum(doc["length"] for doc in documents) / doc_count
        if doc_count else 0
    )

    document_frequency = Counter()

    for doc in documents:
        for token in set(doc["tokens"]):
            document_frequency[token] += 1

    return {
        "documents": documents,
        "doc_count": doc_count,
        "avg_doc_len": avg_doc_len,
        "document_frequency": document_frequency,
    }


def bm25_score(
    query_tokens: list[str],
    document: dict,
    index: dict,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    Compute a BM25-style relevance score.

    Higher score means the recipe text is more relevant to the detected
    fridge ingredients.
    """
    score = 0.0

    doc_count = index["doc_count"]
    avg_doc_len = index["avg_doc_len"]
    document_frequency = index["document_frequency"]

    if doc_count == 0 or avg_doc_len == 0:
        return 0.0

    doc_len = document["length"]
    token_counts = document["token_counts"]

    for token in query_tokens:
        term_frequency = token_counts.get(token, 0)

        if term_frequency == 0:
            continue

        df = document_frequency.get(token, 0)

        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))

        denominator = term_frequency + k1 * (
            1 - b + b * doc_len / avg_doc_len
        )

        score += idf * (
            term_frequency * (k1 + 1)
        ) / denominator

    return score


def get_bm25_candidates(
    available_ingredients: list[str],
    recipes: list[dict],
    candidate_limit: int = CANDIDATE_LIMIT,
) -> list[dict]:
    """
    Use BM25-style search to get candidate recipes.

    This step answers:
    Which recipes are textually related to the detected ingredients?
    """
    index = build_bm25_index(recipes)

    query_text = " ".join(available_ingredients)
    query_tokens = normalize_text(query_text)

    # Remove duplicate query tokens while keeping order.
    query_tokens = list(dict.fromkeys(query_tokens))

    candidates = []

    for doc in index["documents"]:
        recipe = doc["recipe"]
        score = bm25_score(query_tokens, doc, index)

        recipe_tokens = set(normalize_text(recipe_to_search_text(recipe)))
        matched_query_terms = sorted({
            token for token in query_tokens
            if token in recipe_tokens
        })

        candidates.append({
            "recipe": recipe,
            "bm25_score": round(score, 4),
            "matched_query_terms": matched_query_terms,
            "matched_query_count": len(matched_query_terms),
        })

    candidates.sort(key=lambda c: (
        -c["bm25_score"],
        -c["matched_query_count"],
        c["recipe"].get("prep_time", 9999),
    ))

    return candidates[:candidate_limit]


def score_recipe_by_coverage(
    recipe: dict,
    available_ingredients: set[str],
    bm25_score_value: float,
    matched_query_terms: list[str],
    idx: int,
) -> dict:
    """
    Final recipe scoring.

    Coverage is the main score because the user wants recipes based on
    ingredients already present in the fridge.

    BM25 is used as a supporting score for candidate search and tie-breaking.
    """
    recipe_ingredients = set(recipe.get("ingredients", []))

    matched = {
        recipe_ingredient
        for recipe_ingredient in recipe_ingredients
        if any(
            ingredient_in_text(available, recipe_ingredient)
            for available in available_ingredients
        )
    }

    missing = recipe_ingredients - matched

    total_ingredients = len(recipe_ingredients)
    coverage = (
        len(matched) / total_ingredients
        if total_ingredients else 0.0
    )

    return {
        "id": recipe.get("id", f"r{idx:04d}"),
        "title": recipe.get("title", "unknown"),
        "cuisine": recipe.get("cuisine", "unknown"),
        "meal_type": recipe.get("meal_type", "unknown"),
        "prep_time": recipe.get("prep_time", None),
        "instructions": recipe.get("instructions", []),
        "ingredients": sorted(recipe_ingredients),
        "coverage": round(coverage, 4),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "total_ingredients": total_ingredients,
        "matched": sorted(matched),
        "missing": sorted(missing),
        "bm25_score": bm25_score_value,
        "matched_query_terms": matched_query_terms,
    }


def estimate_missing_item_price(item: str) -> str:
    """
    Return a rough static price estimate.

    This is not live pricing. It is only a demo-friendly estimate.
    """
    item_tokens = normalize_text(item)

    for key, price_range in PRICE_ESTIMATES_EUR.items():
        key_tokens = normalize_text(key)

        if any(token in item_tokens for token in key_tokens):
            return price_range

    return "price varies"


def suggest_grocery_options(missing_items: list[str], max_items: int = 3) -> list[dict]:
    """
    Suggest simple grocery options for the first few missing ingredients.

    This is a static Passau-oriented suggestion, not live store lookup.
    """
    suggestions = []

    for item in missing_items[:max_items]:
        suggestions.append({
            "missing_item": item,
            "estimated_price": estimate_missing_item_price(item),
            "suggested_stores": PASSAU_GROCERY_STORES,
            "note": "Static demo suggestion for Passau; not live price or inventory data.",
        })

    return suggestions


def retrieve_recipes_hybrid(
    available_ingredients: list[str],
    top_n: int = TOP_N,
    candidate_limit: int = CANDIDATE_LIMIT,
    recipes: list[dict] | None = None,
) -> list[dict]:
    """
    Hybrid recipe retrieval.

    Step 1:
    BM25-style search finds relevant candidate recipes.

    Step 2:
    Coverage-based ranking decides which recipes are most cookable
    with the detected fridge ingredients.

    Final ranking priority:
    1. Higher ingredient coverage
    2. Fewer missing ingredients
    3. Higher BM25 relevance
    4. Shorter prep time
    """
    if recipes is None:
        recipes = load_recipes()

    available_set = {
        ingredient.strip().lower()
        for ingredient in available_ingredients
        if ingredient.strip()
    }

    candidates = get_bm25_candidates(
        available_ingredients=available_ingredients,
        recipes=recipes,
        candidate_limit=candidate_limit,
    )

    scored_results = []

    for idx, candidate in enumerate(candidates):
        recipe = candidate["recipe"]

        scored = score_recipe_by_coverage(
            recipe=recipe,
            available_ingredients=available_set,
            bm25_score_value=candidate["bm25_score"],
            matched_query_terms=candidate["matched_query_terms"],
            idx=idx,
        )

        scored["grocery_suggestions"] = suggest_grocery_options(
            scored["missing"],
            max_items=3,
        )

        scored_results.append(scored)

    scored_results.sort(key=lambda result: (
        -result["coverage"],
        result["missing_count"],
        -result["bm25_score"],
        result["prep_time"] if result["prep_time"] is not None else 9999,
    ))

    return scored_results[:top_n]


def format_results(results: list[dict]) -> str:
    lines = []

    for rank, recipe in enumerate(results, start=1):
        coverage_pct = f"{recipe['coverage'] * 100:.0f}%"

        lines.append(f"{rank}. {recipe['title']}")
        lines.append(
            f"   Coverage       : {coverage_pct} "
            f"({recipe['matched_count']}/{recipe['total_ingredients']} ingredients)"
        )
        lines.append(f"   BM25 score     : {recipe['bm25_score']}")
        lines.append(f"   Prep time      : {recipe['prep_time']} min")
        lines.append(
            f"   Search matched : "
            f"{', '.join(recipe['matched_query_terms']) or 'none'}"
        )
        lines.append(f"   You have       : {', '.join(recipe['matched']) or 'none'}")
        lines.append(f"   You need       : {', '.join(recipe['missing']) or 'none'}")

        if recipe["grocery_suggestions"]:
            lines.append("   Grocery suggestions for missing items:")
            for suggestion in recipe["grocery_suggestions"]:
                stores = ", ".join(suggestion["suggested_stores"])
                lines.append(
                    f"     - {suggestion['missing_item']} "
                    f"({suggestion['estimated_price']}): {stores}"
                )

        lines.append("")

    return "\n".join(lines)


def main():
    sample_ingredients = [
        "egg",
        "milk",
        "butter",
        "tomato",
        "cheese",
        "carrot",
    ]

    print("Available fridge ingredients:")
    print(", ".join(sample_ingredients))
    print()

    results = retrieve_recipes_hybrid(
        available_ingredients=sample_ingredients,
        top_n=5,
    )

    print("Top hybrid recipe recommendations:\n")
    print(format_results(results))


if __name__ == "__main__":
    main()
