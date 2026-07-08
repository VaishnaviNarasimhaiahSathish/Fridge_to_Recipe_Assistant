import json
import math
import re
from collections import Counter
from pathlib import Path


RECIPES_PATH = Path(__file__).parent.parent.parent / "data" / "recipes.json"
TOP_N = 5


def load_recipes(path: Path = RECIPES_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase word tokens.

    Example:
    "medium tomatoes and feta cheese" ->
    ["medium", "tomatoes", "and", "feta", "cheese"]
    """
    return re.findall(r"[a-z]+", text.lower())


def simple_normalize_token(token: str) -> str:
    """
    Light normalization for plural-like forms.

    This is intentionally simple and explainable.
    It helps with common recipe terms such as:
    tomatoes -> tomato
    carrots -> carrot
    eggs -> egg

    It is not a full lemmatizer. That remains future work.
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
    return [simple_normalize_token(t) for t in tokens]


def recipe_to_search_text(recipe: dict) -> str:
    """
    Build the searchable document text for one recipe.

    We focus mainly on ingredients because the user query comes from
    detected fridge ingredients.
    """
    title = recipe.get("title", "")
    ingredients = " ".join(recipe.get("ingredients", []))

    return f"{title} {ingredients}"


def build_bm25_index(recipes: list[dict]) -> dict:
    """
    Build a lightweight BM25-style index.

    BM25 scores recipes higher when query ingredients appear in the recipe text.
    It also reduces the importance of terms that appear in many recipes.
    """
    documents = []

    for idx, recipe in enumerate(recipes):
        text = recipe_to_search_text(recipe)
        tokens = normalize_tokens(tokenize(text))
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
    Compute BM25 score for one recipe document.

    Higher score means the recipe ingredient text is more relevant
    to the available fridge ingredients.
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

        # Standard BM25-style inverse document frequency.
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))

        denominator = term_frequency + k1 * (1 - b + b * doc_len / avg_doc_len)

        score += idf * ((term_frequency * (k1 + 1)) / denominator)

    return score


def explain_matches(query_tokens: list[str], recipe: dict) -> list[str]:
    """
    Return which query ingredients/tokens appear in this recipe.
    This is useful for debugging and presentation case studies.
    """
    recipe_tokens = set(normalize_tokens(tokenize(recipe_to_search_text(recipe))))

    matched = sorted({
        token for token in query_tokens
        if token in recipe_tokens
    })

    return matched


def search_recipes_bm25(
    available_ingredients: list[str],
    top_n: int = TOP_N,
    recipes: list[dict] | None = None,
) -> list[dict]:
    """
    Search recipes using BM25-style ingredient search.

    Input:
    available_ingredients = ingredients detected from the fridge image

    Output:
    ranked recipes with BM25 score and matched query tokens
    """
    if recipes is None:
        recipes = load_recipes()

    index = build_bm25_index(recipes)

    query_text = " ".join(available_ingredients)
    query_tokens = normalize_tokens(tokenize(query_text))

    # Remove duplicates but keep stable order.
    query_tokens = list(dict.fromkeys(query_tokens))

    results = []

    for doc in index["documents"]:
        recipe = doc["recipe"]
        score = bm25_score(query_tokens, doc, index)

        matched_query_terms = explain_matches(query_tokens, recipe)

        results.append({
            "title": recipe.get("title", "unknown"),
            "cuisine": recipe.get("cuisine", "unknown"),
            "meal_type": recipe.get("meal_type", "unknown"),
            "prep_time": recipe.get("prep_time", None),
            "bm25_score": round(score, 4),
            "matched_query_terms": matched_query_terms,
            "matched_query_count": len(matched_query_terms),
            "ingredients": recipe.get("ingredients", []),
            "instructions": recipe.get("instructions", []),
        })

    results.sort(key=lambda r: (
        -r["bm25_score"],
        -r["matched_query_count"],
        r["prep_time"] if r["prep_time"] is not None else 9999,
    ))

    return results[:top_n]


def format_results(results: list[dict]) -> str:
    lines = []

    for rank, recipe in enumerate(results, start=1):
        lines.append(f"{rank}. {recipe['title']}")
        lines.append(f"   BM25 score          : {recipe['bm25_score']}")
        lines.append(f"   Matched query terms : {', '.join(recipe['matched_query_terms']) or 'none'}")
        lines.append(f"   Prep time           : {recipe['prep_time']} min")
        lines.append(f"   Ingredients         : {', '.join(recipe['ingredients'][:8])}")
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

    results = search_recipes_bm25(sample_ingredients, top_n=5)

    print("Top BM25-style recipe search results:\n")
    print(format_results(results))


if __name__ == "__main__":
    main()