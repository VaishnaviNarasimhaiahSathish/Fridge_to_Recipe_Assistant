import json
import math
import re
from collections import Counter
from pathlib import Path


RECIPES_PATH = Path(__file__).parent.parent.parent / "data" / "recipes.json"

TOP_N = 5
CANDIDATE_LIMIT = 50


STORE_RECOMMENDATIONS = {
    "basic_staple": {
        "recommended_store_type": "Budget or regular supermarket",
        "suggested_stores": ["ALDI", "Lidl", "Penny", "Netto", "REWE", "EDEKA"],
        "shopping_note": "Usually easy to find in most supermarkets or discount stores.",
        "priority": "low",
    },
    "common_grocery": {
        "recommended_store_type": "Regular supermarket",
        "suggested_stores": ["REWE", "EDEKA", "Kaufland", "Lidl", "ALDI"],
        "shopping_note": "Common grocery item; most supermarkets should carry it.",
        "priority": "medium",
    },
    "specific_grocery": {
        "recommended_store_type": "Regular or wider-selection supermarket",
        "suggested_stores": ["REWE", "EDEKA", "Kaufland"],
        "shopping_note": "Specific packaged or recipe item; a larger supermarket is safer.",
        "priority": "medium",
    },
    "special_ingredient": {
        "recommended_store_type": "Wider-selection supermarket",
        "suggested_stores": ["REWE", "EDEKA", "Kaufland"],
        "shopping_note": "More specific ingredient; check a wider-selection store first.",
        "priority": "high",
    },
}


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

NOISY_INGREDIENT_PHRASES = {
    "or a combination",
    "to taste",
    "as needed",
    "optional",
}


INGREDIENT_DISPLAY_REPLACEMENTS = {
    "tyson premium selects breaded chicken nuggets": "chicken nuggets",
    "vermont bread company organic multigrain bread": "multigrain bread",
    "nellie's cage free eggs": "eggs",
    "stonyfield farm organic whole milk plain yogurt": "plain yogurt",
    "ragú robusto chopped tomato": "tomato sauce",
    "ragu robusto chopped tomato": "tomato sauce",
    "prepared garlic tomato sauce": "tomato sauce",
    "light mayonnaise": "mayonnaise",
    "fat free greek yogurt": "greek yogurt",
    "crumbled blue cheese": "blue cheese",
    "shredded mozzarella cheese": "mozzarella cheese",
    "shredded parmesan cheese": "parmesan cheese",
    "grated parmesan cheese": "parmesan cheese",
    "reduced fat swiss cheese": "swiss cheese",
    "large egg whites": "egg whites",
    "large eggs": "eggs",
    "whole milk": "milk",
    "milk or water": "milk",
    "whole milk or water": "milk",
    "or butter": "butter",
    "salt and pepper to taste": "salt and pepper",
    "salt and fresh pepper to taste": "salt and pepper",
    "salt and freshly ground black pepper": "salt and black pepper",
    "freshly ground black pepper": "black pepper",
    "ground black pepper": "black pepper",
    "fresh grated pecorino romano": "pecorino romano",
    "fresh parsley leaves": "parsley",
    "jumbo shells pasta": "pasta shells",
    "olive oil and garlic pasta sauce": "garlic pasta sauce",
}

BASIC_STAPLES = {
    "salt",
    "pepper",
    "black pepper",
    "oil",
    "olive oil",
    "vegetable oil",
    "sugar",
    "flour",
    "water",
    "butter",
}


COMMON_GROCERY = {
    "egg",
    "milk",
    "cheese",
    "yogurt",
    "tomato",
    "cucumber",
    "carrot",
    "onion",
    "garlic",
    "potato",
    "lettuce",
    "spinach",
    "bell pepper",
    "lemon",
    "lime",
    "apple",
    "banana",
    "orange",
    "bread",
    "pasta",
    "rice",
    "mushroom",
    "chicken",
    "pickle",
    "mustard",
    "mayonnaise",
    "ketchup",
}


SPECIAL_INGREDIENTS = {
    "smoked salmon",
    "salmon",
    "goat cheese",
    "feta",
    "feta cheese",
    "blue cheese",
    "parmesan",
    "parmesan cheese",
    "asparagus",
    "capers",
    "fresh herbs",
    "basil",
    "parsley",
    "cilantro",
    "coriander",
    "mint",
    "pecorino",
    "prosciutto",
    "anchovy",
    "anchovies",
    "shrimp",
    "tuna",
    "avocado",
    "hummus",
    "sour cream",
    "cream cheese",
    "coconut milk",
    "almond milk",
    "oat milk",
    "soy milk",
    "arugula",
    "dill",
    "fresh dill",
    "shallot",
    "shallots",
}


SPECIFIC_GROCERY = {
    "pepperoni",
    "corn muffin mix",
    "muffin mix",
    "hash brown",
    "hash brown potato",
    "hash brown potatoes",
    "frozen hash brown",
    "frozen hash brown potatoes",
    "nonstick cooking spray",
    "cooking spray",
    "labaneh",
    "swiss cheese",
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


def normalized_phrase(text: str) -> str:
    """
    Normalize a phrase into a simple token-based phrase.

    Example:
    "large tomatoes" -> "large tomato"
    """
    return " ".join(normalize_text(text))

def clean_recipe_ingredient_for_display(ingredient: str) -> str:
    """
    Clean noisy recipe ingredient phrases for display, scoring, and grocery suggestions.

    This is not meant to fully rewrite the recipe dataset.
    It only removes common brand names, instruction fragments, and overly long
    ingredient phrases that make the demo harder to read.
    """
    cleaned = str(ingredient).strip().lower()

    if not cleaned:
        return ""

    cleaned = cleaned.replace("  ", " ")

    if cleaned in INGREDIENT_DISPLAY_REPLACEMENTS:
        return INGREDIENT_DISPLAY_REPLACEMENTS[cleaned]

    for noisy_phrase in NOISY_INGREDIENT_PHRASES:
        cleaned = cleaned.replace(noisy_phrase, "").strip()

    cleaned = cleaned.replace(" ,", ",")
    cleaned = cleaned.strip(" ,.-")

    if cleaned in INGREDIENT_DISPLAY_REPLACEMENTS:
        return INGREDIENT_DISPLAY_REPLACEMENTS[cleaned]

    return cleaned


def clean_recipe_ingredients_for_display(ingredients: list[str]) -> list[str]:
    """
    Clean a list of recipe ingredient phrases and remove empty/noisy results.
    Duplicates are removed while preserving sorted display stability later.
    """
    cleaned_items = []

    for ingredient in ingredients:
        cleaned = clean_recipe_ingredient_for_display(ingredient)

        if not cleaned:
            continue

        if cleaned not in cleaned_items:
            cleaned_items.append(cleaned)

    return cleaned_items


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


def phrase_matches_item(phrase: str, item: str) -> bool:
    """
    Check whether a category keyword appears in a missing recipe ingredient.

    This is used for difficulty scoring.

    Example:
    phrase "goat cheese" matches item "crumbled goat cheese"
    phrase "salt" matches item "kosher salt"
    """
    return ingredient_in_text(phrase, item)


def categorize_missing_item(item: str) -> str:
    """
    Categorize one missing ingredient by practical shopping difficulty.

    The categories are intentionally simple and explainable:
    - basic_staple: usually available at home or easy to buy cheaply
    - common_grocery: normal supermarket ingredient
    - specific_grocery: packaged/specific item that is not rare but not a basic staple
    - special_ingredient: more specific, fresh, or less commonly available item
    """
    item_clean = normalized_phrase(item)

    for special in SPECIAL_INGREDIENTS:
        if phrase_matches_item(special, item_clean):
            return "special_ingredient"

    for specific in SPECIFIC_GROCERY:
        if phrase_matches_item(specific, item_clean):
            return "specific_grocery"

    for staple in BASIC_STAPLES:
        if phrase_matches_item(staple, item_clean):
            return "basic_staple"

    for common in COMMON_GROCERY:
        if phrase_matches_item(common, item_clean):
            return "common_grocery"

    return "common_grocery"


def score_missing_difficulty(missing_items: list[str]) -> dict:
    """
    Summarize how difficult a recipe is to complete based on missing items.

    This does not claim real store availability.
    It only estimates practical difficulty using ingredient categories.
    """
    basic_staples = []
    common_grocery = []
    specific_grocery = []
    special_ingredients = []

    for item in missing_items:
        category = categorize_missing_item(item)

        if category == "basic_staple":
            basic_staples.append(item)
        elif category == "special_ingredient":
            special_ingredients.append(item)
        elif category == "specific_grocery":
            specific_grocery.append(item)
        else:
            common_grocery.append(item)

    missing_count = len(missing_items)
    special_count = len(special_ingredients)
    specific_count = len(specific_grocery)
    common_count = len(common_grocery)

    if missing_count == 0:
        difficulty = "easy"
        reason = "All listed recipe ingredients are matched."

    elif special_count > 0:
        difficulty = "hard"
        reason = "Some missing ingredients are more specific and may need a wider-selection store."

    elif specific_count > 0:
        difficulty = "medium"
        reason = "Some missing ingredients are specific grocery items, but should still be possible to find."

    elif missing_count <= 2 and common_count <= 2:
        difficulty = "easy"
        reason = "Only a small number of basic or common grocery items are missing."

    elif missing_count <= 5:
        difficulty = "medium"
        reason = "Several common grocery items are missing."

    else:
        difficulty = "hard"
        reason = "Many ingredients are missing, so this recipe may need a larger grocery trip."

    return {
        "missing_difficulty": difficulty,
        "missing_difficulty_reason": reason,
        "missing_difficulty_details": {
            "basic_staples": sorted(basic_staples),
            "common_grocery": sorted(common_grocery),
            "specific_grocery": sorted(specific_grocery),
            "special_ingredients": sorted(special_ingredients),
        },
    }


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


def get_difficulty_penalty(difficulty: str) -> int:
    """
    Convert missing-ingredient difficulty into a small ranking penalty.

    The values are intentionally simple:
    - easy recipes get no penalty
    - medium recipes get a small penalty
    - hard recipes get a stronger penalty

    This makes the final ranking prefer recipes that are easier to complete,
    without ignoring ingredient coverage.
    """
    difficulty = str(difficulty).strip().lower()

    if difficulty == "easy":
        return 0

    if difficulty == "medium":
        return 6

    if difficulty == "hard":
        return 14

    return 6


def get_difficulty_rank(difficulty: str) -> int:
    """
    Lower rank is better.

    Used as a stable tie-breaker after cookability score.
    """
    difficulty = str(difficulty).strip().lower()

    if difficulty == "easy":
        return 0

    if difficulty == "medium":
        return 1

    if difficulty == "hard":
        return 2

    return 1


def calculate_cookability_score(
    coverage: float,
    missing_count: int,
    difficulty: str,
) -> float:
    """
    Calculate an explainable final score for recipe ranking.

    The score rewards:
    - high ingredient coverage

    The score penalizes:
    - many missing ingredients
    - difficult/specific missing ingredients

    Example:
    A recipe with 60% coverage and easy missing items can rank above
    a recipe with similar coverage but hard-to-find missing items.
    """
    coverage_points = coverage * 100
    missing_penalty = missing_count * 2
    difficulty_penalty = get_difficulty_penalty(difficulty)

    return round(
        coverage_points - missing_penalty - difficulty_penalty,
        4,
    )


def score_recipe_by_coverage(
    recipe: dict,
    available_ingredients: set[str],
    bm25_score_value: float,
    matched_query_terms: list[str],
    idx: int,
) -> dict:
    """
    Final recipe scoring.

    Coverage measures how many recipe ingredients are already available.

    Missing ingredient difficulty estimates whether the remaining ingredients
    are basic, common, specific, or special.

    Cookability score combines both:
    - high coverage is good
    - fewer missing ingredients is good
    - easier missing ingredients are better
    """
    recipe_ingredients = set(
        clean_recipe_ingredients_for_display(recipe.get("ingredients", []))
)

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

    missing_sorted = sorted(missing)
    difficulty_info = score_missing_difficulty(missing_sorted)

    missing_difficulty = difficulty_info["missing_difficulty"]
    difficulty_penalty = get_difficulty_penalty(missing_difficulty)
    difficulty_rank = get_difficulty_rank(missing_difficulty)

    cookability_score = calculate_cookability_score(
        coverage=coverage,
        missing_count=len(missing),
        difficulty=missing_difficulty,
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
        "missing": missing_sorted,
        "bm25_score": bm25_score_value,
        "matched_query_terms": matched_query_terms,
        "missing_difficulty": missing_difficulty,
        "missing_difficulty_reason": difficulty_info["missing_difficulty_reason"],
        "missing_difficulty_details": difficulty_info["missing_difficulty_details"],
        "difficulty_penalty": difficulty_penalty,
        "difficulty_rank": difficulty_rank,
        "cookability_score": cookability_score,
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


def suggest_grocery_options(missing_items: list[str], max_items: int = 4) -> list[dict]:
    """
    Suggest grocery options for missing ingredients.

    Suggestions are static and demo-friendly, but now depend on ingredient type:
    - basic staples can be bought almost anywhere
    - common groceries are normal supermarket items
    - specific groceries are better searched in larger stores
    - special ingredients are best checked in wider-selection stores

    This is not live price, stock, or opening-hour data.
    """
    suggestions = []

    for item in missing_items[:max_items]:
        category = categorize_missing_item(item)
        store_info = STORE_RECOMMENDATIONS.get(
            category,
            STORE_RECOMMENDATIONS["common_grocery"],
        )

        suggestions.append({
            "missing_item": item,
            "ingredient_category": category,
            "estimated_price": estimate_missing_item_price(item),
            "recommended_store_type": store_info["recommended_store_type"],
            "suggested_stores": store_info["suggested_stores"],
            "shopping_note": store_info["shopping_note"],
            "priority": store_info["priority"],
            "note": "Static demo suggestion for Passau; not live price or inventory data.",
        })

    return suggestions


def retrieve_recipes_hybrid(
    available_ingredients: list[str],
    top_n: int = TOP_N,
    candidate_limit: int = CANDIDATE_LIMIT,
    recipes: list[dict] | None = None,
    ranking_mode: str = "all",
) -> list[dict]:
    """
    Hybrid recipe retrieval.

    Step 1:
    BM25-style search finds relevant candidate recipes.

    Step 2:
    Coverage-based scoring estimates how cookable each recipe is.

    Ranking modes:
    - all: prioritize fridge ingredient match and cookability
    - quick: keep only recipes up to 30 minutes, then rank by cookability
    - vegetarian: same ranking as all; vegetarian filtering is handled in backend
    """
    if recipes is None:
        recipes = load_recipes()

    ranking_mode = str(ranking_mode).strip().lower()

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
            max_items=4,
        )

        scored_results.append(scored)

    if ranking_mode == "quick":
        scored_results = [
            result for result in scored_results
            if result.get("prep_time") is not None and result.get("prep_time") <= 30
        ]

        scored_results.sort(key=lambda result: (
            -result["cookability_score"],
            -result["coverage"],
            result["difficulty_rank"],
            result["missing_count"],
            result["prep_time"] if result["prep_time"] is not None else 9999,
            -result["bm25_score"],
        ))

    else:
        scored_results.sort(key=lambda result: (
            -result["cookability_score"],
            -result["coverage"],
            result["difficulty_rank"],
            result["missing_count"],
            result["prep_time"] if result["prep_time"] is not None else 9999,
            -result["bm25_score"],
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
        lines.append(f"   Difficulty     : {recipe['missing_difficulty']}")
        lines.append(f"   Reason         : {recipe['missing_difficulty_reason']}")
        lines.append(f"   Cookability    : {recipe['cookability_score']}")
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
                    f"({suggestion['ingredient_category']}, "
                    f"{suggestion['estimated_price']}): {stores}"
                )
                lines.append(
                    f"       Store type: {suggestion['recommended_store_type']}"
                )
                lines.append(
                    f"       Note      : {suggestion['shopping_note']}"
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