import json
from pathlib import Path
from typing import Any


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
    "preserve",
    "water",
    "juice",
    "orange juice",
    "lime juice",
    "soda",
    "broth",
    "beer",
    "wine",
    "cider",
    "lemonade",
    "ice",
}


def load_normalization_map(path: Path) -> dict[str, Any]:
    """
    Load ingredient normalization rules from a JSON file.

    The config maps raw/open-vocabulary ingredient names to normalized names.
    Example:
        "eggs" -> "egg"
        "green bell pepper" -> "bell pepper"
    """
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_ingredient(name: str, normalization_map: dict[str, Any]) -> str:
    """
    Normalize one ingredient name using the configured normalization map.

    If a mapping returns a list, the first item is used as the canonical name.
    This keeps compatibility with existing config formats.
    """
    cleaned = str(name).strip().lower()

    if not cleaned:
        return ""

    normalized = normalization_map.get(cleaned, cleaned)

    if isinstance(normalized, list):
        return str(normalized[0]).strip().lower() if normalized else cleaned

    return str(normalized).strip().lower()


def extract_detected_ingredients(
    prediction_row: dict[str, Any],
    normalization_map: dict[str, Any],
    allowed_confidences: set[str] | None = None,
    ignore_terms: set[str] | None = None,
) -> list[str]:
    """
    Extract user-facing ingredients from one saved VLM prediction row.

    Current demo behavior:
    - normalize raw VLM ingredient names
    - remove generic/non-specific terms
    - keep only high-confidence predictions
    - remove duplicates while preserving order

    This function is used by the FastAPI backend.
    It does not modify evaluation metrics or the original prediction file.
    """
    if allowed_confidences is None:
        allowed_confidences = {"high"}

    if ignore_terms is None:
        ignore_terms = GENERIC_IGNORE_TERMS

    parsed = prediction_row.get("parsed_response", {})

    if not isinstance(parsed, dict):
        return []

    raw_ingredients = parsed.get("ingredients", [])

    if not isinstance(raw_ingredients, list):
        return []

    detected = []

    for ingredient in raw_ingredients:
        if not isinstance(ingredient, dict):
            continue

        raw_name = str(ingredient.get("name", "")).strip().lower()
        confidence = str(ingredient.get("confidence", "")).strip().lower()

        if not raw_name:
            continue

        normalized = normalize_ingredient(raw_name, normalization_map)

        if not normalized:
            continue

        if normalized in ignore_terms:
            continue

        if confidence not in allowed_confidences:
            continue

        if normalized not in detected:
            detected.append(normalized)

    return detected