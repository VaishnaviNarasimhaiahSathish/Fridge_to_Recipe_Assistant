import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grocery.shops import PASSAU_SHOPS, SHOP_DISCLAIMER
from src.ingredients.prediction_processing import (
    extract_detected_ingredients as extract_prediction_ingredients,
    load_normalization_map,
)
from src.recipe.retrieve_recipes_hybrid import load_recipes, retrieve_recipes_hybrid


VLM_OUTPUT_PATH = PROJECT_ROOT / "reports" / "vlm_predictions_100.jsonl"
NORMALIZATION_PATH = PROJECT_ROOT / "configs" / "ingredient_normalization.json"


IMAGE_SEARCH_DIRS = [
    PROJECT_ROOT / "data" / "annotations" / "final_images_200",
    PROJECT_ROOT / "data" / "raw" / "train" / "images",
    PROJECT_ROOT / "data" / "raw" / "valid" / "images",
    PROJECT_ROOT / "data" / "raw" / "test" / "images",
    PROJECT_ROOT / "reports" / "manual_annotation_images",
]


MEAT_KEYWORDS = {
    "chicken",
    "beef",
    "pork",
    "fish",
    "shrimp",
    "bacon",
    "turkey",
    "lamb",
    "meat",
    "sausage",
    "ham",
    "salmon",
    "tuna",
    "anchovy",
    "steak",
    "veal",
}


app = FastAPI(
    title="Fridge-to-Recipe Assistant API",
    description="Backend API for VLM-based fridge ingredient detection and recipe recommendation.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecipeRequest(BaseModel):
    image_id: str
    top_n: int = 8
    preference: str = "all"


def get_filename_from_path(path_value: str) -> str:
    """
    Works for both Mac/Linux paths and Windows-style paths.
    """
    cleaned = str(path_value).replace("\\", "/")
    return cleaned.split("/")[-1]


def resolve_image_path(original_path: str) -> Path | None:
    filename = get_filename_from_path(original_path)

    original_candidate = PROJECT_ROOT / original_path
    if original_candidate.exists():
        return original_candidate

    for directory in IMAGE_SEARCH_DIRS:
        candidate = directory / filename
        if candidate.exists():
            return candidate

    return None


def load_predictions(include_missing_images: bool = False) -> dict[str, dict[str, Any]]:
    if not VLM_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Prediction file not found: {VLM_OUTPUT_PATH}")

    predictions = {}

    with open(VLM_OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
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

            image_path = resolve_image_path(row.get("image_path", ""))

            if image_path is None and not include_missing_images:
                continue

            row["_resolved_image_path"] = str(image_path) if image_path else None
            row["_filename"] = get_filename_from_path(row.get("image_path", image_id))

            predictions[image_id] = row

    return predictions


def extract_detected_ingredients(prediction_row: dict[str, Any]) -> list[str]:
    """
    Extract the user-facing ingredient list from a saved VLM prediction row.

    The actual extraction, normalization, confidence filtering, and generic-term
    filtering live in src/ingredients/prediction_processing.py.
    This wrapper keeps backend API behavior unchanged.
    """
    normalization_map = load_normalization_map(NORMALIZATION_PATH)

    return extract_prediction_ingredients(
        prediction_row=prediction_row,
        normalization_map=normalization_map,
        allowed_confidences={"high"},
    )


def is_vegetarian_recipe(recipe: dict[str, Any]) -> bool:
    ingredients = recipe.get("matched", []) + recipe.get("missing", [])

    for ingredient in ingredients:
        ingredient_lower = str(ingredient).lower()
        if any(meat in ingredient_lower for meat in MEAT_KEYWORDS):
            return False

    return True


def simplify_recipe_for_ui(recipe: dict[str, Any]) -> dict[str, Any]:
    """
    Remove backend/search-specific details from API response.
    The frontend should stay user-facing.
    """
    return {
        "title": recipe.get("title", "Untitled recipe"),
        "cuisine": recipe.get("cuisine", "Cuisine not listed"),
        "meal_type": recipe.get("meal_type", "Meal type not listed"),
        "prep_time": recipe.get("prep_time"),
        "match_percentage": int(round(recipe.get("coverage", 0) * 100)),
        "matched_ingredients": recipe.get("matched", []),
        "missing_ingredients": recipe.get("missing", []),
        "missing_difficulty": recipe.get("missing_difficulty", "medium"),
        "missing_difficulty_reason": recipe.get(
            "missing_difficulty_reason",
            "Some recipe ingredients are still missing.",
        ),
        "missing_difficulty_details": recipe.get(
            "missing_difficulty_details",
            {
                "basic_staples": [],
                "common_grocery": [],
                "special_ingredients": [],
            },
        ),
        "instructions": recipe.get("instructions", []),
    }


@app.get("/")
def root():
    return {
        "message": "Fridge-to-Recipe Assistant API is running.",
        "available_endpoints": [
            "/api/images",
            "/api/images/{image_id}",
            "/api/images/{image_id}/ingredients",
            "/api/recipes",
            "/api/shops",
        ],
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/images")
def get_images():
    predictions = load_predictions(include_missing_images=False)

    images = []

    for image_id, row in predictions.items():
        filename = row.get("_filename", image_id)
        image_url = f"/api/image-file/{quote(image_id, safe='')}"

        images.append(
            {
                "image_id": image_id,
                "filename": filename,
                "image_url": image_url,
            }
        )

    images = sorted(images, key=lambda x: x["filename"])

    return {
        "count": len(images),
        "images": images,
    }


@app.get("/api/images/{image_id}")
def get_image_details(image_id: str):
    predictions = load_predictions(include_missing_images=False)

    if image_id not in predictions:
        raise HTTPException(status_code=404, detail="Image not found locally.")

    row = predictions[image_id]
    ingredients = extract_detected_ingredients(row)

    return {
        "image_id": image_id,
        "filename": row.get("_filename", image_id),
        "image_url": f"/api/image-file/{quote(image_id, safe='')}",
        "detected_ingredients": ingredients,
    }


@app.get("/api/image-file/{image_id}")
def get_image_file(image_id: str):
    predictions = load_predictions(include_missing_images=False)

    if image_id not in predictions:
        raise HTTPException(status_code=404, detail="Image not found locally.")

    image_path = predictions[image_id].get("_resolved_image_path")

    if not image_path:
        raise HTTPException(status_code=404, detail="Image file missing.")

    path = Path(image_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing.")

    return FileResponse(path)


@app.get("/api/images/{image_id}/ingredients")
def get_ingredients(image_id: str):
    predictions = load_predictions(include_missing_images=False)

    if image_id not in predictions:
        raise HTTPException(status_code=404, detail="Image not found locally.")

    row = predictions[image_id]

    return {
        "image_id": image_id,
        "detected_ingredients": extract_detected_ingredients(row),
    }


@app.post("/api/recipes")
def get_recipes(request: RecipeRequest):
    predictions = load_predictions(include_missing_images=False)

    if request.image_id not in predictions:
        raise HTTPException(status_code=404, detail="Image not found locally.")

    row = predictions[request.image_id]
    ingredients = extract_detected_ingredients(row)

    if not ingredients:
        return {
            "image_id": request.image_id,
            "detected_ingredients": [],
            "recipes": [],
        }

    recipes = load_recipes()

    ranked = retrieve_recipes_hybrid(
        available_ingredients=ingredients,
        top_n=50,
        candidate_limit=120,
        recipes=recipes,
    )

    preference = request.preference.strip().lower()

    if preference == "vegetarian":
        ranked = [recipe for recipe in ranked if is_vegetarian_recipe(recipe)]

    elif preference == "quick":
        ranked = [
            recipe for recipe in ranked
            if recipe.get("prep_time") is not None and recipe.get("prep_time") <= 30
        ]

    top_n = max(1, min(request.top_n, 20))
    selected = ranked[:top_n]

    return {
        "image_id": request.image_id,
        "detected_ingredients": ingredients,
        "recipes": [simplify_recipe_for_ui(recipe) for recipe in selected],
    }


@app.get("/api/shops")
def get_shops():
    return {
        "areas": PASSAU_SHOPS,
        "disclaimer": SHOP_DISCLAIMER,
    }