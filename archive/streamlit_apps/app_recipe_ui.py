import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.recipe.retrieve_recipes_hybrid import load_recipes, retrieve_recipes_hybrid


VLM_OUTPUT_PATH = Path("reports/vlm_predictions_100.jsonl")
NORMALIZATION_PATH = Path("configs/ingredient_normalization.json")


IMAGE_SEARCH_DIRS = [
    Path("data/annotations/final_images_200"),
    Path("data/raw/train/images"),
    Path("data/raw/valid/images"),
    Path("data/raw/test/images"),
    Path("reports/manual_annotation_images"),
]


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
    "water", "juice", "orange juice", "lime juice", "soda",
    "broth", "beer", "wine", "cider", "lemonade", "ice",
}


PASSAU_SHOPS = {
    "Innstadt": [
        {
            "name": "EDEKA Maier",
            "address": "Kapuzinerstraße 30",
            "type": "Wider selection",
            "note": "Good for fresh items and slightly specific ingredients.",
        },
        {
            "name": "NORMA",
            "address": "Kapuzinerstraße 42",
            "type": "Budget",
            "note": "Good for basic groceries and staples.",
        },
    ],
    "Altstadt / City Centre": [
        {
            "name": "REWE",
            "address": "Nibelungenplatz 5",
            "type": "Wider selection",
            "note": "Convenient central option for most missing ingredients.",
        },
        {
            "name": "Netto Marken-Discount",
            "address": "Bahnhofstraße 24",
            "type": "Budget",
            "note": "Good for simple staples and cheaper basics.",
        },
        {
            "name": "nah & gut Escherich",
            "address": "Residenzplatz 13",
            "type": "Wider selection",
            "note": "Small central store. Check opening hours before going.",
        },
    ],
    "Haidenhof-Nord (Neuburger Straße)": [
        {
            "name": "EDEKA Schwaiberger",
            "address": "Neuburger Straße 104B",
            "type": "Wider selection",
            "note": "Good for fresh produce and recipe-specific items.",
        },
        {
            "name": "Kaufland",
            "address": "Neuburger Straße 128",
            "type": "Wider selection",
            "note": "Large store, useful for bigger shopping trips.",
        },
    ],
    "Spitalhof": [
        {
            "name": "REWE",
            "address": "Spitalhofstraße 94",
            "type": "Wider selection",
            "note": "Good general-purpose supermarket.",
        },
        {
            "name": "PENNY",
            "address": "Spitalhofstraße 76",
            "type": "Budget",
            "note": "Good for affordable basics.",
        },
    ],
    "Grubweg / Lindau": [
        {
            "name": "Netto Marken-Discount",
            "address": "Schulbergstraße 63",
            "type": "Budget",
            "note": "Good for simple missing items.",
        },
        {
            "name": "NORMA Filiale",
            "address": "Dr.-Fritz-Ebbert-Straße 1",
            "type": "Budget",
            "note": "Good for staples and everyday groceries.",
        },
        {
            "name": "Lidl",
            "address": "Lindau 9",
            "type": "Budget",
            "note": "Good for vegetables, dairy, and basic pantry items.",
        },
    ],
    "Neustift / Auerbach": [
        {
            "name": "Lidl",
            "address": "Graneckerstraße 6",
            "type": "Budget",
            "note": "Good for basic ingredients at low prices.",
        },
        {
            "name": "ALDI SÜD",
            "address": "Graneckerstraße 2",
            "type": "Budget",
            "note": "Right next to Lidl, useful for quick comparison.",
        },
        {
            "name": "ALDI",
            "address": "Neuburger Straße 137",
            "type": "Budget",
            "note": "Good for staples and simple groceries.",
        },
        {
            "name": "REWE",
            "address": "Steinbachstraße 60",
            "type": "Wider selection",
            "note": "Better option for less common ingredients.",
        },
        {
            "name": "PENNY",
            "address": "Max-Matheis-Straße 2A",
            "type": "Budget",
            "note": "Good for cheaper everyday items.",
        },
    ],
    "Hacklberg": [
        {
            "name": "EDEKA Hehenberger",
            "address": "Glockenstraße 6",
            "type": "Wider selection",
            "note": "Good local option. Check opening hours before going.",
        },
    ],
}


MEAT_KEYWORDS = {
    "chicken", "beef", "pork", "fish", "shrimp", "bacon", "turkey", "lamb",
    "meat", "sausage", "ham", "salmon", "tuna", "anchovy", "steak", "veal",
}


def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def load_normalization_map() -> dict:
    if not NORMALIZATION_PATH.exists():
        return {}

    with open(NORMALIZATION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(name: str, norm_map: dict) -> str:
    cleaned = name.strip().lower()
    result = norm_map.get(cleaned, cleaned)

    if isinstance(result, list):
        return result[0] if result else cleaned

    return result


def resolve_image_path(original_path: str) -> Path | None:
    original = Path(str(original_path))

    candidates = []

    if original.exists():
        candidates.append(original)

    filename = original.name

    for directory in IMAGE_SEARCH_DIRS:
        candidates.append(directory / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_vlm_predictions() -> dict:
    predictions = {}

    if not VLM_OUTPUT_PATH.exists():
        return predictions

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
            if image_id:
                image_path = resolve_image_path(row.get("image_path", ""))

                if image_path is not None:
                    predictions[image_id] = row

    return predictions


def get_detected_ingredients(prediction_row: dict) -> list[str]:
    norm_map = load_normalization_map()
    parsed = prediction_row.get("parsed_response", {})
    ingredients = parsed.get("ingredients", [])

    kept = []

    for ingredient in ingredients:
        raw_name = ingredient.get("name", "").strip().lower()
        confidence = ingredient.get("confidence", "").strip().lower()
        normalized = normalize(raw_name, norm_map)

        if not normalized:
            continue

        if normalized in GENERIC_IGNORE_TERMS:
            continue

        if confidence != "high":
            continue

        if normalized not in kept:
            kept.append(normalized)

    return kept


def is_vegetarian_recipe(recipe: dict) -> bool:
    all_ingredients = recipe.get("matched", []) + recipe.get("missing", [])

    for ingredient in all_ingredients:
        lower_ingredient = ingredient.lower()
        if any(meat in lower_ingredient for meat in MEAT_KEYWORDS):
            return False

    return True


def format_recipe_time(prep_time):
    if prep_time is None:
        return "Time not listed"

    return f"{prep_time} min"


def render_pills(items: list[str], class_name: str) -> str:
    if not items:
        return "<span class='empty-text'>None</span>"

    return " ".join(
        f"<span class='pill {class_name}'>{item}</span>"
        for item in items
    )


def inject_css():
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }

        .hero {
            background: linear-gradient(135deg, #172033 0%, #101624 60%, #111827 100%);
            border: 1px solid #273348;
            border-radius: 28px;
            padding: 54px 58px;
            margin-top: 20px;
            margin-bottom: 28px;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25);
        }

        .hero-title {
            font-size: 52px;
            font-weight: 900;
            letter-spacing: -1.4px;
            margin-bottom: 18px;
            color: #f8fafc;
        }

        .hero-subtitle {
            font-size: 20px;
            line-height: 1.6;
            max-width: 760px;
            color: #cbd5e1;
            margin-bottom: 10px;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 14px;
        }

        .step-label {
            display: inline-block;
            padding: 7px 14px;
            border-radius: 999px;
            background: #203047;
            color: #cbd5e1;
            font-weight: 700;
            font-size: 13px;
            margin-bottom: 14px;
        }

        .section-title {
            font-size: 30px;
            font-weight: 850;
            margin-bottom: 10px;
            color: #f8fafc;
        }

        .section-subtitle {
            font-size: 16px;
            color: #94a3b8;
            margin-bottom: 20px;
        }

        .soft-card {
            background-color: #151c2c;
            border: 1px solid #293247;
            border-radius: 24px;
            padding: 28px;
            margin-bottom: 24px;
        }

        .recipe-card {
            background-color: #151c2c;
            border: 1px solid #293247;
            border-radius: 24px;
            padding: 28px;
            margin-bottom: 24px;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
        }

        .recipe-title {
            font-size: 28px;
            font-weight: 850;
            color: #f8fafc;
            margin-bottom: 8px;
        }

        .recipe-meta {
            color: #94a3b8;
            font-size: 15px;
            margin-bottom: 18px;
        }

        .match-box {
            background: #10251d;
            border: 1px solid #2d8a63;
            color: #d1fae5;
            border-radius: 20px;
            padding: 18px;
            text-align: center;
        }

        .match-number {
            font-size: 38px;
            line-height: 1;
            font-weight: 900;
        }

        .match-label {
            font-size: 13px;
            color: #a7f3d0;
            margin-top: 6px;
        }

        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 7px 13px;
            margin: 5px 6px 5px 0;
            font-weight: 650;
            font-size: 14px;
        }

        .pill-have {
            background: #123524;
            border: 1px solid #327a55;
            color: #dcfce7;
        }

        .pill-need {
            background: #3b2413;
            border: 1px solid #a16207;
            color: #fef3c7;
        }

        .pill-shop-budget {
            background: #26354f;
            border: 1px solid #3b82f6;
            color: #dbeafe;
        }

        .pill-shop-selection {
            background: #35264b;
            border: 1px solid #8b5cf6;
            color: #ede9fe;
        }

        .pill-label {
            color: #e2e8f0;
            font-weight: 800;
            margin-top: 14px;
            margin-bottom: 6px;
        }

        .empty-text {
            color: #64748b;
        }

        .shop-card {
            background: #101827;
            border: 1px solid #2a354a;
            border-radius: 18px;
            padding: 18px;
            margin-top: 10px;
            margin-bottom: 12px;
        }

        .shop-name {
            font-size: 19px;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 4px;
        }

        .shop-address {
            color: #cbd5e1;
            margin-bottom: 8px;
        }

        .shop-note {
            color: #94a3b8;
            font-size: 14px;
        }

        div.stButton > button {
            border-radius: 999px;
            padding: 0.65rem 1.2rem;
            font-weight: 800;
            border: 1px solid #334155;
        }

        div.stButton > button[kind="primary"] {
            background: #22c55e;
            border: 1px solid #22c55e;
            color: #052e16;
        }

        img {
            border-radius: 22px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state():
    defaults = {
        "stage": "landing",
        "selected_image_id": None,
        "detected_ingredients": [],
        "recipes": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_app():
    st.session_state.stage = "landing"
    st.session_state.selected_image_id = None
    st.session_state.detected_ingredients = []
    st.session_state.recipes = []


def show_landing(predictions: dict):
    st.markdown(
        """
        <div class="hero">
            <div class="step-label">Fridge image → recipe ideas</div>
            <div class="hero-title">Fridge-to-Recipe Assistant</div>
            <div class="hero-subtitle">
                Turn a fridge image into recipe ideas. The assistant detects visible ingredients
                and recommends recipes you can cook with minimal extra shopping.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Choose a fridge image</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Pick one image and start the recipe assistant.</div>',
        unsafe_allow_html=True,
    )

    image_ids = sorted(predictions.keys())

    selected = st.selectbox(
        "Select image",
        image_ids,
        format_func=lambda image_id: Path(predictions[image_id].get("image_path", image_id)).name,
        label_visibility="collapsed",
    )

    selected_path = resolve_image_path(predictions[selected].get("image_path", ""))

    preview_col, action_col = st.columns([1.2, 1])

    with preview_col:
        if selected_path is not None:
            st.image(str(selected_path), caption="Selected fridge image", use_column_width=True)

    with action_col:
        st.write("")
        st.write("")
        st.markdown(
            """
            <div class="small-muted">
                The assistant will look at the selected fridge image, identify visible ingredients,
                and then suggest recipes based on what is available.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button("Analyze fridge", type="primary", use_container_width=True):
            st.session_state.selected_image_id = selected
            st.session_state.stage = "analyzing"
            safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def show_analysis():
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analyzing your fridge</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Please wait while the assistant prepares your ingredients.</div>',
        unsafe_allow_html=True,
    )

    progress = st.progress(0)
    status = st.empty()

    steps = [
        ("Scanning fridge image...", 25),
        ("Detecting ingredients...", 60),
        ("Preparing recipe suggestions...", 100),
    ]

    for message, value in steps:
        status.markdown(f"### {message}")
        progress.progress(value)
        time.sleep(0.7)

    selected_id = st.session_state.selected_image_id
    predictions = load_vlm_predictions()
    row = predictions.get(selected_id, {})

    st.session_state.detected_ingredients = get_detected_ingredients(row)
    st.session_state.stage = "ingredients"

    time.sleep(0.3)
    safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def show_ingredients(predictions: dict):
    selected_id = st.session_state.selected_image_id
    row = predictions.get(selected_id, {})
    image_path = resolve_image_path(row.get("image_path", ""))
    ingredients = st.session_state.detected_ingredients

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-label">Step 1 complete</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Detected ingredients</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">These are the ingredients the assistant found in the fridge image.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        if image_path is not None:
            st.image(str(image_path), caption="Analyzed fridge image", use_column_width=True)

    with right:
        if ingredients:
            st.markdown('<div class="pill-label">Available ingredients</div>', unsafe_allow_html=True)
            st.markdown(render_pills(ingredients, "pill-have"), unsafe_allow_html=True)
        else:
            st.warning("No clear ingredients were detected for this image. Try another image.")

        st.write("")

        button_col1, button_col2 = st.columns(2)

        with button_col1:
            if st.button("Find recipes", type="primary", use_container_width=True, disabled=not bool(ingredients)):
                recipes = load_recipes()
                ranked = retrieve_recipes_hybrid(
                    available_ingredients=ingredients,
                    top_n=30,
                    candidate_limit=100,
                    recipes=recipes,
                )
                st.session_state.recipes = ranked
                st.session_state.stage = "recipes"
                safe_rerun()

        with button_col2:
            if st.button("Choose another image", use_container_width=True):
                reset_app()
                safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def show_shop_selector():
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Where would you like to shop?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Choose your area first, then pick a nearby shop for missing ingredients.</div>',
        unsafe_allow_html=True,
    )

    area = st.selectbox(
        "Select your area",
        list(PASSAU_SHOPS.keys()),
    )

    shops = PASSAU_SHOPS[area]

    shop_names = [
        f"{shop['name']} — {shop['address']}"
        for shop in shops
    ]

    selected_shop_label = st.selectbox(
        "Select a shop",
        shop_names,
    )

    selected_index = shop_names.index(selected_shop_label)
    selected_shop = shops[selected_index]

    badge_class = (
        "pill-shop-budget"
        if selected_shop["type"] == "Budget"
        else "pill-shop-selection"
    )

    st.markdown(
        f"""
        <div class="shop-card">
            <div class="shop-name">{selected_shop["name"]}</div>
            <div class="shop-address">{selected_shop["address"]}</div>
            <span class="pill {badge_class}">{selected_shop["type"]}</span>
            <div class="shop-note">{selected_shop["note"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    return selected_shop


def show_recipe_card(recipe: dict, selected_shop: dict):
    coverage_pct = int(round(recipe.get("coverage", 0) * 100))
    matched = recipe.get("matched", [])
    missing = recipe.get("missing", [])
    instructions = recipe.get("instructions", [])

    st.markdown('<div class="recipe-card">', unsafe_allow_html=True)

    title_col, match_col = st.columns([3, 1])

    with title_col:
        st.markdown(
            f'<div class="recipe-title">{recipe.get("title", "Untitled recipe")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="recipe-meta">
                {recipe.get("cuisine", "Cuisine not listed")} ·
                {recipe.get("meal_type", "Meal type not listed")} ·
                {format_recipe_time(recipe.get("prep_time"))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with match_col:
        st.markdown(
            f"""
            <div class="match-box">
                <div class="match-number">{coverage_pct}%</div>
                <div class="match-label">match</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="pill-label">Matched ingredients</div>', unsafe_allow_html=True)
        st.markdown(render_pills(matched, "pill-have"), unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="pill-label">Still needed</div>', unsafe_allow_html=True)
        st.markdown(render_pills(missing, "pill-need"), unsafe_allow_html=True)

    if missing:
        st.markdown('<div class="pill-label">Suggested shopping option</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="shop-card">
                <div class="shop-name">{selected_shop["name"]}</div>
                <div class="shop-address">{selected_shop["address"]}</div>
                <div class="shop-note">
                    You can check this shop for: {", ".join(missing[:4])}
                    {"..." if len(missing) > 4 else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Instructions"):
        if instructions:
            for index, step in enumerate(instructions, start=1):
                st.markdown(f"**{index}.** {step}")
        else:
            st.write("No instructions available for this recipe.")

    st.markdown("</div>", unsafe_allow_html=True)


def show_recipes():
    recipes = st.session_state.recipes

    st.markdown('<div class="step-label">Recipe suggestions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recipes you can make</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Pick a nearby shopping area and review the missing ingredients for each recipe.</div>',
        unsafe_allow_html=True,
    )

    if not recipes:
        st.warning("No recipes found for this image.")
        if st.button("Choose another image"):
            reset_app()
            safe_rerun()
        return

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])

    with filter_col1:
        recipe_count = st.selectbox(
            "Number of recipes",
            [3, 5, 8, 10],
            index=1,
        )

    with filter_col2:
        food_filter = st.selectbox(
            "Preference",
            ["All recipes", "Vegetarian", "Quick meals"],
        )

    with filter_col3:
        if st.button("Choose another image", use_container_width=True):
            reset_app()
            safe_rerun()

    filtered_recipes = recipes

    if food_filter == "Vegetarian":
        filtered_recipes = [
            recipe for recipe in filtered_recipes
            if is_vegetarian_recipe(recipe)
        ]

    elif food_filter == "Quick meals":
        filtered_recipes = [
            recipe for recipe in filtered_recipes
            if recipe.get("prep_time") is not None and recipe.get("prep_time") <= 30
        ]

    selected_shop = show_shop_selector()

    for recipe in filtered_recipes[:recipe_count]:
        show_recipe_card(recipe, selected_shop)


def main():
    st.set_page_config(
        page_title="Fridge-to-Recipe Assistant",
        page_icon="🧊",
        layout="wide",
    )

    inject_css()
    initialize_state()

    predictions = load_vlm_predictions()

    if not predictions:
        st.error(
            "No usable fridge images were found. Please check that the prediction file exists and the images are available locally."
        )
        return

    if st.session_state.stage == "landing":
        show_landing(predictions)

    elif st.session_state.stage == "analyzing":
        show_analysis()

    elif st.session_state.stage == "ingredients":
        show_ingredients(predictions)

    elif st.session_state.stage == "recipes":
        show_recipes()


if __name__ == "__main__":
    main()