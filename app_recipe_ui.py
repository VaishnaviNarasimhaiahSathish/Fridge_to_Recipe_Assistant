import json
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.recipe.retrieve_recipes import load_recipes, retrieve_recipes, format_results
from src.data.image_resolver import resolve_image_path


VLM_OUTPUT_PATH   = Path("reports/vlm_predictions_100.jsonl")
NORMALIZATION_PATH = Path("configs/ingredient_normalization.json")

HIGH_CONFIDENCE_ONLY = True


def load_normalization_map() -> dict:
    with open(NORMALIZATION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


MEAT_KEYWORDS = {
    "chicken", "beef", "pork", "fish", "shrimp", "bacon", "turkey", "lamb",
    "meat", "sausage", "ham", "salmon", "tuna", "anchovy", "steak", "veal",
}

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


def normalize(name: str, norm_map: dict) -> str:
    result = norm_map.get(name.strip().lower(), name.strip().lower())
    if isinstance(result, list):
        return result[0] if result else name.strip().lower()
    return result


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
                predictions[image_id] = row
    return predictions


def extract_ingredients(parsed_response: dict, high_only: bool) -> list[str]:
    accepted = {"high"} if high_only else {"high", "medium"}
    names = []
    for ing in parsed_response.get("ingredients", []):
        conf = str(ing.get("confidence", "")).strip().lower()
        if conf in accepted:
            names.append(ing.get("name", "").strip().lower())
    return names


def is_vegetarian(recipe: dict) -> bool:
    for ingredient in recipe["matched"] + recipe["missing"]:
        if any(meat in ingredient for meat in MEAT_KEYWORDS):
            return False
    return True


def get_image_ids() -> list[str]:
    predictions = load_vlm_predictions()
    return sorted(
        img_id for img_id, row in predictions.items()
        if row.get("status") == "success" and row.get("parsed_response")
    )


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fridge-to-Recipe Assistant",
    page_icon="🧊",
    layout="wide",
)

st.title("🧊 Fridge-to-Recipe Assistant")
st.caption("Select a fridge image → review extracted ingredients → get recipe recommendations")

# ── Sidebar controls ──────────────────────────────────────────────────────────

st.sidebar.header("Settings")

high_only = st.sidebar.toggle(
    "High confidence ingredients only",
    value=True,
    help="Filter out medium-confidence VLM predictions (reduces false positives)"
)

top_n = st.sidebar.slider(
    "How many recipes to show",
    min_value=5, max_value=20, value=5
)

st.sidebar.divider()
st.sidebar.caption("Phase 2 finding: high-confidence filtering improves precision from 0.487 → 0.552")

# ── Main layout ───────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Step 1 — Select a fridge image")

    image_ids = get_image_ids()

    if not image_ids:
        st.error("No VLM predictions found. Check that reports/vlm_predictions_100.jsonl exists.")
        st.stop()

    selected_id = st.selectbox(
        "Choose image ID",
        options=image_ids,
        format_func=lambda x: x[:40] + "..." if len(x) > 40 else x,
    )

    # Show fridge image if available
    predictions = load_vlm_predictions()
    row = predictions.get(selected_id, {})
    image_path = resolve_image_path(row.get("image_path", ""))

    if image_path is not None:
        st.image(str(image_path), caption=selected_id, use_container_width=True)
    else:
        st.info(f"Image not found in train/valid/test splits: `{row.get('image_path', '')}`")

    st.subheader("Step 2 — Extracted ingredients")

    norm_map = load_normalization_map()
    parsed = row.get("parsed_response", {})

    if not parsed:
        st.warning("No parsed VLM response for this image.")
        st.stop()

    # Show all predictions with confidence badges
    all_ingredients = parsed.get("ingredients", [])
    st.caption(f"{len(all_ingredients)} total VLM predictions")

    kept = []
    removed = []

    for ing in all_ingredients:
        conf  = str(ing.get("confidence", "")).strip().lower()
        name  = ing.get("name", "").strip().lower()
        norm  = normalize(name, norm_map)
        accepted_conf = conf == "high" if high_only else conf in {"high", "medium"}
        ignored = norm in GENERIC_IGNORE_TERMS

        if accepted_conf and not ignored:
            kept.append({"raw": name, "normalized": norm, "confidence": conf})
        else:
            removed.append({"raw": name, "normalized": norm,
                            "confidence": conf, "ignored": ignored})

    if kept:
        st.markdown("**Kept (used for recipe matching):**")
        cols = st.columns(3)
        for i, ing in enumerate(kept):
            badge = "🟢" if ing["confidence"] == "high" else "🟡"
            cols[i % 3].markdown(f"{badge} {ing['normalized']}")
    else:
        st.warning("No ingredients passed the confidence filter.")

    with st.expander(f"Filtered out ({len(removed)} predictions)"):
        for ing in removed:
            reason = "ignored term" if ing["ignored"] else f"{ing['confidence']} confidence"
            st.caption(f"✗ {ing['normalized']} — {reason}")

with col_right:
    st.subheader("Step 3 — Recipe recommendations")

    if not kept:
        st.info("No ingredients available for recipe matching.")
    else:
        available = [ing["normalized"] for ing in kept]
        recipes   = load_recipes()

        # Rank a larger candidate pool by coverage first, then narrow with
        # search/dietary filters, so filtering doesn't starve the results.
        ranked = retrieve_recipes(available, top_n=200, recipes=recipes)

        st.caption(f"Matching against {len(available)} confirmed ingredients from {len(recipes)} recipes")

        search_col, filter_col = st.columns([1, 1])
        with search_col:
            search_query = st.text_input("🔍 Search recipes by name", "")
        with filter_col:
            diet_filter = st.radio(
                "Dietary filter",
                options=["All", "Vegetarian", "Quick (<30min)"],
                horizontal=True,
            )

        filtered = ranked
        if search_query.strip():
            q = search_query.strip().lower()
            filtered = [r for r in filtered if q in r["title"].lower()]
        if diet_filter == "Vegetarian":
            filtered = [r for r in filtered if is_vegetarian(r)]
        elif diet_filter == "Quick (<30min)":
            filtered = [r for r in filtered if r["prep_time"] < 30]

        results = filtered[:top_n]

        st.caption(f"Showing {len(results)} of {len(filtered)} matching recipes ({len(ranked)} ranked candidates)")

        if not results:
            st.warning("No recipes match your search/filter combination. Try widening the filters.")

        for rank, recipe in enumerate(results, start=1):
            coverage_pct = int(recipe["coverage"] * 100)

            st.markdown(f"#### #{rank} — {recipe['title']}")
            header_col, cov_col = st.columns([2, 1])

            with header_col:
                st.markdown(
                    f"🍽️ **{recipe['cuisine']}** &nbsp;·&nbsp; "
                    f"**{recipe['meal_type']}** &nbsp;·&nbsp; "
                    f"⏱️ {recipe['prep_time']} min"
                )

            with cov_col:
                st.metric(
                    "Coverage",
                    f"{coverage_pct}%",
                    f"{recipe['matched_count']}/{recipe['total_ingredients']} ingredients"
                )

            badge_cols = st.columns(2)
            with badge_cols[0]:
                st.markdown("**✅ You have:**")
                if recipe["matched"]:
                    st.markdown(
                        " ".join(
                            f":green-background[{i}]" for i in recipe["matched"]
                        )
                    )
                else:
                    st.markdown("—")
            with badge_cols[1]:
                st.markdown("**🛒 You need:**")
                if recipe["missing"]:
                    st.markdown(
                        " ".join(
                            f":red-background[{i}]" for i in recipe["missing"]
                        )
                    )
                else:
                    st.markdown("—")

            with st.expander("Full instructions"):
                for step_num, step in enumerate(recipe["instructions"], start=1):
                    st.markdown(f"{step_num}. {step}")

            st.divider()