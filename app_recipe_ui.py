import json
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.recipe.retrieve_recipes import load_recipes, retrieve_recipes, format_results


VLM_OUTPUT_PATH   = Path("reports/vlm_predictions_100.jsonl")
NORMALIZATION_PATH = Path("configs/ingredient_normalization.json")

HIGH_CONFIDENCE_ONLY = True


def load_normalization_map() -> dict:
    with open(NORMALIZATION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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
    "Number of recipes to recommend",
    min_value=1, max_value=10, value=5
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
    image_path = Path(row.get("image_path", ""))

    if image_path.exists():
        st.image(str(image_path), caption=selected_id, use_container_width=True)
    else:
        st.info(f"Image file not found locally: `{image_path}`")

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
        results   = retrieve_recipes(available, top_n=top_n, recipes=recipes)

        st.caption(f"Matching against {len(available)} confirmed ingredients from {len(recipes)} recipes")

        for rank, recipe in enumerate(results, start=1):
            coverage_pct = int(recipe["coverage"] * 100)

            with st.expander(
                f"#{rank} — {recipe['title']}  ·  {coverage_pct}% coverage",
                expanded=(rank <= 2),
            ):
                meta_col, cov_col = st.columns([2, 1])

                with meta_col:
                    st.markdown(f"**Cuisine:** {recipe['cuisine']}")
                    st.markdown(f"**Meal type:** {recipe['meal_type']}")
                    st.markdown(f"**Prep time:** {recipe['prep_minutes']} min")

                with cov_col:
                    st.metric(
                        "Ingredient coverage",
                        f"{coverage_pct}%",
                        f"{recipe['matched_count']}/{recipe['total_ingredients']} ingredients"
                    )

                st.markdown("**✅ You have:**")
                st.markdown(", ".join(f"`{i}`" for i in recipe["matched"]) or "—")

                if recipe["missing"]:
                    st.markdown("**🛒 You need:**")
                    st.markdown(", ".join(f"`{i}`" for i in recipe["missing"]))

                st.divider()
                st.markdown("**Instructions**")
                st.write(recipe["instructions"])