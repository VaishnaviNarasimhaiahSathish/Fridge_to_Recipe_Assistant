import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.recipe.retrieve_recipes import load_recipes, retrieve_recipes
from src.data.image_resolver import resolve_image_path


VLM_OUTPUT_PATH = Path("reports/vlm_predictions_100.jsonl")
NORMALIZATION_PATH = Path("configs/ingredient_normalization.json")


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


def is_vegetarian(recipe: dict) -> bool:
    for ingredient in recipe["matched"] + recipe["missing"]:
        if any(meat in ingredient for meat in MEAT_KEYWORDS):
            return False
    return True


def get_image_ids() -> tuple[list[str], int]:
    predictions = load_vlm_predictions()
    eligible = [
        img_id for img_id, row in predictions.items()
        if row.get("status") == "success" and row.get("parsed_response")
    ]

    # Some ground truth images live only on a teammate's machine and haven't
    # been pushed to the repo yet — only offer images we can actually display.
    available = [
        img_id for img_id in eligible
        if resolve_image_path(predictions[img_id].get("image_path", "")) is not None
    ]
    skipped_missing_image = len(eligible) - len(available)

    return sorted(available), skipped_missing_image


def split_counts(image_ids: list[str], predictions: dict) -> dict:
    counts = {"train": 0, "valid": 0, "test": 0}
    for img_id in image_ids:
        resolved = resolve_image_path(predictions[img_id].get("image_path", ""))
        if resolved is None:
            continue
        split = resolved.parent.parent.name
        counts[split] = counts.get(split, 0) + 1
    return counts


def coverage_color(coverage: float) -> str:
    pct = coverage * 100
    if pct >= 50:
        return "#3ddc84"
    if pct >= 30:
        return "#ffae42"
    return "#ff5c5c"


def render_pills(items: list[str], css_class: str) -> str:
    if not items:
        return "<span class='muted'>None</span>"
    return " ".join(f"<span class='pill {css_class}'>{item}</span>" for item in items)


def inject_css():
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1400px !important;
            padding-top: 1.5rem !important;
        }

        h1 { font-size: 40px !important; font-weight: 900 !important; }

        section[data-testid="stSidebar"] {
            width: 330px !important;
        }

        section[data-testid="stSidebar"] h3 {
            margin-top: 6px !important;
        }

        .subtitle {
            color: #9aa7bd;
            font-size: 16px;
            margin-bottom: 18px;
        }

        .card {
            background-color: #1c2230;
            border: 1px solid #2c3445;
            border-radius: 14px;
            padding: 20px 22px;
            margin-bottom: 18px;
        }

        .card-title {
            font-size: 20px;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 14px;
            margin: 4px 6px 4px 0;
            font-weight: 600;
            font-size: 14px;
        }

        .pill-have {
            background-color: #193b2f;
            border: 1px solid #3f8f6b;
            color: #d7f5e6;
        }

        .pill-need {
            background-color: #3b1f1f;
            border: 1px solid #8f4a4a;
            color: #ffd9d9;
        }

        .pill-medium {
            background-color: #3b3219;
            border: 1px solid #9b7b2f;
            color: #ffe8b3;
        }

        .muted { color: #6b7688; }

        .recipe-card {
            background-color: #1c2230;
            border: 1px solid #2c3445;
            border-radius: 14px;
            padding: 24px 26px;
            margin-bottom: 22px;
        }

        .recipe-title {
            font-size: 26px;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .meta-line {
            color: #9aa7bd;
            font-size: 15px;
            margin-bottom: 16px;
        }

        .coverage-wrap {
            text-align: right;
        }

        .coverage-number {
            font-size: 38px;
            font-weight: 900;
            line-height: 1;
        }

        .coverage-label {
            color: #9aa7bd;
            font-size: 13px;
        }

        .pill-section-label {
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 6px;
            margin-top: 10px;
        }

        img {
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fridge-to-Recipe Assistant",
    page_icon="🧊",
    layout="wide",
)

inject_css()

st.title("🧊 Fridge-to-Recipe Assistant")
st.markdown(
    "<div class='subtitle'>Select a fridge image → review extracted ingredients → get recipe recommendations</div>",
    unsafe_allow_html=True,
)

predictions = load_vlm_predictions()
image_ids, skipped_missing_image = get_image_ids()

if not image_ids:
    st.error("No VLM predictions found. Check that reports/vlm_predictions_100.jsonl exists.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.subheader("⚙️ Filters")

high_only = st.sidebar.toggle(
    "High confidence ingredients only",
    value=True,
    help="Filter out medium-confidence VLM predictions (reduces false positives)",
)

diet_filter = st.sidebar.radio(
    "Dietary filter",
    options=["All", "Vegetarian", "Quick (<30min)"],
)

top_n = st.sidebar.slider(
    "How many recipes to show",
    min_value=5, max_value=20, value=5,
)

st.sidebar.caption("Phase 2 finding: high-confidence filtering improves precision from 0.487 → 0.552")

st.sidebar.divider()
st.sidebar.subheader("🖼️ Dataset")

counts = split_counts(image_ids, predictions)
st.sidebar.metric("Locally available images", len(image_ids))
st.sidebar.caption(
    f"train: {counts['train']} · valid: {counts['valid']} · test: {counts['test']}"
)
if skipped_missing_image:
    st.sidebar.warning(
        f"Skipping {skipped_missing_image} images not found locally "
        "(teammate images pending upload)"
    )

st.sidebar.divider()
st.sidebar.subheader("🐞 Debug")
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# ── Main layout ───────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Step 1 — Select a fridge image</div>", unsafe_allow_html=True)

    selected_id = st.selectbox(
        "Choose image ID",
        options=image_ids,
        format_func=lambda x: x[:40] + "..." if len(x) > 40 else x,
    )

    row = predictions.get(selected_id, {})
    image_path = resolve_image_path(row.get("image_path", ""))

    if image_path is not None:
        st.image(str(image_path), caption=selected_id, use_container_width=True)
    else:
        st.info(f"Image not found in train/valid/test splits: `{row.get('image_path', '')}`")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Step 2 — Extracted ingredients</div>", unsafe_allow_html=True)

    norm_map = load_normalization_map()
    parsed = row.get("parsed_response", {})

    if not parsed:
        st.warning("No parsed VLM response for this image.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    all_ingredients = parsed.get("ingredients", [])
    st.caption(f"{len(all_ingredients)} total VLM predictions")

    kept = []
    removed = []
    debug_rows = []

    for ing in all_ingredients:
        conf = str(ing.get("confidence", "")).strip().lower()
        raw_name = ing.get("name", "").strip().lower()
        norm = normalize(raw_name, norm_map)
        accepted_conf = conf == "high" if high_only else conf in {"high", "medium"}
        ignored = norm in GENERIC_IGNORE_TERMS
        is_kept = accepted_conf and not ignored

        if is_kept:
            kept.append({"raw": raw_name, "normalized": norm, "confidence": conf})
        else:
            removed.append({"raw": raw_name, "normalized": norm, "confidence": conf, "ignored": ignored})

        reason = "kept" if is_kept else ("ignored term" if ignored else f"filtered: {conf or 'no'} confidence")
        debug_rows.append({
            "raw_extracted": raw_name,
            "normalized": norm,
            "confidence": conf,
            "result": reason,
        })

    if kept:
        st.markdown("<div class='pill-section-label'>Kept (used for recipe matching)</div>", unsafe_allow_html=True)
        high_pills = render_pills([i["normalized"] for i in kept if i["confidence"] == "high"], "pill-have")
        medium_pills = render_pills([i["normalized"] for i in kept if i["confidence"] != "high"], "pill-medium")
        st.markdown(high_pills, unsafe_allow_html=True)
        if any(i["confidence"] != "high" for i in kept):
            st.caption("🟡 medium confidence")
            st.markdown(medium_pills, unsafe_allow_html=True)
    else:
        st.warning("No ingredients passed the confidence filter.")

    with st.expander(f"Filtered out ({len(removed)} predictions)"):
        for ing in removed:
            reason = "ignored term" if ing["ignored"] else f"{ing['confidence']} confidence"
            st.caption(f"✗ {ing['normalized']} — {reason}")

    if show_debug:
        with st.expander("Debug: extraction → normalization → matching pipeline", expanded=True):
            st.write("**Pipeline trace for this image:**")
            st.dataframe(debug_rows, use_container_width=True, hide_index=True)
            st.write(f"**Ingredients passed to recipe matcher:** {[i['normalized'] for i in kept] or 'none'}")

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='card-title' style='font-size:22px;'>Step 3 — Recipe recommendations</div>", unsafe_allow_html=True)

    if not kept:
        st.info("No ingredients available for recipe matching.")
    else:
        available = [ing["normalized"] for ing in kept]
        recipes = load_recipes()

        # Rank a larger candidate pool by coverage first, then narrow with
        # search/dietary filters, so filtering doesn't starve the results.
        ranked = retrieve_recipes(available, top_n=200, recipes=recipes)

        st.caption(f"Matching against {len(available)} confirmed ingredients from {len(recipes)} recipes")

        search_query = st.text_input("🔍 Search recipes by name", "")

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

        if show_debug and results:
            with st.expander("Debug: top recipe match trace", expanded=False):
                for r in results[:3]:
                    st.write(
                        f"**{r['title']}** — coverage {r['coverage']:.2f} "
                        f"({r['matched_count']}/{r['total_ingredients']})"
                    )
                    st.caption(f"matched on: {r['matched']}")

        for recipe in results:
            coverage_pct = int(round(recipe["coverage"] * 100))
            color = coverage_color(recipe["coverage"])

            st.markdown("<div class='recipe-card'>", unsafe_allow_html=True)

            title_col, cov_col = st.columns([3, 1])
            with title_col:
                st.markdown(f"<div class='recipe-title'>{recipe['title']}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='meta-line'>🍽️ {recipe['cuisine']} &nbsp;·&nbsp; "
                    f"{recipe['meal_type']} &nbsp;·&nbsp; ⏱️ {recipe['prep_time']} min</div>",
                    unsafe_allow_html=True,
                )
            with cov_col:
                st.markdown(
                    f"<div class='coverage-wrap'>"
                    f"<div class='coverage-number' style='color:{color};'>{coverage_pct}%</div>"
                    f"<div class='coverage-label'>{recipe['matched_count']}/{recipe['total_ingredients']} ingredients</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            pill_col1, pill_col2 = st.columns(2)
            with pill_col1:
                st.markdown("<div class='pill-section-label'>✅ You have</div>", unsafe_allow_html=True)
                st.markdown(render_pills(recipe["matched"], "pill-have"), unsafe_allow_html=True)
            with pill_col2:
                st.markdown("<div class='pill-section-label'>🛒 You need</div>", unsafe_allow_html=True)
                st.markdown(render_pills(recipe["missing"], "pill-need"), unsafe_allow_html=True)

            with st.expander("Full instructions"):
                for step_num, step in enumerate(recipe["instructions"], start=1):
                    st.markdown(f"{step_num}. {step}")

            st.markdown("</div>", unsafe_allow_html=True)
