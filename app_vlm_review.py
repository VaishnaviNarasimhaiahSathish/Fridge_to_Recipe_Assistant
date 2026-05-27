import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


VLM_OUTPUT_PATH = Path("reports/vlm_predictions_v1.jsonl")
MANUAL_GT_PATH = Path("reports/manual_ground_truth_50.csv")


def load_jsonl_latest_success(path: Path):
    """
    Load JSONL and keep the latest row per image_id.
    If an image has earlier error rows and later success rows,
    the latest row is used.
    """
    if not path.exists():
        st.error(f"VLM output file not found: {path}")
        return []

    latest_by_image = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_id = row.get("image_id")

            if image_id:
                latest_by_image[image_id] = row

    return list(latest_by_image.values())


def split_ingredient_list(value):
    """
    Splits manual ingredient lists written with either semicolons or commas.

    Examples:
    - milk;yogurt;butter
    - milk, yogurt, butter
    """

    if pd.isna(value):
        return []

    if not isinstance(value, str):
        return []

    # Support both semicolon and comma separated annotation styles
    parts = re.split(r"[;,]", value)

    return [item.strip().lower() for item in parts if item.strip()]


def load_manual_ground_truth(path: Path):
    if not path.exists():
        st.warning(f"Manual ground truth file not found: {path}")
        return {}

    df = pd.read_csv(path)

    gt_dict = {}

    for _, row in df.iterrows():
        image_id = row.get("image_id", "")

        if not image_id:
            continue

        visible_items = split_ingredient_list(row.get("visible_ingredients", ""))

        gt_dict[image_id] = {
            "visible_ingredients": visible_items,
            "image_path": row.get("image_path", ""),
        }

    return gt_dict


def extract_json_from_text(text):
    """
    Handles:
    1. Plain JSON
    2. JSON wrapped in ```json ... ```
    3. JSON with extra text before/after
    """
    if text is None:
        return None

    if not isinstance(text, str):
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL,
    )

    if fenced_match:
        try:
            return json.loads(fenced_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    object_match = re.search(r"\{.*\}", text, re.DOTALL)

    if object_match:
        try:
            return json.loads(object_match.group(0).strip())
        except json.JSONDecodeError:
            pass

    return None


def get_parsed_response(row):
    parsed = row.get("parsed_response")

    if isinstance(parsed, dict):
        return parsed

    return extract_json_from_text(row.get("raw_response"))


def ingredients_to_dataframe(parsed_response):
    if not isinstance(parsed_response, dict):
        return pd.DataFrame()

    ingredients = parsed_response.get("ingredients", [])

    if not isinstance(ingredients, list):
        return pd.DataFrame()

    rows = []

    for item in ingredients:
        if isinstance(item, dict):
            rows.append(
                {
                    "name": item.get("name", ""),
                    "quantity": item.get("quantity", ""),
                    "unit": item.get("unit", ""),
                    "confidence": item.get("confidence", ""),
                    "visual_evidence": item.get("visual_evidence", ""),
                }
            )
        elif isinstance(item, str):
            rows.append(
                {
                    "name": item,
                    "quantity": "",
                    "unit": "",
                    "confidence": "",
                    "visual_evidence": "",
                }
            )

    return pd.DataFrame(rows)


def uncertain_to_dataframe(parsed_response):
    if not isinstance(parsed_response, dict):
        return pd.DataFrame()

    uncertain_items = parsed_response.get("uncertain_items", [])

    if not isinstance(uncertain_items, list):
        uncertain_items = parsed_response.get("uncertain", [])

    if not isinstance(uncertain_items, list):
        return pd.DataFrame()

    rows = []

    for item in uncertain_items:
        if isinstance(item, dict):
            rows.append(
                {
                    "name": item.get("name", ""),
                    "reason": item.get("reason", ""),
                }
            )
        elif isinstance(item, str):
            rows.append(
                {
                    "name": item,
                    "reason": "",
                }
            )

    return pd.DataFrame(rows)


def normalize_for_display(items):
    return [str(item).strip().lower() for item in items if str(item).strip()]


def get_simple_vlm_names(ingredients_df):
    if ingredients_df.empty or "name" not in ingredients_df.columns:
        return []

    return normalize_for_display(ingredients_df["name"].tolist())


def compute_quick_overlap(manual_items, vlm_items):
    manual_set = set(normalize_for_display(manual_items))
    vlm_set = set(normalize_for_display(vlm_items))

    matched = sorted(manual_set & vlm_set)
    missed = sorted(manual_set - vlm_set)
    extra = sorted(vlm_set - manual_set)

    return matched, missed, extra


def render_tags(items, css_class):
    if not items:
        st.write("None")
        return

    html = " ".join(
        f"<span class='tag {css_class}'>{item}</span>"
        for item in items
    )

    st.markdown(html, unsafe_allow_html=True)


def inject_css():
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 98vw !important;
            padding-top: 1.2rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }

        h1 {
            font-size: 42px !important;
            font-weight: 900 !important;
            margin-bottom: 10px !important;
        }

        h2 {
            font-size: 30px !important;
        }

        h3 {
            font-size: 24px !important;
        }

        p, li, div, span, label {
            font-size: 16px !important;
        }

        section[data-testid="stSidebar"] {
            width: 340px !important;
        }

        section[data-testid="stSidebar"] * {
            font-size: 15px !important;
        }

        .subtitle-box {
            background-color: #202735;
            border-left: 6px solid #6d8fd6;
            padding: 14px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 17px !important;
            line-height: 1.45;
        }

        .tag {
            display: inline-block;
            border: 1px solid #555;
            padding: 7px 12px;
            margin: 5px 5px 5px 0;
            border-radius: 999px;
            font-size: 15px !important;
            font-weight: 600;
        }

        .gt-tag {
            background-color: #252d3a;
            border: 1px solid #53647e;
        }

        .matched-tag {
            background-color: #193b2f;
            border: 1px solid #3f8f6b;
        }

        .missed-tag {
            background-color: #3b1f1f;
            border: 1px solid #8f4a4a;
        }

        .extra-tag {
            background-color: #3b3219;
            border: 1px solid #9b7b2f;
        }

        div[data-testid="stMetricValue"] {
            font-size: 30px !important;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 15px !important;
        }

        div[data-testid="stDataFrame"] {
            font-size: 15px !important;
        }

        img {
            max-height: 760px !important;
            object-fit: contain !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="VLM Structured Output Review",
        layout="wide",
    )

    inject_css()

    st.title("Fridge-to-Recipe Assistant: VLM Output Review")

    st.markdown(
        """
        <div class="subtitle-box">
        This UI helps inspect the final 50-image structured VLM run. It shows each fridge image,
        manual open-vocabulary ground truth, VLM extracted ingredients, uncertain items,
        and quick comparison hints.
        </div>
        """,
        unsafe_allow_html=True,
    )

    vlm_rows = load_jsonl_latest_success(VLM_OUTPUT_PATH)
    manual_gt = load_manual_ground_truth(MANUAL_GT_PATH)

    if not vlm_rows:
        st.stop()

    image_ids = [row.get("image_id") for row in vlm_rows if row.get("image_id")]

    st.sidebar.header("Controls")

    selected_image_id = st.sidebar.selectbox(
        "Select image",
        image_ids,
    )

    show_raw = st.sidebar.checkbox("Show raw response", value=False)
    show_quick_overlap = st.sidebar.checkbox(
        "Show quick overlap with manual GT",
        value=True,
    )
    show_all_rows = st.sidebar.checkbox("Show run status table", value=False)

    selected_row = next(
        row for row in vlm_rows
        if row.get("image_id") == selected_image_id
    )

    selected_index = image_ids.index(selected_image_id)
    st.sidebar.write(f"Image {selected_index + 1} of {len(image_ids)}")

    parsed_response = get_parsed_response(selected_row)

    ingredients_df = ingredients_to_dataframe(parsed_response)
    uncertain_df = uncertain_to_dataframe(parsed_response)

    gt_info = manual_gt.get(selected_image_id, {})
    manual_items = gt_info.get("visible_ingredients", [])

    image_path = Path(selected_row.get("image_path", ""))

    col1, col2 = st.columns([1.15, 1])

    with col1:
        st.subheader("Input Image")

        if image_path.exists():
            image = Image.open(image_path)

            try:
                st.image(
                    image,
                    caption=selected_image_id,
                    use_container_width=True,
                )
            except TypeError:
                st.image(
                    image,
                    caption=selected_image_id,
                    use_column_width=True,
                )
        else:
            st.error(f"Image not found: {image_path}")

    with col2:
        st.subheader("Run Info")

        st.write(f"**Model:** {selected_row.get('model', '')}")
        st.write(f"**Status:** {selected_row.get('status', '')}")
        st.write(f"**Elapsed time:** {selected_row.get('elapsed_seconds', '')} seconds")
        st.write(f"**Finish reason:** {selected_row.get('finish_reason', '')}")

        st.subheader("Manual Ground Truth")
        render_tags(manual_items, "gt-tag")

        st.subheader("VLM Extracted Ingredients")

        if ingredients_df.empty:
            st.warning("No parsed VLM ingredients found for this image.")
        else:
            st.dataframe(
                ingredients_df,
                use_container_width=True,
                height=270,
            )

        st.subheader("VLM Uncertain Items")

        if uncertain_df.empty:
            st.write("No uncertain items found.")
        else:
            st.dataframe(
                uncertain_df,
                use_container_width=True,
                height=190,
            )

    if show_quick_overlap:
        st.divider()
        st.subheader("Quick Comparison Hints")

        vlm_names = get_simple_vlm_names(ingredients_df)
        matched, missed, extra = compute_quick_overlap(manual_items, vlm_names)

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("Matched names", len(matched))

        with metric_col2:
            st.metric("Missed manual items", len(missed))

        with metric_col3:
            st.metric("Extra VLM names", len(extra))

        st.markdown("### Matched")
        render_tags(matched, "matched-tag")

        st.markdown("### Missed from manual ground truth")
        render_tags(missed, "missed-tag")

        st.markdown("### Extra VLM predictions")
        st.caption(
            "These are not automatically false positives. They may be correct visible items missed during manual annotation, synonyms, or actual false positives."
        )
        render_tags(extra, "extra-tag")

    if show_all_rows:
        st.divider()
        st.subheader("Run Status Table")

        status_rows = []

        for row in vlm_rows:
            parsed = get_parsed_response(row)
            ing_df = ingredients_to_dataframe(parsed)

            status_rows.append(
                {
                    "image_id": row.get("image_id", ""),
                    "split": row.get("split", ""),
                    "status": row.get("status", ""),
                    "elapsed_seconds": row.get("elapsed_seconds", ""),
                    "finish_reason": row.get("finish_reason", ""),
                    "num_vlm_ingredients": len(ing_df),
                }
            )

        st.dataframe(
            pd.DataFrame(status_rows),
            use_container_width=True,
            height=350,
        )

    if show_raw:
        st.divider()
        st.subheader("Raw VLM Response")
        st.code(selected_row.get("raw_response", ""), language="json")


if __name__ == "__main__":
    main()