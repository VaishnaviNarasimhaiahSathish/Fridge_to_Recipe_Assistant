import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


PREDICTIONS_PATH = Path("reports/vlm_predictions_raw.jsonl")
FLAT_CSV_PATH = Path("reports/vlm_predictions_flat.csv")


def load_jsonl(path: Path):
    rows = []

    if not path.exists():
        st.error(f"File not found: {path}")
        return rows

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def load_flat_csv(path: Path):
    if not path.exists():
        return None

    return pd.read_csv(path)


def extract_json_from_text(text: str):
    if text is None:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
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


def parse_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        if value.strip() == "":
            return []
        return [item.strip() for item in value.split(";") if item.strip()]

    return []


def get_response_lists(row):
    parsed_response = row.get("parsed_response")

    if not isinstance(parsed_response, dict):
        parsed_response = extract_json_from_text(row.get("raw_response", ""))

    if isinstance(parsed_response, dict):
        ingredients = parsed_response.get("ingredients", [])
        uncertain = parsed_response.get("uncertain", [])
        return parse_list(ingredients), parse_list(uncertain)

    return [], []


def inject_css():
    st.markdown(
        """
        <style>
        /* Make full app wider */
        .main .block-container {
            max-width: 95vw !important;
            padding-top: 1.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* Bigger page title */
        .main-title {
            font-size: 54px !important;
            font-weight: 900 !important;
            margin-bottom: 20px !important;
            line-height: 1.1 !important;
        }

        .subtitle-box {
            background-color: #202735;
            border-left: 6px solid #6d8fd6;
            padding: 22px 28px;
            border-radius: 10px;
            margin-bottom: 32px;
            font-size: 22px !important;
            line-height: 1.5;
        }

        /* Bigger headings */
        h1 {
            font-size: 54px !important;
        }

        h2 {
            font-size: 40px !important;
        }

        h3 {
            font-size: 30px !important;
        }

        /* Bigger normal text */
        p, li, div, span, label {
            font-size: 21px !important;
        }

        /* Sidebar larger */
        section[data-testid="stSidebar"] {
            width: 360px !important;
        }

        section[data-testid="stSidebar"] * {
            font-size: 19px !important;
        }

        /* Make selectbox readable */
        div[data-baseweb="select"] {
            font-size: 19px !important;
        }

        /* Info cards */
        .info-card {
            background-color: #1f1f1f;
            padding: 24px 28px;
            border-radius: 16px;
            border: 1px solid #444;
            margin-bottom: 20px;
        }

        .card-title {
            font-size: 22px !important;
            font-weight: 800;
            margin-bottom: 10px;
            color: #f5f5f5;
        }

        .card-text {
            font-size: 22px !important;
            color: #e8e8e8;
        }

        /* Ingredient tags */
        .tag {
            display: inline-block;
            border: 1px solid #555;
            padding: 10px 16px;
            margin: 7px 7px 7px 0;
            border-radius: 999px;
            font-size: 20px !important;
            font-weight: 600;
        }

        .ingredient-tag {
            background-color: #193b2f;
            border: 1px solid #3f8f6b;
        }

        .uncertain-tag {
            background-color: #3b3219;
            border: 1px solid #9b7b2f;
        }

        .reference-tag {
            background-color: #252d3a;
            border: 1px solid #53647e;
        }

        /* Metrics */
        div[data-testid="stMetricValue"] {
            font-size: 44px !important;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 22px !important;
        }

        /* Dataframe text */
        div[data-testid="stDataFrame"] {
            font-size: 18px !important;
        }

        /* Image caption bigger */
        div[data-testid="stImageCaption"] {
            font-size: 18px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_tags(items, tag_class="tag"):
    if not items:
        st.write("None found")
        return

    tag_html = " ".join(
        [f"<span class='tag {tag_class}'>{item}</span>" for item in items]
    )
    st.markdown(tag_html, unsafe_allow_html=True)


def render_info_card(title, text):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">{title}</div>
            <div class="card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="VLM Trial Review",
        layout="wide",
    )

    inject_css()

    st.markdown(
    """
    <div class="main-title">
    Fridge-to-Recipe Assistant: VLM Trial Review
    </div>
    """,
    unsafe_allow_html=True,
)

    st.markdown(
    """
    <div class="subtitle-box">
    This UI is for reviewing the preliminary <b>open-vocabulary VLM ingredient extraction</b> results.
    The VLM receives only the image and the prompt. Dataset labels are not shown to the VLM and are only optional partial reference metadata.
    </div>
    """,
    unsafe_allow_html=True,
)

    prediction_rows = load_jsonl(PREDICTIONS_PATH)
    flat_df = load_flat_csv(FLAT_CSV_PATH)

    if not prediction_rows:
        st.stop()

    st.sidebar.header("Controls")

    image_ids = [row["image_id"] for row in prediction_rows]

    selected_image_id = st.sidebar.selectbox(
        "Select image",
        image_ids,
    )

    selected_row = next(
        row for row in prediction_rows
        if row["image_id"] == selected_image_id
    )

    selected_index = image_ids.index(selected_image_id)

    st.sidebar.write(f"Image {selected_index + 1} of {len(image_ids)}")

    show_dataset_reference = st.sidebar.checkbox(
        "Show dataset labels as partial reference",
        value=False,
    )

    show_raw_response = st.sidebar.checkbox(
        "Show raw VLM response",
        value=False,
    )

    show_table = st.sidebar.checkbox(
        "Show full trial table",
        value=True,
    )

    image_path = Path(selected_row["image_path"])

    ingredients, uncertain = get_response_lists(selected_row)
    dataset_reference = parse_list(selected_row.get("ground_truth_class_names", ""))

    col1, col2 = st.columns([1.45, 1])

    with col1:
        st.subheader("Input Image")

        if image_path.exists():
            image = Image.open(image_path)
            try:
                st.image(image, caption=selected_image_id, use_container_width=True)
            except TypeError:
                st.image(image, caption=selected_image_id, use_column_width=True)
        else:
            st.error(f"Image file not found: {image_path}")

    with col2:
        st.subheader("Open-Vocabulary VLM Output")

        render_info_card("Split", selected_row.get("split", ""))
        render_info_card("Model", selected_row.get("model", ""))
        render_info_card("Status", selected_row.get("status", ""))

        st.markdown("### Predicted visible ingredients")
        render_tags(ingredients, "ingredient-tag")

        st.markdown("### Uncertain / unclear items")
        render_tags(uncertain, "uncertain-tag")

        if show_dataset_reference:
            st.markdown("### Dataset labels, partial reference only")
            st.caption(
                "These labels were not given to the VLM. They cover only 22 selected dataset classes and are not complete ground truth for open-vocabulary extraction."
            )
            render_tags(dataset_reference, "reference-tag")

    st.divider()

    st.subheader("Quick Counts")

    if show_dataset_reference:
        count_col1, count_col2, count_col3 = st.columns(3)

        with count_col1:
            st.metric("Predicted ingredients", len(ingredients))

        with count_col2:
            st.metric("Uncertain items", len(uncertain))

        with count_col3:
            st.metric("Dataset reference labels", len(dataset_reference))

    else:
        count_col1, count_col2 = st.columns(2)

        with count_col1:
            st.metric("Predicted ingredients", len(ingredients))

        with count_col2:
            st.metric("Uncertain items", len(uncertain))

    if show_table and flat_df is not None:
        st.divider()
        st.subheader("Overall Trial Table")

        display_columns = [
            "image_id",
            "split",
            "parse_status",
            "ingredients",
            "uncertain",
            "num_predicted_ingredients",
            "num_uncertain_items",
            "num_open_vocab_predictions",
        ]

        if show_dataset_reference:
            display_columns.insert(3, "ground_truth_class_names")
            display_columns.append("num_in_dataset_predictions")

        available_columns = [
            col for col in display_columns
            if col in flat_df.columns
        ]

        st.dataframe(
            flat_df[available_columns],
            use_container_width=True,
            height=470,
        )

    if show_raw_response:
        st.divider()
        st.subheader("Raw VLM Response")
        st.code(selected_row.get("raw_response", ""), language="json")


if __name__ == "__main__":
    main()