from pathlib import Path
import base64
import json
import re
import sys
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from src.data.image_resolver import resolve_image_path

st.set_page_config(page_title="Gemma VLM Annotation Review", layout="wide")

PROJECT_ROOT = Path.cwd()
CSV_PATH = PROJECT_ROOT / "data" / "annotations" / "gemma4_batch_100" / "gemma4_annotations_reviewed.csv"
BASE_URL = "https://llms.innkube.fim.uni-passau.de"
QWEN_MODEL_ID = "qwen35-397b"
MAX_IMAGE_SIZE = 512

st.title("Gemma 4 VLM Annotation Review")
st.caption("Review and correct the 100 Gemma-generated fridge image labels.")

# ── Sidebar: API key ───────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("Auto-Review with Qwen")
api_key = st.sidebar.text_input(
    "INNKUBE API key (sk-...ymYw)",
    type="password",
    placeholder="sk-...",
    help="Same key you use for Gemma annotations — stored only in memory.",
)

# ── Helper: resize + encode image ─────────────────────────────────────────
def encode_image(image_path: Path) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        longest = max(img.size)
        if longest > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / longest
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Helper: call Qwen vision ───────────────────────────────────────────────
def auto_review_image(image_path: Path, gemma_prediction: str, api_key: str) -> dict:
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "Run: pip install openai"}

    if not image_path.exists():
        return {"error": f"Image not found: {image_path}"}

    try:
        image_b64 = encode_image(image_path)
    except Exception as e:
        return {"error": f"Could not read image: {e}"}

    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=60.0)

    prompt = f"""The previous model predicted these visible ingredients: "{gemma_prediction}"

Look at this fridge image carefully and identify ALL visible food items.

Respond ONLY in this exact JSON format (no markdown, no extra text):
{{
  "corrected_ingredients": "item1, item2, item3",
  "missing_from_gemma": "items Gemma missed, or empty string if none",
  "incorrect_from_gemma": "items Gemma hallucinated, or empty string if none",
  "review_notes": "brief note: what was missed, image quality, or Gemma was accurate",
  "review_status": "ok or corrected or unclear_image"
}}"""

    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL_ID,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]}],
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        # Strip Qwen thinking tags if present
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        matched = re.search(r"\{.*\}", raw, re.DOTALL)
        if matched:
            try:
                return json.loads(matched.group(0))
            except Exception:
                pass
        return {"error": f"Could not parse response: {raw[:200]}"}
    except Exception as e:
        return {"error": str(e)}


# ── Load / save CSV ────────────────────────────────────────────────────────
if not CSV_PATH.exists():
    st.error(f"CSV not found: {CSV_PATH}")
    st.stop()


@st.cache_data(ttl=2)
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "reviewed" not in df.columns:
        df["reviewed"] = False
    if "corrected_visible_ingredients" not in df.columns:
        df["corrected_visible_ingredients"] = df["visible_ingredients"].fillna("")
    if "review_notes" not in df.columns:
        df["review_notes"] = ""
    df["review_notes"] = df["review_notes"].fillna("").astype(str)
    df["corrected_visible_ingredients"] = df["corrected_visible_ingredients"].fillna("").astype(str)
    return df


def save_data(df: pd.DataFrame) -> None:
    df.to_csv(CSV_PATH, index=False)
    st.cache_data.clear()


df = load_data(CSV_PATH)

required_cols = {"image_id", "image_path", "visible_ingredients"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

total = len(df)
reviewed_count = df["reviewed"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()

# ── Sidebar: progress ──────────────────────────────────────────────────────
st.sidebar.header("Progress")
st.sidebar.write(f"Total images: {total}")
st.sidebar.write(f"Reviewed: {reviewed_count}")
st.sidebar.write(f"Remaining: {total - reviewed_count}")
st.sidebar.progress(reviewed_count / total if total else 0)

mode = st.sidebar.radio("Navigation mode", ["Review next unreviewed", "Go to image number"])

# ── Batch auto-review ──────────────────────────────────────────────────────
if api_key:
    unreviewed_count = total - reviewed_count
    if unreviewed_count > 0:
        st.sidebar.divider()
        if st.sidebar.button(f"Auto-review ALL {unreviewed_count} unreviewed"):
            unreviewed_df = df[~df["reviewed"].astype(str).str.lower().isin(["true", "1", "yes"])]
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            errors = []

            for i, (idx_val, row) in enumerate(unreviewed_df.iterrows()):
                status_text.text(f"Processing {i+1}/{len(unreviewed_df)}: {row['image_id'][:35]}...")
                image_path = resolve_image_path(row["image_path"])
                if image_path is None:
                    errors.append(f"Row {idx_val}: Image not found in train/valid/test splits: {row['image_path']}")
                    progress_bar.progress((i + 1) / len(unreviewed_df))
                    continue
                result = auto_review_image(image_path, str(row["visible_ingredients"]), api_key)

                if "error" in result:
                    errors.append(f"Row {idx_val}: {result['error']}")
                else:
                    df.loc[idx_val, "corrected_visible_ingredients"] = result.get("corrected_ingredients", "")
                    notes = result.get("review_notes", "")
                    if result.get("missing_from_gemma"):
                        notes += f" | Missing: {result['missing_from_gemma']}"
                    if result.get("incorrect_from_gemma"):
                        notes += f" | Incorrect: {result['incorrect_from_gemma']}"
                    df.loc[idx_val, "review_notes"] = notes.strip(" |")
                    df.loc[idx_val, "reviewed"] = True

                progress_bar.progress((i + 1) / len(unreviewed_df))

                if (i + 1) % 10 == 0:
                    save_data(df)
                    status_text.text(f"Checkpoint saved at {i+1} images...")

            save_data(df)
            status_text.text("Done!")
            if errors:
                st.sidebar.warning(f"{len(errors)} errors. First: {errors[0][:100]}")
            st.rerun()
else:
    st.sidebar.info("Paste your INNKUBE key above to enable auto-review")

# ── Navigation ─────────────────────────────────────────────────────────────
if "idx" not in st.session_state:
    st.session_state.idx = 0

if mode == "Review next unreviewed":
    unreviewed_indices = df[~df["reviewed"].astype(str).str.lower().isin(["true", "1", "yes"])].index.tolist()
    if len(unreviewed_indices) == 0:
        st.success("All 100 images are reviewed!")
        st.dataframe(df, use_container_width=True)
        st.stop()
    if st.session_state.idx not in unreviewed_indices:
        st.session_state.idx = unreviewed_indices[0]
else:
    selected_number = st.sidebar.number_input(
        "Image number", min_value=1, max_value=total,
        value=min(st.session_state.idx + 1, total),
    )
    st.session_state.idx = int(selected_number) - 1

idx = st.session_state.idx
row = df.loc[idx]
image_path = resolve_image_path(row["image_path"])

# ── Main layout ────────────────────────────────────────────────────────────
left, right = st.columns([1.3, 1])

with left:
    st.subheader(f"Image {idx + 1} of {total}")
    st.write(f"Image ID: `{row['image_id']}`")
    st.write(f"Image path: `{row['image_path']}`")
    if image_path is not None:
        st.image(Image.open(image_path), use_container_width=True)
    else:
        st.error(f"Image not found in train/valid/test splits: {row['image_path']}")

with right:
    st.subheader("Gemma prediction")
    st.text_area("Original Gemma visible_ingredients",
                 value=str(row["visible_ingredients"]), height=120, disabled=True)

    # Single image auto-review
    if api_key and image_path is not None:
        if st.button("Auto-review this image with Qwen"):
            with st.spinner("Asking Qwen to review..."):
                result = auto_review_image(image_path, str(row["visible_ingredients"]), api_key)
            if "error" in result:
                st.error(result["error"])
            else:
                df.loc[idx, "corrected_visible_ingredients"] = result.get("corrected_ingredients", "")
                notes = result.get("review_notes", "")
                if result.get("missing_from_gemma"):
                    notes += f" | Missing: {result['missing_from_gemma']}"
                if result.get("incorrect_from_gemma"):
                    notes += f" | Incorrect: {result['incorrect_from_gemma']}"
                df.loc[idx, "review_notes"] = notes.strip(" |")
                save_data(df)
                st.success(f"Status: {result.get('review_status', '?')}")
                st.rerun()

    corrected = st.text_area(
        "Corrected visible ingredients",
        value=str(row.get("corrected_visible_ingredients", row["visible_ingredients"])),
        height=160,
    )

    notes = st.text_area(
        "Review notes",
        value=str(row.get("review_notes", "")),
        height=80,
        placeholder="Optional: missing item, wrong label, unclear image, etc.",
    )

    mark_reviewed = st.checkbox(
        "Mark as reviewed",
        value=str(row.get("reviewed", False)).lower() in ["true", "1", "yes"],
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Previous"):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()
    with col2:
        if st.button("Save"):
            df.loc[idx, "corrected_visible_ingredients"] = corrected.strip()
            df.loc[idx, "review_notes"] = notes.strip()
            df.loc[idx, "reviewed"] = bool(mark_reviewed)
            save_data(df)
            st.success("Saved.")
    with col3:
        if st.button("Save & Next"):
            df.loc[idx, "corrected_visible_ingredients"] = corrected.strip()
            df.loc[idx, "review_notes"] = notes.strip()
            df.loc[idx, "reviewed"] = True
            save_data(df)
            next_unreviewed = df[~df["reviewed"].astype(str).str.lower().isin(["true", "1", "yes"])].index.tolist()
            st.session_state.idx = next_unreviewed[0] if next_unreviewed else min(idx + 1, total - 1)
            st.rerun()

st.divider()
st.subheader("Current reviewed CSV preview")
st.dataframe(df.tail(10), use_container_width=True)
st.info(f"Saving to: {CSV_PATH}")