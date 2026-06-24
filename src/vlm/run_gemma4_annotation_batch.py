"""Create a non-overlapping 100-image Gemma 4 annotation batch.

This runner reads the existing manual ground truth only to exclude its image IDs.
It never writes to the existing annotation file or to the raw dataset.
"""

import argparse
import base64
import csv
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://llms.innkube.fim.uni-passau.de"
API_MODEL_ID = "gemma4-31b-it"
MODEL_NAME = "Gemma 4 31B-it"
ANNOTATION_BATCH = "gemma4_batch_100_new"
EXISTING_ANNOTATIONS_PATH = REPO_ROOT / "data" / "annotations" / "manual_ground_truth_100" / "manual_ground_truth_100.csv"
PROMPT_PATH = REPO_ROOT / "configs" / "vlm_prompt_with_counts.txt"
OUTPUT_PATH = REPO_ROOT / "data" / "annotations" / "gemma4_batch_100" / "gemma4_annotations_raw.csv"
RAW_DATASET_PATH = REPO_ROOT / "data" / "raw"
SPLITS = ("train", "valid", "test")
MAX_IMAGE_SIZE = 512
SLEEP_SECONDS = 1
TIMEOUT_SECONDS = 180.0


def load_existing_image_ids() -> set[str]:
    with EXISTING_ANNOTATIONS_PATH.open(newline="", encoding="utf-8") as file:
        return {row["image_id"] for row in csv.DictReader(file) if row.get("image_id")}


def select_candidates(existing_ids: set[str], limit: int) -> list[Path]:
    """Select a stable, non-overlapping sequence of raw images."""
    candidates: list[Path] = []
    seen_ids: set[str] = set()
    for split in SPLITS:
        image_dir = RAW_DATASET_PATH / split / "images"
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            if image_path.name in existing_ids or image_path.name in seen_ids:
                continue
            candidates.append(image_path)
            seen_ids.add(image_path.name)
            if len(candidates) == limit:
                return candidates
    if len(candidates) != limit:
        raise RuntimeError(f"Only found {len(candidates)} non-overlapping images; expected {limit}.")
    return candidates


def resize_and_encode_image(image_path: Path) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        longest_side = max(image.size)
        if longest_side > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / longest_side
            image = image.resize((int(image.width * scale), int(image.height * scale)))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_response(response_text: str | None) -> dict | None:
    if not isinstance(response_text, str):
        return None
    text = response_text.strip()
    for candidate in (text,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    matched = re.search(r"\{.*\}", text, re.DOTALL)
    if matched:
        try:
            return json.loads(matched.group(0))
        except json.JSONDecodeError:
            pass
    return None


def visible_ingredients(parsed: dict | None, response_text: str | None) -> str:
    if parsed and isinstance(parsed.get("ingredients"), list):
        names = []
        for ingredient in parsed["ingredients"]:
            name = ingredient.get("name") if isinstance(ingredient, dict) else ingredient
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        if names:
            return ", ".join(names)
    # Retain a non-JSON response as an annotation instead of silently discarding it.
    return (response_text or "").strip()


def load_completed_ids() -> set[str]:
    if not OUTPUT_PATH.exists():
        return set()
    with OUTPUT_PATH.open(newline="", encoding="utf-8") as file:
        return {row["image_id"] for row in csv.DictReader(file) if row.get("image_id")}


def annotate_image(client: OpenAI, prompt: str, image_path: Path) -> tuple[str, str | None]:
    encoded_image = resize_and_encode_image(image_path)
    response = client.chat.completions.create(
        model=API_MODEL_ID,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
        ]}],
        temperature=0,
    )
    return response.choices[0].message.content or "", response.choices[0].finish_reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.limit != 100:
        raise ValueError("This batch is intentionally fixed at 100 images.")
    api_key = os.getenv("INNKUBE_API_KEY")
    if not api_key:
        raise EnvironmentError('INNKUBE_API_KEY is not set. In PowerShell run: $env:INNKUBE_API_KEY = "<PASTE_KEY_HERE>"')

    existing_ids = load_existing_image_ids()
    candidates = select_candidates(existing_ids, args.limit)
    candidate_ids = {path.name for path in candidates}
    if candidate_ids & existing_ids:
        raise RuntimeError("Candidate selection overlaps with the existing annotation IDs.")

    completed_ids = load_completed_ids()
    unexpected_completed = completed_ids - candidate_ids
    if unexpected_completed:
        raise RuntimeError("The output file contains IDs outside this deterministic batch.")

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=TIMEOUT_SECONDS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_PATH.exists()
    fields = ["image_id", "image_path", "visible_ingredients", "model_name", "annotation_batch", "status", "finish_reason", "elapsed_seconds"]

    with OUTPUT_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for index, image_path in enumerate(candidates, start=1):
            if image_path.name in completed_ids:
                print(f"[{index}/100] Already complete: {image_path.name}")
                continue
            print(f"[{index}/100] Annotating: {image_path.name}")
            started = time.monotonic()
            try:
                response_text, finish_reason = annotate_image(client, prompt, image_path)
                parsed = parse_response(response_text)
                writer.writerow({
                    "image_id": image_path.name,
                    "image_path": image_path.relative_to(REPO_ROOT).as_posix(),
                    "visible_ingredients": visible_ingredients(parsed, response_text),
                    "model_name": MODEL_NAME,
                    "annotation_batch": ANNOTATION_BATCH,
                    "status": "success",
                    "finish_reason": finish_reason,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                })
                file.flush()
                completed_ids.add(image_path.name)
            except Exception as error:
                print(f"[{index}/100] Failed: {image_path.name}: {error}")
            time.sleep(SLEEP_SECONDS)

    if len(completed_ids) != 100:
        raise RuntimeError(f"Completed {len(completed_ids)} of 100 annotations. Re-run this command to resume.")
    print(f"Saved 100 annotations to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
