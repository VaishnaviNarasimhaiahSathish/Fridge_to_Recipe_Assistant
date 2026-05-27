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


BASE_URL = "https://llms.innkube.fim.uni-passau.de"
MODEL_NAME = "qwen35-397b"

PROMPT_PATH = Path("configs/vlm_prompt_with_counts.txt")
SUBSET_PATH = Path("reports/vlm_eval_subset.csv")

# Final 50-image structured VLM output file
OUTPUT_PATH = Path("reports/vlm_predictions_v1.jsonl")

NUM_IMAGES = 50
SLEEP_SECONDS = 1
MAX_IMAGE_SIZE = 512
TIMEOUT_SECONDS = 180.0


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


def load_subset_rows(subset_path: Path, limit: int):
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset CSV not found: {subset_path}")

    with open(subset_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    return rows[:limit]


def resize_and_encode_image(image_path: Path, max_size: int = MAX_IMAGE_SIZE) -> tuple[str, str]:

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        original_width, original_height = image.size
        longest_side = max(original_width, original_height)

        if longest_side > max_size:
            scale = max_size / longest_side
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            image = image.resize((new_width, new_height))

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)

    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime_type = "image/jpeg"

    return encoded_image, mime_type


def try_parse_json(response_text):

    if response_text is None:
        return None

    if not isinstance(response_text, str):
        return None

    response_text = response_text.strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        response_text,
        re.DOTALL,
    )
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    object_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if object_match:
        try:
            return json.loads(object_match.group(0).strip())
        except json.JSONDecodeError:
            pass

    return None


def call_vlm(client: OpenAI, prompt: str, image_path: Path):
    image_base64, mime_type = resize_and_encode_image(image_path)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        temperature=0,
    )

    message = response.choices[0].message
    finish_reason = response.choices[0].finish_reason
    response_text = message.content

    return response_text, finish_reason


def load_completed_image_ids(output_path: Path):
    
    completed_image_ids = set()

    if not output_path.exists():
        return completed_image_ids

    with open(output_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if row.get("status") == "success" and row.get("image_id"):
                completed_image_ids.add(row["image_id"])

    return completed_image_ids


def main():
    api_key = os.getenv("INNKUBE_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "INNKUBE_API_KEY is not set. Run:\n"
            'export INNKUBE_API_KEY="your_key_here"'
        )

    prompt = load_prompt(PROMPT_PATH)
    rows = load_subset_rows(SUBSET_PATH, NUM_IMAGES)

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=TIMEOUT_SECONDS,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    completed_image_ids = load_completed_image_ids(OUTPUT_PATH)
    print(f"Already completed images: {len(completed_image_ids)}")
    print(f"Total target images: {len(rows)}")

    # append mode is important for resume logic
    with open(OUTPUT_PATH, "a", encoding="utf-8") as output_file:
        for index, row in enumerate(rows, start=1):
            image_id = row["image_id"]

            if image_id in completed_image_ids:
                print(f"[{index}/{len(rows)}] Skipping already completed: {image_id}")
                continue

            image_path = Path(row["image_path"])

            print(f"[{index}/{len(rows)}] Processing {image_id}")

            if not image_path.exists():
                result = {
                    "image_id": image_id,
                    "split": row.get("split"),
                    "image_path": row.get("image_path"),
                    "model": MODEL_NAME,
                    "prompt_file": str(PROMPT_PATH),
                    "status": "error",
                    "error": f"Image file not found: {image_path}",
                    "elapsed_seconds": None,
                    "finish_reason": None,
                    "raw_response": None,
                    "parsed_response": None,
                }

                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()
                continue

            start_time = time.time()

            try:
                response_text, finish_reason = call_vlm(client, prompt, image_path)
                elapsed_seconds = round(time.time() - start_time, 2)
                parsed_response = try_parse_json(response_text)

                result = {
                    "image_id": image_id,
                    "split": row.get("split"),
                    "image_path": row.get("image_path"),
                    "model": MODEL_NAME,
                    "prompt_file": str(PROMPT_PATH),
                    "status": "success",
                    "error": None,
                    "elapsed_seconds": elapsed_seconds,
                    "finish_reason": finish_reason,
                    "raw_response": response_text,
                    "parsed_response": parsed_response,
                }

                print(f"Finished in {elapsed_seconds} seconds")

            except Exception as error:
                elapsed_seconds = round(time.time() - start_time, 2)

                result = {
                    "image_id": image_id,
                    "split": row.get("split"),
                    "image_path": row.get("image_path"),
                    "model": MODEL_NAME,
                    "prompt_file": str(PROMPT_PATH),
                    "status": "error",
                    "error": str(error),
                    "elapsed_seconds": elapsed_seconds,
                    "finish_reason": None,
                    "raw_response": None,
                    "parsed_response": None,
                }

                print(f"Error after {elapsed_seconds} seconds: {error}")

            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()

            if result["status"] == "success":
                completed_image_ids.add(image_id)

            time.sleep(SLEEP_SECONDS)

    print(f"\nSaved VLM outputs to: {OUTPUT_PATH}")
    print(f"Completed images after this run: {len(load_completed_image_ids(OUTPUT_PATH))}")


if __name__ == "__main__":
    main()
    