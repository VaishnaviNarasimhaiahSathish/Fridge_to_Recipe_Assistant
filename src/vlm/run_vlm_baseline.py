import base64
import csv
import json
import os
import time
from pathlib import Path

from openai import OpenAI


BASE_URL = "https://llms.innkube.fim.uni-passau.de"
MODEL_NAME = "qwen35-397b"

PROMPT_PATH = Path("configs/vlm_prompt.txt")
SUBSET_PATH = Path("reports/vlm_eval_subset.csv")
OUTPUT_PATH = Path("reports/vlm_predictions_raw.jsonl")

SLEEP_SECONDS = 1


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


def load_subset_rows(subset_path: Path):
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset CSV not found: {subset_path}")

    with open(subset_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def encode_image_as_base64(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")


def get_image_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"

    raise ValueError(f"Unsupported image format: {suffix}")


def try_parse_json(response_text: str):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None


def call_vlm(client: OpenAI, prompt: str, image_path: Path):
    image_base64 = encode_image_as_base64(image_path)
    mime_type = get_image_mime_type(image_path)

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

    return response.choices[0].message.content


def main():
    api_key = os.getenv("INNKUBE_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "INNKUBE_API_KEY is not set. Run:\n"
            'export INNKUBE_API_KEY="your_key_here"'
        )

    prompt = load_prompt(PROMPT_PATH)
    rows = load_subset_rows(SUBSET_PATH)

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for index, row in enumerate(rows, start=1):
            image_path = Path(row["image_path"])

            print(f"[{index}/{len(rows)}] Processing {row['image_id']}")

            if not image_path.exists():
                result = {
                    "image_id": row["image_id"],
                    "split": row["split"],
                    "image_path": row["image_path"],
                    "label_path": row["label_path"],
                    "ground_truth_class_names": row["ground_truth_class_names"],
                    "num_objects": row["num_objects"],
                    "model": MODEL_NAME,
                    "status": "error",
                    "error": f"Image file not found: {image_path}",
                    "raw_response": None,
                    "parsed_response": None,
                }
                output_file.write(json.dumps(result) + "\n")
                continue

            try:
                response_text = call_vlm(client, prompt, image_path)
                parsed_response = try_parse_json(response_text)

                result = {
                    "image_id": row["image_id"],
                    "split": row["split"],
                    "image_path": row["image_path"],
                    "label_path": row["label_path"],
                    "ground_truth_class_names": row["ground_truth_class_names"],
                    "num_objects": row["num_objects"],
                    "model": MODEL_NAME,
                    "status": "success",
                    "error": None,
                    "raw_response": response_text,
                    "parsed_response": parsed_response,
                }

            except Exception as error:
                result = {
                    "image_id": row["image_id"],
                    "split": row["split"],
                    "image_path": row["image_path"],
                    "label_path": row["label_path"],
                    "ground_truth_class_names": row["ground_truth_class_names"],
                    "num_objects": row["num_objects"],
                    "model": MODEL_NAME,
                    "status": "error",
                    "error": str(error),
                    "raw_response": None,
                    "parsed_response": None,
                }

            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()

            time.sleep(SLEEP_SECONDS)

    print(f"\nSaved VLM predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()