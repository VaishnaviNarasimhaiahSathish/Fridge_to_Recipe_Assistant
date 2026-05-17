import base64
import csv
import json
import os
from pathlib import Path

from openai import OpenAI


BASE_URL = "https://llms.innkube.fim.uni-passau.de"
MODEL_NAME = "qwen35-397b"

PROMPT_PATH = Path("configs/vlm_prompt.txt")
SUBSET_PATH = Path("reports/vlm_eval_subset.csv")


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


def load_first_image_path(subset_path: Path) -> Path:
    if not subset_path.exists():
        raise FileNotFoundError(f"VLM subset CSV not found: {subset_path}")

    with open(subset_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        first_row = next(reader, None)

    if first_row is None:
        raise ValueError(f"No rows found in {subset_path}")

    image_path = Path(first_row["image_path"])

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print("Testing image:")
    print(f"Image ID: {first_row['image_id']}")
    print(f"Split: {first_row['split']}")
    print(f"Ground truth ingredients: {first_row['ground_truth_class_names']}")
    print(f"Image path: {image_path}")

    return image_path


def encode_image_as_base64(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return encoded_image


def get_image_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"

    raise ValueError(f"Unsupported image format: {suffix}")


def extract_json_from_response(response_text: str):
    """
    Tries to parse the VLM response as JSON.
    If parsing fails, returns None so we can inspect the raw response.
    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None


def main():
    api_key = os.getenv("INNKUBE_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "INNKUBE_API_KEY is not set. Run:\n"
            'export INNKUBE_API_KEY="your_key_here"'
        )

    prompt = load_prompt(PROMPT_PATH)
    image_path = load_first_image_path(SUBSET_PATH)

    image_base64 = encode_image_as_base64(image_path)
    mime_type = get_image_mime_type(image_path)

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

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

    response_text = response.choices[0].message.content

    print("\nRaw VLM response:")
    print(response_text)

    parsed_json = extract_json_from_response(response_text)

    if parsed_json is not None:
        print("\nParsed JSON response:")
        print(json.dumps(parsed_json, indent=2))
    else:
        print("\nCould not parse response as JSON. We may need to adjust the prompt.")


if __name__ == "__main__":
    main()
    