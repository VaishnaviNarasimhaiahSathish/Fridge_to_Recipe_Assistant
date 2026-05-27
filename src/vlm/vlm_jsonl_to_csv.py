import csv
import json
import re
from pathlib import Path


INPUT_PATH = Path("reports/vlm_predictions_v1.jsonl")
OUTPUT_PATH = Path("reports/vlm_predictions_v1.csv")


def extract_json_from_text(text):
    if text is None or not isinstance(text, str):
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


def format_ingredients(parsed_response):
    if not isinstance(parsed_response, dict):
        return ""

    ingredients = parsed_response.get("ingredients", [])

    if not isinstance(ingredients, list):
        return ""

    formatted_items = []

    for item in ingredients:
        if isinstance(item, dict):
            name = item.get("name", "")
            quantity = item.get("quantity", "")
            unit = item.get("unit", "")
            confidence = item.get("confidence", "")
            evidence = item.get("visual_evidence", "")

            details = []

            if quantity:
                details.append(str(quantity))

            if unit:
                details.append(str(unit))

            if confidence:
                details.append(str(confidence))

            if details:
                formatted = f"{name} ({', '.join(details)})"
            else:
                formatted = name

            if evidence:
                formatted += f" - {evidence}"

            formatted_items.append(formatted)

        elif isinstance(item, str):
            formatted_items.append(item)

    return "; ".join(formatted_items)


def format_uncertain_items(parsed_response):
    if not isinstance(parsed_response, dict):
        return ""

    uncertain_items = parsed_response.get("uncertain_items", [])

    if not isinstance(uncertain_items, list):
        uncertain_items = parsed_response.get("uncertain", [])

    if not isinstance(uncertain_items, list):
        return ""

    formatted_items = []

    for item in uncertain_items:
        if isinstance(item, dict):
            name = item.get("name", "")
            reason = item.get("reason", "")

            if reason:
                formatted_items.append(f"{name}: {reason}")
            else:
                formatted_items.append(name)

        elif isinstance(item, str):
            formatted_items.append(item)

    return "; ".join(formatted_items)


def load_latest_rows(path):
    """
    Keeps only the latest row per image_id.
    This handles reruns where an image may first have an error row
    and later a successful row.
    """
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


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    rows = load_latest_rows(INPUT_PATH)

    output_rows = []

    for row in rows:
        parsed_response = get_parsed_response(row)

        output_rows.append(
            {
                "image_id": row.get("image_id", ""),
                "elapsed_seconds": row.get("elapsed_seconds", ""),
                "vlm_ingredients": format_ingredients(parsed_response),
                "vlm_uncertain_items": format_uncertain_items(parsed_response),
                "raw_response": row.get("raw_response", ""),
            }
        )

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "image_id",
            "elapsed_seconds",
            "vlm_ingredients",
            "vlm_uncertain_items",
            "raw_response",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Converted VLM JSONL to readable CSV: {OUTPUT_PATH}")
    print(f"Rows written: {len(output_rows)}")


if __name__ == "__main__":
    main()