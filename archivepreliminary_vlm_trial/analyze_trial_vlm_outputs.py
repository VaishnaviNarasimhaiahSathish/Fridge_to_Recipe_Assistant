import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


INPUT_PATH = Path("reports/vlm_predictions_raw.jsonl")
DATA_YAML_PATH = Path("data/raw/data.yaml")

OUTPUT_FLAT_CSV = Path("reports/vlm_predictions_flat.csv")
OUTPUT_INGREDIENT_FREQ_CSV = Path("reports/vlm_ingredient_frequencies.csv")
OUTPUT_UNCERTAIN_FREQ_CSV = Path("reports/vlm_uncertain_frequencies.csv")
OUTPUT_COMBINED_FREQ_CSV = Path("reports/vlm_item_frequencies_combined.csv")
OUTPUT_ANALYSIS_MD = Path("reports/vlm_output_analysis.md")

FIGURES_DIR = Path("reports/figures")
TOP_INGREDIENTS_FIG = FIGURES_DIR / "vlm_top_ingredients.png"
TOP_UNCERTAIN_FIG = FIGURES_DIR / "vlm_top_uncertain_items.png"
PREDICTIONS_PER_IMAGE_FIG = FIGURES_DIR / "vlm_predictions_per_image.png"
DATASET_VS_OPEN_VOCAB_FIG = FIGURES_DIR / "vlm_dataset_vs_open_vocab.png"

TOP_N = 20


def load_dataset_classes(data_yaml_path: Path):
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return set(clean_item(name) for name in data.get("names", []))


def clean_item(item: str) -> str:
    item = item.strip().lower()
    item = item.replace("_", " ")
    item = re.sub(r"\s+", " ", item)

    # Simple plural normalization for common cases
    plural_map = {
        "tomatoes": "tomato",
        "bell peppers": "bell pepper",
        "carrots": "carrot",
        "eggs": "eggs",
        "strawberry": "strawberries",
        "blueberry": "blueberries",
        "berries": "berries",
    }

    return plural_map.get(item, item)


def extract_json_from_text(text: str):
    """
    Handles:
    1. Plain JSON
    2. JSON wrapped in ```json ... ```
    3. JSON with extra text before/after
    """

    if text is None:
        return None

    text = text.strip()

    # Case 1: direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Case 2: markdown fenced JSON
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Case 3: extract first JSON-like object
    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        candidate = object_match.group(0).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def load_jsonl(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"VLM output file not found: {input_path}")

    rows = []

    with open(input_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"Skipping invalid JSONL line {line_number}: {error}")

    return rows


def get_response_dict(row):
    parsed_response = row.get("parsed_response")

    if isinstance(parsed_response, dict):
        return parsed_response, "parsed_response"

    raw_response = row.get("raw_response")
    extracted_response = extract_json_from_text(raw_response)

    if isinstance(extracted_response, dict):
        return extracted_response, "extracted_from_raw_response"

    return None, "parse_failed"


def list_from_response(response_dict, key):
    if not isinstance(response_dict, dict):
        return []

    value = response_dict.get(key, [])

    if isinstance(value, list):
        return [clean_item(str(item)) for item in value if str(item).strip()]

    return []


def split_ground_truth(ground_truth_text: str):
    if not ground_truth_text:
        return []

    return [clean_item(item) for item in ground_truth_text.split(";") if item.strip()]


def write_flat_csv(flat_rows):
    OUTPUT_FLAT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "split",
        "status",
        "parse_status",
        "model",
        "ground_truth_class_names",
        "ingredients",
        "uncertain",
        "num_ground_truth_classes",
        "num_predicted_ingredients",
        "num_uncertain_items",
        "num_in_dataset_predictions",
        "num_open_vocab_predictions",
    ]

    with open(OUTPUT_FLAT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def write_frequency_csv(path: Path, counter: Counter, item_type: str, dataset_classes):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["item", "type", "count", "in_dataset_22_classes"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item, count in counter.most_common():
            writer.writerow({
                "item": item,
                "type": item_type,
                "count": count,
                "in_dataset_22_classes": item in dataset_classes,
            })


def write_combined_frequency_csv(ingredient_counter, uncertain_counter, dataset_classes):
    OUTPUT_COMBINED_FREQ_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_COMBINED_FREQ_CSV, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["item", "type", "count", "in_dataset_22_classes"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item, count in ingredient_counter.most_common():
            writer.writerow({
                "item": item,
                "type": "ingredient",
                "count": count,
                "in_dataset_22_classes": item in dataset_classes,
            })

        for item, count in uncertain_counter.most_common():
            writer.writerow({
                "item": item,
                "type": "uncertain",
                "count": count,
                "in_dataset_22_classes": item in dataset_classes,
            })


def plot_top_counter(counter: Counter, title: str, output_path: Path, xlabel: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_items = counter.most_common(TOP_N)

    if not top_items:
        print(f"No data available for {title}")
        return

    labels = [item for item, _ in reversed(top_items)]
    values = [count for _, count in reversed(top_items)]

    plt.figure(figsize=(10, 7))
    plt.barh(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure: {output_path}")


def plot_predictions_per_image(flat_rows):
    counts = [int(row["num_predicted_ingredients"]) for row in flat_rows]

    if not counts:
        return

    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=range(0, max(counts) + 2))
    plt.title("Number of VLM Ingredient Predictions per Image")
    plt.xlabel("Predicted ingredients per image")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(PREDICTIONS_PER_IMAGE_FIG, dpi=200)
    plt.close()

    print(f"Saved figure: {PREDICTIONS_PER_IMAGE_FIG}")


def plot_dataset_vs_open_vocab(total_in_dataset, total_open_vocab):
    labels = ["In 22 annotated classes", "Open-vocabulary / outside 22 classes"]
    values = [total_in_dataset, total_open_vocab]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values)
    plt.title("VLM Predictions: Dataset Classes vs Open Vocabulary")
    plt.ylabel("Number of predicted ingredient mentions")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(DATASET_VS_OPEN_VOCAB_FIG, dpi=200)
    plt.close()

    print(f"Saved figure: {DATASET_VS_OPEN_VOCAB_FIG}")


def write_markdown_summary(
    total_images,
    success_count,
    parse_success_count,
    parse_failed_count,
    ingredient_counter,
    uncertain_counter,
    flat_rows,
    total_in_dataset,
    total_open_vocab,
):
    total_predictions = sum(ingredient_counter.values())
    total_uncertain = sum(uncertain_counter.values())

    prediction_counts = [
        int(row["num_predicted_ingredients"])
        for row in flat_rows
    ]

    avg_predictions = (
        sum(prediction_counts) / len(prediction_counts)
        if prediction_counts else 0
    )

    top_ingredients = ingredient_counter.most_common(15)
    top_uncertain = uncertain_counter.most_common(15)

    lines = []

    lines.append("# Trial VLM Output Analysis\n")

    lines.append("## Goal\n")
    lines.append(
        "This analysis summarizes the first open-vocabulary VLM ingredient extraction run "
        "on the 50-image evaluation subset."
    )

    lines.append("\n## Run Summary\n")
    lines.append(f"- Total images in input file: {total_images}")
    lines.append(f"- Successful VLM calls: {success_count}")
    lines.append(f"- Responses parsed successfully: {parse_success_count}")
    lines.append(f"- Responses needing parser improvement: {parse_failed_count}")
    lines.append(f"- Total predicted ingredient mentions: {total_predictions}")
    lines.append(f"- Total uncertain item mentions: {total_uncertain}")
    lines.append(f"- Average predicted ingredients per image: {avg_predictions:.2f}")

    lines.append("\n## Dataset Classes vs Open-Vocabulary Predictions\n")
    lines.append(
        "The dataset contains annotations for 22 selected ingredient classes. "
        "However, the VLM can identify additional visible fridge items outside this label space."
    )
    lines.append(f"- Predictions matching the 22 annotated classes: {total_in_dataset}")
    lines.append(f"- Open-vocabulary predictions outside the 22 classes: {total_open_vocab}")

    lines.append("\n## Most Frequent Predicted Ingredients\n")
    lines.append("| Ingredient | Count |")
    lines.append("|---|---:|")
    for item, count in top_ingredients:
        lines.append(f"| {item} | {count} |")

    lines.append("\n## Most Frequent Uncertain Items\n")
    lines.append("| Item | Count |")
    lines.append("|---|---:|")
    for item, count in top_uncertain:
        lines.append(f"| {item} | {count} |")

    lines.append("\n## Observed Issues\n")
    lines.append("- Some responses are valid JSON but wrapped inside markdown code blocks.")
    lines.append("- The VLM sometimes produces many ingredients per image, suggesting that the prompt may be too broad.")
    lines.append("- Some outputs contain naming variations, for example singular/plural forms.")
    lines.append("- Some outputs are broad or ambiguous, such as `berries`, `sauce`, `meat`, or `fruit`.")
    lines.append(
        "- Many outputs are outside the 22 annotated classes. These should not automatically be treated as hallucinations "
        "because the dataset labels are not exhaustive."
    )

    lines.append("\n## Interpretation\n")
    lines.append(
        "The first trial confirms that the InnKube VLM endpoint can process fridge images and return structured ingredient lists. "
        "The results also support the open-vocabulary direction of the project, because the model identifies items beyond the dataset's 22 annotated classes. "
        "However, the output requires better parsing, normalization, and prompt refinement before final evaluation."
    )

    lines.append("\n## Recommended Next Improvements\n")
    lines.append("1. Improve JSON parsing to handle markdown-wrapped JSON.")
    lines.append("2. Refine the prompt to reduce guessing and move unclear packaged items to the uncertain list.")
    lines.append("3. Normalize ingredient names before recipe recommendation.")
    lines.append("4. Manually inspect a small sample of open-vocabulary predictions to separate visible unannotated items from hallucinations.")

    lines.append("\n## Generated Files\n")
    lines.append(f"- `{OUTPUT_FLAT_CSV}`")
    lines.append(f"- `{OUTPUT_INGREDIENT_FREQ_CSV}`")
    lines.append(f"- `{OUTPUT_UNCERTAIN_FREQ_CSV}`")
    lines.append(f"- `{OUTPUT_COMBINED_FREQ_CSV}`")
    lines.append(f"- `{TOP_INGREDIENTS_FIG}`")
    lines.append(f"- `{TOP_UNCERTAIN_FIG}`")
    lines.append(f"- `{PREDICTIONS_PER_IMAGE_FIG}`")
    lines.append(f"- `{DATASET_VS_OPEN_VOCAB_FIG}`")

    OUTPUT_ANALYSIS_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANALYSIS_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved analysis summary: {OUTPUT_ANALYSIS_MD}")


def main():
    dataset_classes = load_dataset_classes(DATA_YAML_PATH)
    raw_rows = load_jsonl(INPUT_PATH)

    flat_rows = []
    ingredient_counter = Counter()
    uncertain_counter = Counter()

    success_count = 0
    parse_success_count = 0
    parse_failed_count = 0

    total_in_dataset = 0
    total_open_vocab = 0

    for row in raw_rows:
        if row.get("status") == "success":
            success_count += 1

        response_dict, parse_status = get_response_dict(row)

        if response_dict is None:
            ingredients = []
            uncertain_items = []
            parse_failed_count += 1
        else:
            ingredients = list_from_response(response_dict, "ingredients")
            uncertain_items = list_from_response(response_dict, "uncertain")
            parse_success_count += 1

        ground_truth_items = split_ground_truth(row.get("ground_truth_class_names", ""))

        ingredient_counter.update(ingredients)
        uncertain_counter.update(uncertain_items)

        in_dataset_predictions = [
            item for item in ingredients
            if item in dataset_classes
        ]

        open_vocab_predictions = [
            item for item in ingredients
            if item not in dataset_classes
        ]

        total_in_dataset += len(in_dataset_predictions)
        total_open_vocab += len(open_vocab_predictions)

        flat_rows.append({
            "image_id": row.get("image_id", ""),
            "split": row.get("split", ""),
            "status": row.get("status", ""),
            "parse_status": parse_status,
            "model": row.get("model", ""),
            "ground_truth_class_names": ";".join(ground_truth_items),
            "ingredients": ";".join(ingredients),
            "uncertain": ";".join(uncertain_items),
            "num_ground_truth_classes": len(set(ground_truth_items)),
            "num_predicted_ingredients": len(ingredients),
            "num_uncertain_items": len(uncertain_items),
            "num_in_dataset_predictions": len(in_dataset_predictions),
            "num_open_vocab_predictions": len(open_vocab_predictions),
        })

    write_flat_csv(flat_rows)

    write_frequency_csv(
        OUTPUT_INGREDIENT_FREQ_CSV,
        ingredient_counter,
        "ingredient",
        dataset_classes,
    )

    write_frequency_csv(
        OUTPUT_UNCERTAIN_FREQ_CSV,
        uncertain_counter,
        "uncertain",
        dataset_classes,
    )

    write_combined_frequency_csv(
        ingredient_counter,
        uncertain_counter,
        dataset_classes,
    )

    plot_top_counter(
        ingredient_counter,
        "Top VLM-Predicted Ingredients",
        TOP_INGREDIENTS_FIG,
        "Frequency",
    )

    plot_top_counter(
        uncertain_counter,
        "Top Uncertain VLM Items",
        TOP_UNCERTAIN_FIG,
        "Frequency",
    )

    plot_predictions_per_image(flat_rows)

    plot_dataset_vs_open_vocab(total_in_dataset, total_open_vocab)

    write_markdown_summary(
        total_images=len(raw_rows),
        success_count=success_count,
        parse_success_count=parse_success_count,
        parse_failed_count=parse_failed_count,
        ingredient_counter=ingredient_counter,
        uncertain_counter=uncertain_counter,
        flat_rows=flat_rows,
        total_in_dataset=total_in_dataset,
        total_open_vocab=total_open_vocab,
    )

    print("\nAnalysis complete.")
    print(f"Flat CSV: {OUTPUT_FLAT_CSV}")
    print(f"Ingredient frequencies: {OUTPUT_INGREDIENT_FREQ_CSV}")
    print(f"Uncertain frequencies: {OUTPUT_UNCERTAIN_FREQ_CSV}")
    print(f"Combined frequencies: {OUTPUT_COMBINED_FREQ_CSV}")
    print(f"Markdown summary: {OUTPUT_ANALYSIS_MD}")


if __name__ == "__main__":
    main()