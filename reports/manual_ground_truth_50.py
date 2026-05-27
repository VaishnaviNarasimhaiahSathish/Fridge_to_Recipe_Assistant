import csv
from pathlib import Path


INPUT_CSV = Path("reports/vlm_eval_subset.csv")
OUTPUT_CSV = Path("reports/manual_ground_truth_50.csv")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    rows_out = []

    with open(INPUT_CSV, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows_out.append({
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "visible_ingredients": "",
                "uncertain_items": "",
                "notes": "",
                "annotator": ""
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "image_id",
            "image_path",
            "visible_ingredients",
            "uncertain_items",
            "notes",
            "annotator",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Created manual ground truth template: {OUTPUT_CSV}")
    print(f"Total rows: {len(rows_out)}")


if __name__ == "__main__":
    main()