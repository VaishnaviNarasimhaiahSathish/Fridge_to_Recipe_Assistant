from pathlib import Path
import random
import csv
import yaml


DATASET_DIR = Path("data/raw")
OUTPUT_PATH = Path("reports/vlm_eval_subset.csv")

RANDOM_SEED = 42

# First VLM experiment subset
SPLIT_SAMPLE_SIZES = {
    "valid": 30,
    "test": 20,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_class_names(data_yaml_path: Path):
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    with open(data_yaml_path, "r") as file:
        data_config = yaml.safe_load(file)

    return data_config["names"]


def get_image_files(split: str):
    image_dir = DATASET_DIR / split / "images"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return sorted([
        file for file in image_dir.rglob("*")
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ])


def get_label_path(image_path: Path, split: str):
    label_dir = DATASET_DIR / split / "labels"
    return label_dir / f"{image_path.stem}.txt"


def read_label_file(label_path: Path, class_names):
    class_ids = []
    class_name_list = []

    if not label_path.exists():
        return class_ids, class_name_list, 0

    with open(label_path, "r") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            class_id = int(line.split()[0])
            class_ids.append(class_id)
            class_name_list.append(class_names[class_id])

    unique_class_ids = sorted(set(class_ids))
    unique_class_names = sorted(set(class_name_list))

    return unique_class_ids, unique_class_names, len(class_ids)


def main():
    random.seed(RANDOM_SEED)

    class_names = load_class_names(DATASET_DIR / "data.yaml")

    rows = []

    for split, sample_size in SPLIT_SAMPLE_SIZES.items():
        image_files = get_image_files(split)

        if len(image_files) < sample_size:
            raise ValueError(
                f"Requested {sample_size} images from {split}, "
                f"but only found {len(image_files)} images."
            )

        sampled_images = random.sample(image_files, sample_size)

        for image_path in sampled_images:
            label_path = get_label_path(image_path, split)

            ground_truth_class_ids, ground_truth_class_names, num_objects = read_label_file(
                label_path,
                class_names
            )

            rows.append({
                "image_id": image_path.name,
                "split": split,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "ground_truth_class_ids": ";".join(map(str, ground_truth_class_ids)),
                "ground_truth_class_names": ";".join(ground_truth_class_names),
                "num_objects": num_objects,
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="") as csvfile:
        fieldnames = [
            "image_id",
            "split",
            "image_path",
            "label_path",
            "ground_truth_class_ids",
            "ground_truth_class_names",
            "num_objects",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created VLM evaluation subset: {OUTPUT_PATH}")
    print(f"Total selected images: {len(rows)}")


if __name__ == "__main__":
    main()