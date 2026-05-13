from pathlib import Path
from collections import Counter
import yaml


DATASET_DIR = Path("data/raw")


def count_files(folder: Path, extensions):
    if not folder.exists():
        return 0

    return len([
        file for file in folder.rglob("*")
        if file.suffix.lower() in extensions
    ])


def count_class_annotations(label_dir: Path):
    class_counter = Counter()

    if not label_dir.exists():
        return class_counter

    for label_file in label_dir.rglob("*.txt"):
        with open(label_file, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                class_id = int(line.split()[0])
                class_counter[class_id] += 1

    return class_counter


def main():
    data_yaml_path = DATASET_DIR / "data.yaml"

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    with open(data_yaml_path, "r") as file:
        data_config = yaml.safe_load(file)

    class_names = data_config.get("names", [])
    splits = ["train", "valid", "test"]

    total_class_counter = Counter()

    report_lines = []
    report_lines.append("# Dataset Audit Report\n")
    report_lines.append("## Split Summary\n")

    for split in splits:
        image_dir = DATASET_DIR / split / "images"
        label_dir = DATASET_DIR / split / "labels"

        num_images = count_files(image_dir, {".jpg", ".jpeg", ".png"})
        num_labels = count_files(label_dir, {".txt"})
        split_class_counter = count_class_annotations(label_dir)

        total_class_counter.update(split_class_counter)

        report_lines.append(f"### {split}")
        report_lines.append(f"- Images: {num_images}")
        report_lines.append(f"- Label files: {num_labels}")
        report_lines.append(f"- Total object annotations: {sum(split_class_counter.values())}\n")

    report_lines.append("## Classes\n")
    report_lines.append(f"Number of classes: {len(class_names)}\n")

    for idx, class_name in enumerate(class_names):
        report_lines.append(f"- {idx}: {class_name}")

    report_lines.append("\n## Class Distribution\n")
    report_lines.append("| Class ID | Class Name | Annotation Count |")
    report_lines.append("|---:|---|---:|")

    for class_id, class_name in enumerate(class_names):
        count = total_class_counter.get(class_id, 0)
        report_lines.append(f"| {class_id} | {class_name} | {count} |")

    output_path = Path("reports/dataset_audit_report.md")

    with open(output_path, "w") as file:
        file.write("\n".join(report_lines))

    print(f"Dataset audit saved to {output_path}")


if __name__ == "__main__":
    main()