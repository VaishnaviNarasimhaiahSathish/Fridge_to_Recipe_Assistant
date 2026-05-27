import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PER_IMAGE_PATH = Path("reports/evaluation/vlm_per_image_evaluation.csv")
FALSE_POSITIVES_PATH = Path("reports/evaluation/vlm_false_positives.csv")
FALSE_NEGATIVES_PATH = Path("reports/evaluation/vlm_false_negatives.csv")

OUTPUT_DIR = Path("reports/evaluation/figures")


def save_bar_chart(labels, values, title, ylabel, output_path, rotate=False):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)

    if rotate:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_horizontal_bar_chart(labels, values, title, xlabel, output_path):
    plt.figure(figsize=(10, 7))
    plt.barh(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_overall_metrics(df):
    micro_tp = df["tp"].sum()
    micro_fp = df["fp"].sum()
    micro_fn = df["fn"].sum()

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0
    )

    mean_jaccard = df["jaccard"].mean()
    exact_match_accuracy = df["exact_match"].mean()

    labels = [
        "Precision",
        "Recall",
        "F1-score",
        "Mean Jaccard",
        "Exact Match",
    ]

    values = [
        micro_precision,
        micro_recall,
        micro_f1,
        mean_jaccard,
        exact_match_accuracy,
    ]

    save_bar_chart(
        labels,
        values,
        "Overall VLM Ingredient Extraction Metrics",
        "Score",
        OUTPUT_DIR / "overall_metrics.png",
        rotate=True,
    )


def plot_tp_fp_fn(df):
    labels = ["True Positives", "False Positives", "False Negatives"]
    values = [df["tp"].sum(), df["fp"].sum(), df["fn"].sum()]

    save_bar_chart(
        labels,
        values,
        "Total TP / FP / FN Counts",
        "Count",
        OUTPUT_DIR / "tp_fp_fn_counts.png",
        rotate=True,
    )


def plot_per_image_f1_bar(df):
    df_sorted = df.sort_values("f1", ascending=False).reset_index(drop=True)

    labels = [str(i + 1) for i in range(len(df_sorted))]
    values = df_sorted["f1"].tolist()

    save_bar_chart(
        labels,
        values,
        "Per-Image F1-score Sorted High to Low",
        "F1-score",
        OUTPUT_DIR / "per_image_f1_sorted.png",
        rotate=False,
    )


def plot_f1_histogram(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df["f1"], bins=10)
    plt.title("Distribution of Per-Image F1-scores")
    plt.xlabel("F1-score")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "f1_score_distribution.png", dpi=300)
    plt.close()


def plot_precision_recall_scatter(df):
    plt.figure(figsize=(8, 7))
    plt.scatter(df["recall"], df["precision"])

    for index, row in df.iterrows():
        plt.annotate(
            str(index + 1),
            (row["recall"], row["precision"]),
            fontsize=8,
            alpha=0.7,
        )

    plt.title("Per-Image Precision vs Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_vs_recall_scatter.png", dpi=300)
    plt.close()


def plot_runtime_distribution(df):
    if "elapsed_seconds" not in df.columns:
        return

    runtime = pd.to_numeric(df["elapsed_seconds"], errors="coerce").dropna()

    if runtime.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.hist(runtime, bins=10)
    plt.title("VLM Runtime Distribution")
    plt.xlabel("Elapsed seconds per image")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "runtime_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(runtime) + 1), runtime.tolist(), marker="o")
    plt.title("VLM Runtime per Image")
    plt.xlabel("Image index")
    plt.ylabel("Elapsed seconds")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "runtime_per_image.png", dpi=300)
    plt.close()


def load_error_counts(path, column_name):
    if not path.exists():
        return Counter()

    counter = Counter()

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            item = row.get(column_name, "").strip()

            if item:
                counter[item] += 1

    return counter


def plot_top_errors():
    fp_counts = load_error_counts(FALSE_POSITIVES_PATH, "false_positive")
    fn_counts = load_error_counts(FALSE_NEGATIVES_PATH, "false_negative")

    top_fp = fp_counts.most_common(15)
    top_fn = fn_counts.most_common(15)

    if top_fp:
        labels = [item for item, _ in top_fp]
        values = [count for _, count in top_fp]

        save_horizontal_bar_chart(
            labels,
            values,
            "Top 15 False Positive Ingredients",
            "Frequency",
            OUTPUT_DIR / "top_false_positives.png",
        )

    if top_fn:
        labels = [item for item, _ in top_fn]
        values = [count for _, count in top_fn]

        save_horizontal_bar_chart(
            labels,
            values,
            "Top 15 False Negative Ingredients",
            "Frequency",
            OUTPUT_DIR / "top_false_negatives.png",
        )


def create_visual_summary_markdown(df):
    micro_tp = df["tp"].sum()
    micro_fp = df["fp"].sum()
    micro_fn = df["fn"].sum()

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0
    )

    mean_jaccard = df["jaccard"].mean()
    exact_match_accuracy = df["exact_match"].mean()

    markdown = f"""# VLM Evaluation Visual Summary

## Final Metrics

| Metric | Value |
|---|---:|
| Precision | {micro_precision:.4f} |
| Recall | {micro_recall:.4f} |
| F1-score | {micro_f1:.4f} |
| Mean Jaccard Similarity | {mean_jaccard:.4f} |
| Exact Match Accuracy | {exact_match_accuracy:.4f} |

## Generated Figures

1. `overall_metrics.png`
2. `tp_fp_fn_counts.png`
3. `per_image_f1_sorted.png`
4. `f1_score_distribution.png`
5. `precision_vs_recall_scatter.png`
6. `top_false_positives.png`
7. `top_false_negatives.png`
8. `runtime_distribution.png`
9. `runtime_per_image.png`

## Interpretation

The VLM achieved higher recall than precision, meaning it detected many visible ingredients but also produced extra predictions. This behavior is typical for open-vocabulary fridge scenes where food items are cluttered, partially occluded, or visible through packaging.

Exact match accuracy is low because it requires the full predicted ingredient set to exactly match the ground truth for an image. Mean Jaccard similarity is more informative because it measures partial set overlap.
"""

    output_path = OUTPUT_DIR / "visual_evaluation_summary.md"
    output_path.write_text(markdown, encoding="utf-8")


def main():
    if not PER_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Per-image evaluation file not found: {PER_IMAGE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PER_IMAGE_PATH)

    plot_overall_metrics(df)
    plot_tp_fp_fn(df)
    plot_per_image_f1_bar(df)
    plot_f1_histogram(df)
    plot_precision_recall_scatter(df)
    plot_runtime_distribution(df)
    plot_top_errors()
    create_visual_summary_markdown(df)

    print("Visualizations created successfully.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()