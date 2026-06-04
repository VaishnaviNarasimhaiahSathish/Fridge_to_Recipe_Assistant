from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


EVALUATION_PATH = Path("reports/evaluation_100/vlm_per_image_evaluation.csv")
FALSE_POSITIVES_PATH = Path("reports/evaluation_100/vlm_false_positives.csv")
FALSE_NEGATIVES_PATH = Path("reports/evaluation_100/vlm_false_negatives.csv")


OUTPUT_DIR = Path("reports/evaluation_100/figures")

TOP_N_ITEMS = 20

def load_data():
    evaluation_df = pd.read_csv(EVALUATION_PATH)
    false_positive_df = pd.read_csv(FALSE_POSITIVES_PATH)
    false_negative_df = pd.read_csv(FALSE_NEGATIVES_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return evaluation_df, false_positive_df, false_negative_df


def plot_top_false_positives(false_positive_df):
    counts = (
        false_positive_df["false_positive"]
        .value_counts()
        .head(TOP_N_ITEMS)
        .sort_values()
    )

    plt.figure(figsize=(10, 7))
    plt.barh(counts.index, counts.values)
    plt.xlabel("Frequency")
    plt.ylabel("False Positive Ingredient")
    plt.title(f"Top {TOP_N_ITEMS} False Positive Ingredients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_false_positives_100.png", dpi=300)
    plt.close()


def plot_top_false_negatives(false_negative_df):
    counts = (
        false_negative_df["false_negative"]
        .value_counts()
        .head(TOP_N_ITEMS)
        .sort_values()
    )

    plt.figure(figsize=(10, 7))
    plt.barh(counts.index, counts.values)
    plt.xlabel("Frequency")
    plt.ylabel("False Negative Ingredient")
    plt.title(f"Top {TOP_N_ITEMS} False Negative Ingredients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_false_negatives_100.png", dpi=300)
    plt.close()

def plot_precision_vs_recall(evaluation_df):
    plt.figure(figsize=(8, 7))
    plt.scatter(
        evaluation_df["recall"],
        evaluation_df["precision"],
        s=60,
        alpha=0.75,
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs Recall per Image")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_vs_recall_100.png", dpi=300)
    plt.close()


def main():
    evaluation_df, false_positive_df, false_negative_df = load_data()

    plot_top_false_positives(false_positive_df)
    plot_top_false_negatives(false_negative_df)
    plot_precision_vs_recall(evaluation_df)

    print("Requested visualizations created.")
    print(f"Output folder: {OUTPUT_DIR}")
    print()
    print("Generated files:")
    print("- top_false_positives_100.png")
    print("- top_false_negatives_100.png")
    print("- precision_vs_recall_100.png")


if __name__ == "__main__":
    main()