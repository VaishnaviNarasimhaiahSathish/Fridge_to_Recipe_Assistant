import csv
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("reports/evaluation_100/vlm_per_image_evaluation.csv")
OUTPUT_DIR = Path("reports/evaluation_100")
BOOTSTRAP_OUTPUT = OUTPUT_DIR / "vlm_bootstrap_metrics.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "vlm_bootstrap_summary.md"

N_BOOTSTRAPS = 10000
RANDOM_SEED = 42


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(sample_df):
    total_tp = sample_df["tp"].sum()
    total_fp = sample_df["fp"].sum()
    total_fn = sample_df["fn"].sum()

    micro_precision = safe_divide(total_tp, total_tp + total_fp)
    micro_recall = safe_divide(total_tp, total_tp + total_fn)
    micro_f1 = safe_divide(
        2 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    )

    per_image_precision = []
    per_image_recall = []
    per_image_f1 = []

    for _, row in sample_df.iterrows():
        tp = row["tp"]
        fp = row["fp"]
        fn = row["fn"]

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        per_image_precision.append(precision)
        per_image_recall.append(recall)
        per_image_f1.append(f1)

    macro_precision = float(np.mean(per_image_precision))
    macro_recall = float(np.mean(per_image_recall))
    macro_f1 = float(np.mean(per_image_f1))

    mean_jaccard = float(sample_df["jaccard"].mean())
    exact_match_accuracy = float(sample_df["exact_match"].mean())

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "mean_jaccard": mean_jaccard,
        "exact_match_accuracy": exact_match_accuracy,
    }


def confidence_interval(values, lower=2.5, upper=97.5):
    return (
        float(np.percentile(values, lower)),
        float(np.percentile(values, upper)),
    )


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required_columns = ["tp", "fp", "fn", "jaccard", "exact_match"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    rng = np.random.default_rng(RANDOM_SEED)

    original_metrics = compute_metrics(df)

    bootstrap_rows = []

    n_images = len(df)

    for i in range(N_BOOTSTRAPS):
        sampled_indices = rng.integers(
            low=0,
            high=n_images,
            size=n_images,
        )

        sample_df = df.iloc[sampled_indices]
        metrics = compute_metrics(sample_df)
        metrics["bootstrap_id"] = i + 1

        bootstrap_rows.append(metrics)

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(BOOTSTRAP_OUTPUT, index=False)

    metric_names = [
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "mean_jaccard",
        "exact_match_accuracy",
    ]

    summary_rows = []

    for metric in metric_names:
        lower, upper = confidence_interval(bootstrap_df[metric])

        summary_rows.append(
            {
                "metric": metric,
                "original_value": original_metrics[metric],
                "bootstrap_mean": bootstrap_df[metric].mean(),
                "ci_lower_95": lower,
                "ci_upper_95": upper,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_markdown = "# Bootstrap Confidence Intervals for VLM Evaluation\n\n"

    summary_markdown += (
        "Image-level bootstrap resampling was applied to the 100-image evaluation set. "
        "In each bootstrap iteration, 100 images were sampled with replacement and evaluation "
        "metrics were recalculated. The 2.5th and 97.5th percentiles of the bootstrap "
        "distribution are reported as 95% confidence intervals.\n\n"
    )

    summary_markdown += "| Metric | Original Value | Bootstrap Mean | 95% CI Lower | 95% CI Upper |\n"
    summary_markdown += "|---|---:|---:|---:|---:|\n"

    for _, row in summary_df.iterrows():
        summary_markdown += (
            f"| {row['metric']} "
            f"| {row['original_value']:.4f} "
            f"| {row['bootstrap_mean']:.4f} "
            f"| {row['ci_lower_95']:.4f} "
            f"| {row['ci_upper_95']:.4f} |\n"
        )

    SUMMARY_OUTPUT.write_text(summary_markdown, encoding="utf-8")

    print("Bootstrap evaluation complete.")
    print(f"Bootstrap samples saved to: {BOOTSTRAP_OUTPUT}")
    print(f"Bootstrap summary saved to: {SUMMARY_OUTPUT}")
    print()
    print(summary_markdown)


if __name__ == "__main__":
    main()