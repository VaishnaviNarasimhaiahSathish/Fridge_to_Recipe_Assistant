"""Latency analysis for the Gemma 4 100-image annotation batch.

Uses elapsed_seconds from the raw annotation run (not the Qwen review pass)
since that column reflects actual VLM inference time per image.
"""

import csv
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

INPUT_CSV = Path("data/annotations/gemma4_batch_100/gemma4_annotations_raw.csv")

OUTPUT_DIR = Path("reports/gemma4_batch_100")
FIGURES_DIR = OUTPUT_DIR / "figures"
SUMMARY_OUTPUT = OUTPUT_DIR / "gemma4_latency_summary.md"


def count_ingredients(value: str) -> int:
    if not value:
        return 0
    return len([part.strip() for part in re.split(r"[;,]", value) if part.strip()])


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            elapsed = row.get("elapsed_seconds", "").strip()
            if not elapsed:
                continue
            try:
                elapsed_seconds = float(elapsed)
            except ValueError:
                continue
            rows.append({
                "image_id": row["image_id"],
                "elapsed_seconds": elapsed_seconds,
                "ingredient_count": count_ingredients(row.get("visible_ingredients", "")),
            })
    return rows


def plot_latency_per_image(rows: list[dict]) -> None:
    ids = [r["image_id"][:20] for r in rows]
    times = [r["elapsed_seconds"] for r in rows]
    mean_time = statistics.mean(times)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(times)), times, width=0.8)
    ax.axhline(mean_time, linestyle="--", linewidth=1.2, label=f"Mean = {mean_time:.1f}s")
    ax.set_xlabel("Image index")
    ax.set_ylabel("Elapsed seconds")
    ax.set_title("Gemma 4 inference time per image")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gemma4_latency_per_image.png", dpi=300)
    plt.close()


def plot_latency_vs_ingredient_count(rows: list[dict]) -> None:
    counts = [r["ingredient_count"] for r in rows]
    times = [r["elapsed_seconds"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(counts, times, alpha=0.7)
    ax.set_xlabel("Ingredients detected")
    ax.set_ylabel("Elapsed seconds")
    ax.set_title("Gemma 4 latency vs ingredient count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gemma4_latency_vs_ingredient_count.png", dpi=300)
    plt.close()


def ingredient_count_buckets(rows: list[dict]) -> list[tuple[str, int, float]]:
    buckets = [
        ("0-3", lambda c: c <= 3),
        ("4-6", lambda c: 4 <= c <= 6),
        ("7-9", lambda c: 7 <= c <= 9),
        ("10+", lambda c: c >= 10),
    ]
    results = []
    for label, predicate in buckets:
        matching = [r["elapsed_seconds"] for r in rows if predicate(r["ingredient_count"])]
        avg = statistics.mean(matching) if matching else 0.0
        results.append((label, len(matching), avg))
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows(INPUT_CSV)
    times = [r["elapsed_seconds"] for r in rows]

    sorted_by_time = sorted(rows, key=lambda r: r["elapsed_seconds"])
    fastest5 = sorted_by_time[:5]
    slowest5 = sorted_by_time[-5:][::-1]

    plot_latency_per_image(rows)
    plot_latency_vs_ingredient_count(rows)

    buckets = ingredient_count_buckets(rows)

    def fmt_image_rows(image_rows):
        return "\n".join(
            f"| {r['image_id']} | {r['elapsed_seconds']:.2f}s | {r['ingredient_count']} |"
            for r in image_rows
        )

    summary = f"""# Gemma 4 Latency Analysis

## Input File

- `{INPUT_CSV}`

## Summary Statistics

| Metric | Value |
|---|---:|
| Images analysed | {len(times)} |
| Min inference time | {min(times):.2f}s |
| Max inference time | {max(times):.2f}s |
| Mean inference time | {statistics.mean(times):.2f}s |
| Median inference time | {statistics.median(times):.2f}s |
| Stdev | {statistics.stdev(times):.2f}s |
| Total inference time | {sum(times):.0f}s ({sum(times)/60:.1f} min) |

## Slowest 5 Images

| Image | Elapsed | Ingredients detected |
|---|---:|---:|
{fmt_image_rows(slowest5)}

## Fastest 5 Images

| Image | Elapsed | Ingredients detected |
|---|---:|---:|
{fmt_image_rows(fastest5)}

## Latency by Ingredient Count

| Ingredient count | Images | Avg latency |
|---|---:|---:|
{chr(10).join(f"| {label} | {count} | {avg:.2f}s |" for label, count, avg in buckets)}

## Figures

![Latency per image](figures/gemma4_latency_per_image.png)
![Latency vs ingredient count](figures/gemma4_latency_vs_ingredient_count.png)

## Notes

- Gemma 4 was queried with images resized to 512px on the longest side (see `src/vlm/run_gemma4_annotation_batch.py`).
- Compare against the Qwen baseline latency in `reports/latency_analysis/vlm_latency_summary.md` (50-image subset) for a cross-model latency reference, keeping in mind the two batches use different images and were run at different times under different server load.
"""

    SUMMARY_OUTPUT.write_text(summary, encoding="utf-8")

    print("Gemma 4 latency analysis complete.")
    print(f"Summary report: {SUMMARY_OUTPUT}")
    print(f"Mean: {statistics.mean(times):.2f}s  Median: {statistics.median(times):.2f}s")


if __name__ == "__main__":
    main()
