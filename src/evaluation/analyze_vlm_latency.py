import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


PREDICTIONS_CSV = Path("reports/evaluation_50/inputs/vlm_predictions_v1.csv")

OUTPUT_DIR  = Path("reports/latency_analysis")
FIGURES_DIR = OUTPUT_DIR / "figures"


def load_times(path: Path) -> list[tuple[str, float]]:
    results = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("elapsed_seconds", "").strip()
            if t:
                try:
                    results.append((row["image_id"], float(t)))
                except ValueError:
                    continue
    return results


def print_summary(times: list[float]):
    print(f"Images analysed : {len(times)}")
    print(f"Min             : {min(times):.1f}s")
    print(f"Max             : {max(times):.1f}s")
    print(f"Mean            : {statistics.mean(times):.1f}s")
    print(f"Median          : {statistics.median(times):.1f}s")
    print(f"Stdev           : {statistics.stdev(times):.1f}s")
    print()

    buckets = [
        ("under 10s",   lambda t: t < 10),
        ("10 – 30s",    lambda t: 10 <= t < 30),
        ("30 – 60s",    lambda t: 30 <= t < 60),
        ("60 – 120s",   lambda t: 60 <= t < 120),
        ("over 120s",   lambda t: t >= 120),
    ]
    print("Latency distribution:")
    for label, fn in buckets:
        count = sum(1 for t in times if fn(t))
        print(f"  {label:12s}: {count:3d} images")


def plot_runtime_per_image(data: list[tuple[str, float]]):
    ids   = [d[0][:20] for d in data]
    times = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(times)), times, width=0.8)
    ax.axhline(statistics.mean(times), linestyle="--", linewidth=1.2,
               label=f"Mean = {statistics.mean(times):.1f}s")
    ax.set_xlabel("Image index")
    ax.set_ylabel("Elapsed seconds")
    ax.set_title("VLM inference time per image")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "vlm_runtime_per_image.png", dpi=300)
    plt.close()
    print("Saved: reports/latency_analysis/figures/vlm_runtime_per_image.png")


def plot_runtime_distribution(times: list[float]):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(times, bins=20, edgecolor="white", linewidth=0.6)
    ax.axvline(statistics.mean(times), linestyle="--", linewidth=1.2,
               label=f"Mean = {statistics.mean(times):.1f}s")
    ax.axvline(statistics.median(times), linestyle="--", linewidth=1.2,
               label=f"Median = {statistics.median(times):.1f}s")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Number of images")
    ax.set_title("VLM inference time distribution")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "vlm_runtime_distribution.png", dpi=300)
    plt.close()
    print("Saved: reports/latency_analysis/figures/vlm_runtime_distribution.png")


def write_summary_md(times: list[float]):
    total_time = sum(times)
    mean_time  = statistics.mean(times)
    median_time = statistics.median(times)

    slow_count = sum(1 for t in times if t > 60)
    fast_count = sum(1 for t in times if t < 10)

    summary = f"""# VLM Latency Analysis

## Input File

- Predictions: `reports/evaluation_50/inputs/vlm_predictions_v1.csv`

## Summary Statistics

| Metric | Value |
|---|---:|
| Images analysed | {len(times)} |
| Min inference time | {min(times):.1f}s |
| Max inference time | {max(times):.1f}s |
| Mean inference time | {mean_time:.1f}s |
| Median inference time | {median_time:.1f}s |
| Total inference time | {total_time:.0f}s ({total_time/60:.1f} min) |
| Images under 10s | {fast_count} |
| Images over 60s | {slow_count} |

## Latency Distribution

| Range | Count |
|---|---:|
| Under 10s | {sum(1 for t in times if t < 10)} |
| 10 – 30s | {sum(1 for t in times if 10 <= t < 30)} |
| 30 – 60s | {sum(1 for t in times if 30 <= t < 60)} |
| 60 – 120s | {sum(1 for t in times if 60 <= t < 120)} |
| Over 120s | {sum(1 for t in times if t >= 120)} |

## Key Findings

- Mean inference time of {mean_time:.1f}s per image makes real-time use impractical.
- {slow_count} images ({slow_count/len(times)*100:.0f}%) took over 60 seconds — likely caused by
  server load or large image sizes at the InnKube endpoint.
- {fast_count} images ({fast_count/len(times)*100:.0f}%) completed under 10 seconds, showing
  the endpoint is capable of fast responses under low load.

## Recommended Strategies

### 1. Prediction caching (implemented)
Cache VLM predictions by image file hash. If the same image is queried again,
return the cached result instantly without hitting the API.
See: `src/vlm/vlm_prediction_cache.py`

### 2. Image resizing before inference
Resize images to a maximum of 512px on the longest side before sending to
the VLM. Smaller images reduce token count and inference time.
Add to `src/vlm/run_vlm_baseline.py` before encoding:
```python
from PIL import Image
img = Image.open(image_path)
img.thumbnail((512, 512))
img.save(resized_path)
```

### 3. Prompt length reduction
The current structured prompt (`configs/vlm_prompt_with_counts.txt`) is long.
A shorter prompt with fewer output fields reduces generation time.
Consider removing the `visual_evidence` field for speed-optimised runs.

### 4. Batch offline processing
Run VLM inference overnight for all images, store results in cache.
The Streamlit UI then reads from cache with zero latency.

## Output Files

| File | Description |
|---|---|
| `reports/latency_analysis/vlm_latency_summary.md` | This report |
| `reports/latency_analysis/figures/vlm_runtime_per_image.png` | Runtime per image bar chart |
| `reports/latency_analysis/figures/vlm_runtime_distribution.png` | Runtime distribution histogram |
"""
    out_path = OUTPUT_DIR / "vlm_latency_summary.md"
    out_path.write_text(summary, encoding="utf-8")
    print(f"Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data  = load_times(PREDICTIONS_CSV)
    times = [t for _, t in data]

    print_summary(times)
    plot_runtime_per_image(data)
    plot_runtime_distribution(times)
    write_summary_md(times)

    print()
    print("Latency analysis complete.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()