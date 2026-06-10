# VLM Latency Analysis

## Input File

- Predictions: `reports/evaluation_50/inputs/vlm_predictions_v1.csv`

## Summary Statistics

| Metric | Value |
|---|---:|
| Images analysed | 50 |
| Min inference time | 0.7s |
| Max inference time | 292.9s |
| Mean inference time | 49.0s |
| Median inference time | 17.8s |
| Total inference time | 2449s (40.8 min) |
| Images under 10s | 12 |
| Images over 60s | 11 |

## Latency Distribution

| Range | Count |
|---|---:|
| Under 10s | 12 |
| 10 – 30s | 22 |
| 30 – 60s | 5 |
| 60 – 120s | 5 |
| Over 120s | 6 |

## Key Findings

- Mean inference time of 49.0s per image makes real-time use impractical.
- 11 images (22%) took over 60 seconds — likely caused by
  server load or large image sizes at the InnKube endpoint.
- 12 images (24%) completed under 10 seconds, showing
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
