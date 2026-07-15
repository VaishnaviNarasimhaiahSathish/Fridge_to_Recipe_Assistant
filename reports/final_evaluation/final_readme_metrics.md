# Final README Metrics Block

Use this block to update the README after reviewing the final outputs.

## Final Ingredient Extraction Evaluation

The final 100-image evaluation was computed against manually reviewed fridge-image ground truth using the updated normalization rules.

| Strategy | Precision | Recall | Micro F1 | Mean Jaccard | Exact Match |
|---|---:|---:|---:|---:|---:|
| High + medium predictions | 0.4899 | 0.6061 | 0.5418 | 0.3960 | 0.0200 |
| High confidence only app setting | 0.5547 | 0.5547 | 0.5547 | 0.4072 | 0.0300 |

The deployed app uses the **high confidence only** setting to reduce unreliable ingredient guesses before recipe recommendation.

## Error Analysis

False positives in the high-confidence setting were grouped into practical categories:

- Common fridge default: 123 (36%)
- Ambiguous visual: 116 (34%)
- Other: 98 (29%)
- Context guess: 1 (0%)


## Gemma Comparison

Gemma was evaluated against the Qwen-reviewed correction set as a proxy reference.

| Metric | Value |
|---|---:|
| Images | 100 |
| Precision | 0.8211 |
| Recall | 0.2572 |
| Micro F1 | 0.3917 |
| Mean Jaccard | 0.2512 |
| Avg Gemma ingredients/image | 3.69 |
| Avg Qwen-reference ingredients/image | 11.78 |


## Latency Summary

| Model | Images | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Qwen | 100 | 50.85s | 45.35s | 0.66s | 292.92s |
| Gemma | 100 | 7.06s | 6.42s | 3.06s | 17.53s |


## Recommended README Figures

Include only these figures in the README:

1. `reports/final_evaluation/figures/final_main_metrics.png`
2. `reports/final_evaluation/figures/final_confidence_comparison.png`
3. `reports/final_evaluation/figures/final_error_categories.png`
4. `reports/final_evaluation/figures/final_gemma_vs_qwen.png`
5. `reports/final_evaluation/figures/final_latency_comparison.png`

Keep the larger false-positive and false-negative charts in `reports/final_evaluation/figures/` for presentation backup.
