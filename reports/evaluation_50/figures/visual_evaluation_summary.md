# VLM Evaluation Visual Summary

## Final Metrics

| Metric | Value |
|---|---:|
| Precision | 0.5311 |
| Recall | 0.6240 |
| F1-score | 0.5738 |
| Mean Jaccard Similarity | 0.4398 |
| Exact Match Accuracy | 0.0200 |

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
