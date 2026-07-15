# Recipe Recommendation Evaluation

## Inputs

- VLM predictions: `reports/vlm_predictions_100.jsonl`
- Normalization: `configs/ingredient_normalization.json`
- Recipe database: `data/recipes.json`

The evaluation uses high-confidence VLM ingredients because this matches the React + FastAPI demo app operating point.

## Summary Metrics

| Metric | Value |
|---|---:|
| Images evaluated | 100 |
| Mean detected ingredients per image | 7.59 |
| Median detected ingredients per image | 7.00 |
| Mean top-1 recipe coverage | 0.3913 |
| Median top-1 recipe coverage | 0.3750 |
| Mean best top-3 recipe coverage | 0.4033 |
| Mean best top-5 recipe coverage | 0.4050 |
| Mean missing ingredients in top recipe | 4.35 |
| Median missing ingredients in top recipe | 5.00 |
| Images with at least one recipe >= 50% match | 29 (29%) |
| Images with at least one recipe >= 70% match | 2 (2%) |

## Top Recipe Difficulty

| Difficulty | Images |
|---|---:|
| Easy | 9 |
| Medium | 69 |
| Hard | 22 |

## Interpretation

This evaluation measures whether the detected fridge ingredients can retrieve feasible recipes from the local recipe database. It does not measure taste preference, user satisfaction, or nutritional quality.

A higher top-k coverage means that the recommender is finding recipes whose ingredients overlap well with what the VLM detected. Missing-ingredient difficulty indicates whether the remaining ingredients are basic staples, common groceries, or more specific items.

## Generated Files

| File | Description |
|---|---|
| `recipe_recommendation_per_image.csv` | Per-image recommendation metrics |
| `figures/recipe_top1_coverage_distribution.png` | Top-1 coverage distribution |
| `figures/recipe_topk_coverage.png` | Mean top-k coverage comparison |
| `figures/recipe_match_thresholds.png` | Share of images with usable recipe matches |
| `figures/recipe_top1_difficulty_distribution.png` | Difficulty distribution for top recipe |
