# Trial VLM Output Analysis

## Goal

This analysis summarizes the first open-vocabulary VLM ingredient extraction run on the 50-image evaluation subset.

## Run Summary

- Total images in input file: 50
- Successful VLM calls: 50
- Responses parsed successfully: 50
- Responses needing parser improvement: 0
- Total predicted ingredient mentions: 702
- Total uncertain item mentions: 188
- Average predicted ingredients per image: 14.04

## Dataset Classes vs Open-Vocabulary Predictions

The dataset contains annotations for 22 selected ingredient classes. However, the VLM can identify additional visible fridge items outside this label space.
- Predictions matching the 22 annotated classes: 321
- Open-vocabulary predictions outside the 22 classes: 381

## Most Frequent Predicted Ingredients

| Ingredient | Count |
|---|---:|
| eggs | 29 |
| butter | 27 |
| yogurt | 25 |
| milk | 24 |
| cheese | 24 |
| lettuce | 20 |
| carrot | 20 |
| pickles | 19 |
| tomato | 19 |
| mustard | 18 |
| mayonnaise | 16 |
| chicken | 16 |
| soy sauce | 13 |
| blueberries | 13 |
| ketchup | 12 |

## Most Frequent Uncertain Items

| Item | Count |
|---|---:|
| relish | 7 |
| cream cheese | 7 |
| yogurt | 7 |
| jam | 7 |
| pickles | 6 |
| cheese | 6 |
| mushrooms | 6 |
| mustard | 5 |
| juice | 5 |
| salsa | 5 |
| meat | 5 |
| frozen vegetables | 4 |
| wine | 4 |
| butter | 4 |
| bread | 3 |

## Observed Issues

- Some responses are valid JSON but wrapped inside markdown code blocks.
- The VLM sometimes produces many ingredients per image, suggesting that the prompt may be too broad.
- Some outputs contain naming variations, for example singular/plural forms.
- Some outputs are broad or ambiguous, such as `berries`, `sauce`, `meat`, or `fruit`.
- Many outputs are outside the 22 annotated classes. These should not automatically be treated as hallucinations because the dataset labels are not exhaustive.

## Interpretation

The first trial confirms that the InnKube VLM endpoint can process fridge images and return structured ingredient lists. The results also support the open-vocabulary direction of the project, because the model identifies items beyond the dataset's 22 annotated classes. However, the output requires better parsing, normalization, and prompt refinement before final evaluation.

## Recommended Next Improvements

1. Improve JSON parsing to handle markdown-wrapped JSON.
2. Refine the prompt to reduce guessing and move unclear packaged items to the uncertain list.
3. Normalize ingredient names before recipe recommendation.
4. Manually inspect a small sample of open-vocabulary predictions to separate visible unannotated items from hallucinations.

## Generated Files

- `reports/vlm_predictions_flat.csv`
- `reports/vlm_ingredient_frequencies.csv`
- `reports/vlm_uncertain_frequencies.csv`
- `reports/vlm_item_frequencies_combined.csv`
- `reports/figures/vlm_top_ingredients.png`
- `reports/figures/vlm_top_uncertain_items.png`
- `reports/figures/vlm_predictions_per_image.png`
- `reports/figures/vlm_dataset_vs_open_vocab.png`