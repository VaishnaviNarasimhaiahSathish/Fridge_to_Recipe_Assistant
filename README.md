# Fridge-to-Recipe Assistant

This project builds a Vision-Language Model (VLM) focused fridge-to-recipe assistant for the Applied Artificial Intelligence Lab.

The goal is to identify visible ingredients from fridge images using a VLM and use the extracted ingredients as the basis for recipe recommendation.

The project covers open-vocabulary ingredient extraction, normalization, quantitative evaluation, qualitative error analysis, confidence filtering, and recipe retrieval and recommendation.

---

## Project Idea

The assistant takes a fridge image as input, extracts visible ingredients using a Vision-Language Model, and uses the final ingredient list for recipe recommendation.

The pipeline is:

```text
Fridge image → VLM-based ingredient extraction → confidence filtering → ingredient normalization → recipe retrieval → recipe ranking → recipe recommendation
```

---

## Dataset

This project uses the Roboflow `fridge-detection-merged` dataset.

The raw dataset is stored locally under:

```text
data/raw/
```

The dataset contains fridge images with cluttered shelves, occluded objects, packaging, and partially visible food items. The original YOLO-style labels cover only a limited set of ingredient classes, so they are treated as partial reference information only.

The main evaluation uses manually reviewed open-vocabulary image-level ground truth labels.

### Dataset Example

![Sample fridge images](reports/figures/sample_grid_valid.png)

Dataset statistics, split sizes, class names, and class distribution are documented in:

```text
reports/dataset_audit_report.md
```

Visual dataset inspection notes are documented in:

```text
reports/dataset_visual_inspection.md
```

---

## Repository Structure

```text
fridge-to-recipe-assistant/
├── README.md
├── ai_tool_usage.md
├── app_vlm_review.py
├── app_recipe_ui.py
├── app_gemma_review.py
├── configs/
│   ├── ingredient_normalization.json
│   ├── vlm_prompt.txt
│   ├── vlm_prompt_with_counts.txt
│   └── recipe_prediction_prompt.txt
├── data/
│   ├── raw/                              # local only, not committed
│   ├── recipes.json                      # local recipe dataset (500 recipes from RecipeNLG Lite)
│   ├── recipes_manual_40.json             # original 40 hand-written recipes (kept for reference)
│   └── annotations/
│       ├── manual_ground_truth_100/
│       │   └── manual_ground_truth_100.csv
│       └── gemma4_batch_100/
│           ├── gemma4_annotations_raw.csv
│           └── gemma4_annotations_reviewed.csv
├── reports/
│   ├── dataset_audit_report.md
│   ├── dataset_visual_inspection.md
│   ├── figures/
│   │   ├── sample_grid_train.png
│   │   ├── sample_grid_valid.png
│   │   └── sample_grid_test.png
│   ├── vlm_predictions_100.jsonl
│   ├── gemma4_batch_100/
│   │   ├── gemma_vs_qwen_summary.md
│   │   ├── gemma_vs_qwen_per_image.csv
│   │   ├── gemma_missed_ingredients.csv
│   │   ├── gemma4_latency_summary.md
│   │   └── figures/
│   ├── evaluation_100/
│   │   ├── vlm_per_image_evaluation.csv
│   │   ├── vlm_false_positives.csv
│   │   ├── vlm_false_negatives.csv
│   │   ├── vlm_evaluation_summary.md
│   │   ├── vlm_bootstrap_metrics.csv
│   │   ├── vlm_bootstrap_summary.md
│   │   └── figures/
│   │       ├── precision_vs_recall_100.png
│   │       ├── top_false_positives_100.png
│   │       └── top_false_negatives_100.png
│   ├── error_analysis_100/
│   │   ├── vlm_error_analysis_summary.md
│   │   ├── vlm_fp_categorized.csv
│   │   ├── vlm_worst_images.csv
│   │   └── figures/
│   │       ├── vlm_fp_error_categories.png
│   │       ├── vlm_top_false_positives.png
│   │       ├── vlm_top_false_negatives.png
│   │       └── vlm_f1_distribution.png
│   ├── confidence_filtering_100/
│   │   ├── vlm_confidence_filtering_summary.md
│   │   └── vlm_confidence_per_image.csv
│   ├── evaluation_50/
│   │   ├── inputs/
│   │   ├── vlm_per_image_evaluation.csv
│   │   ├── vlm_false_positives.csv
│   │   ├── vlm_false_negatives.csv
│   │   ├── vlm_evaluation_summary.md
│   │   ├── vlm_bootstrap_metrics.csv
│   │   ├── vlm_bootstrap_summary.md
│   │   └── figures/
│   ├── preliminary_vlm_trial/
│   │   ├── analyze_trial_vlm_outputs.py
│   │   ├── app_trial_review.py
│   │   ├── vlm_predictions_raw.jsonl
│   │   ├── vlm_predictions_flat.csv
│   │   ├── vlm_ingredient_frequencies.csv
│   │   ├── vlm_uncertain_frequencies.csv
│   │   ├── vlm_item_frequencies_combined.csv
│   │   ├── vlm_output_analysis.md
│   │   └── figures/
│   └── missing_images.txt
├── scripts/
│   └── build_recipe_dataset.py
└── src/
    ├── data/
    │   ├── dataset_audit.py
    │   ├── create_sample_grid.py
    │   └── create_vlm_eval_subset.py
    ├── evaluation/
    │   ├── evaluate_vlm_predictions.py
    │   ├── bootstrap_evaluation_metrics.py
    │   ├── visualize_vlm_evaluation.py
    │   ├── visualize_vlm_error_analysis.py
    │   ├── analyze_vlm_error_analysis.py
    │   └── filter_vlm_by_confidence.py
    ├── recipe/
    │   ├── __init__.py
    │   └── retrieve_recipes.py
    └── vlm/
        ├── check_innkube_connection.py
        ├── run_vlm_baseline.py
        ├── run_vlm_counts_trial.py
        ├── run_gemma4_annotation_batch.py
        └── vlm_jsonl_to_csv.py
```

---

## VLM-Based Ingredient Extraction

The project uses a VLM-first approach for ingredient extraction.

The VLM receives a fridge image and returns structured ingredient predictions. The final structured prompt asks the model to return:

* ingredient name
* quantity
* unit
* confidence
* visual evidence
* uncertain items

The structured prompt is stored in:

```text
configs/vlm_prompt_with_counts.txt
```

The VLM output is saved in JSONL format:

```text
reports/vlm_predictions_100.jsonl
```

---

## Manual Ground Truth

<!-- TODO: teammate to push 50 missing images to repo -->
50 of the 100 ground truth images are not yet pushed to this repo (pending upload from a teammate's machine). `src/evaluation/evaluate_vlm_predictions.py` and `src/evaluation/filter_vlm_by_confidence.py` detect this automatically — rows whose image isn't found in `data/raw/{train,valid,test}/images` are skipped with a console warning (`Skipping N images not found locally (teammate images pending upload)`), and evaluation continues on the remaining images instead of crashing. The metrics reported below were computed on the full 100-image set before the local image gap appeared; running either script locally right now will reproduce only the 50-image subset (different, lower numbers) until the teammate's images are committed — see `reports/missing_images.txt`. Re-run both scripts once they're pushed to refresh the full 100-image numbers.

The final evaluation uses a manually reviewed 100-image ground truth file:

```text
data/annotations/manual_ground_truth_100/manual_ground_truth_100.csv
```

Each image is annotated at image level with visible ingredients.

The manual review process follows these rules:

* Include ingredients that are clearly visible.
* Include packaged items only when the label or contents are visually identifiable.
* Do not add guessed or unverifiable items.
* Do not blindly accept VLM predictions.
* If a VLM prediction is clearly visible in the image but missed in the first annotation, add it to the reviewed ground truth.
* If a VLM prediction is plausible but not visually confirmable, keep it as a false positive.

This keeps the evaluation fair while reducing incompleteness in the manual ground truth.

---

## Ingredient Normalization

Because the task is open-vocabulary, the same ingredient can appear under different names.

Examples:

```text
eggs → egg
green bell pepper → bell pepper
```

Normalization rules are stored in:

```text
configs/ingredient_normalization.json
```

The same normalization is applied to both manual ground truth labels and VLM prediction labels.

Generic or uncertain labels are excluded from the main ingredient-level evaluation, for example:

```text
unknown bottle
prepared food
container
package
```

This prevents vague predictions from being counted as specific ingredient detections.

---

## Evaluation

The final evaluation was performed on 100 manually reviewed fridge images.

Since this is an open-vocabulary multi-label ingredient extraction task, standard classification accuracy is not the main metric. Precision, recall, and F1-score are used as the primary metrics. Exact match accuracy and mean Jaccard similarity are reported as additional image-level metrics.

The final normalized evaluation results are:

| Metric                  |  Value |
| ----------------------- | -----: |
| True Positives          |    477 |
| False Positives         |    519 |
| False Negatives         |    309 |
| Micro Precision         | 0.4789 |
| Micro Recall            | 0.6069 |
| Micro F1-score          | 0.5354 |
| Macro Precision         | 0.5007 |
| Macro Recall            | 0.6324 |
| Macro F1-score          | 0.5357 |
| Exact Match Accuracy    | 0.0200 |
| Mean Jaccard Similarity | 0.3910 |

The VLM achieved higher recall than precision. This means that it detected a reasonable portion of visible ingredients, but it also produced many additional predictions that were not confirmed in the reviewed ground truth.

---

## Precision vs Recall

![Precision vs Recall](reports/evaluation_100/figures/precision_vs_recall_100.png)

This plot shows one point per image. It helps identify whether the VLM is mostly over-predicting or missing ingredients.

Images with high recall but low precision indicate that the model found several correct ingredients but also added many extra predictions.

## Top False Positives

![Top false positives](reports/evaluation_100/figures/top_false_positives_100.png)

This plot shows the most frequent extra VLM predictions that were not present in the reviewed ground truth.

These false positives often come from:

* plausible but unverifiable guesses
* ambiguous packaging
* partially visible containers
* unreadable product labels
* context-based assumptions about common fridge items

## Top False Negatives

![Top false negatives](reports/evaluation_100/figures/top_false_negatives_100.png)

This plot shows the most frequent ground-truth ingredients missed by the VLM.

These false negatives often occur when ingredients are:

* small
* occluded
* in the background
* inside transparent or unclear packaging
* visually similar to other items
* not easily readable from the image

---

## Bootstrap Confidence Intervals

Image-level bootstrap resampling was used to estimate uncertainty in the final 100-image evaluation.

In each bootstrap iteration, 100 images were sampled with replacement from the evaluated subset, and the metrics were recalculated. This was repeated for 10,000 bootstrap iterations. The 2.5th and 97.5th percentiles of the bootstrap distributions were used as 95% confidence intervals.

| Metric                  | Original Value | Bootstrap Mean | 95% CI Lower | 95% CI Upper |
| ----------------------- | -------------: | -------------: | -----------: | -----------: |
| Micro Precision         |         0.4789 |         0.4789 |       0.4379 |       0.5204 |
| Micro Recall            |         0.6069 |         0.6072 |       0.5666 |       0.6475 |
| Micro F1-score          |         0.5354 |         0.5352 |       0.5000 |       0.5703 |
| Macro Precision         |         0.5007 |         0.5003 |       0.4566 |       0.5444 |
| Macro Recall            |         0.6324 |         0.6327 |       0.5878 |       0.6763 |
| Macro F1-score          |         0.5357 |         0.5356 |       0.4961 |       0.5738 |
| Mean Jaccard Similarity |         0.3910 |         0.3908 |       0.3537 |       0.4284 |
| Exact Match Accuracy    |         0.0200 |         0.0200 |       0.0000 |       0.0500 |

The bootstrap means are very close to the original metric values, which suggests that the reported scores are stable for the selected 100-image evaluation subset.

The micro-F1 score remains within approximately 0.50 to 0.57 under bootstrap resampling. The recall confidence interval is higher than the precision confidence interval, supporting the observation that the VLM is more recall-oriented than precision-oriented.

---

## Phase 1 — Error Analysis

Qualitative error analysis was performed on the 519 false positives and 309 false negatives from the 100-image evaluation.

False positives were categorized into four error types:

| Category | Count | Share | Description |
| ----------------------- | -----: | -----: | --- |
| Common fridge default | 162 | 31% | Model predicts staples (cheese, butter, yogurt) without visual confirmation |
| Ambiguous visual | 153 | 29% | Ingredients easily confused with others in a cluttered fridge |
| Other | 164 | 32% | Specific items the model over-predicts for unclear reasons |
| Context guess | 40 | 8% | Liquids inferred from context (water, juice, soda) rather than visual evidence |

The analysis script and outputs are in:

```text
src/evaluation/analyze_vlm_error_analysis.py
reports/error_analysis_100/
```

Run:

```bash
python src/evaluation/analyze_vlm_error_analysis.py
```

---

## Phase 2 — Confidence Filtering

VLM predictions include a per-ingredient confidence field (`high` or `medium`). Filtering out medium-confidence predictions reduces false positives at the cost of slightly lower recall.

| Strategy | TP | FP | FN | Precision | Recall | Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (high + medium) | 458 | 482 | 298 | 0.4872 | 0.6058 | 0.5401 |
| High confidence only | 419 | 340 | 337 | 0.5520 | 0.5542 | **0.5531** |

Filtering medium-confidence predictions reduces false positives by 29% and improves precision by 6.5 points. For recipe recommendation, high-confidence-only is preferred because fewer incorrect ingredients produce more trustworthy recipe suggestions.

The filtering script and outputs are in:

```text
src/evaluation/filter_vlm_by_confidence.py
reports/confidence_filtering_100/
```

Run:

```bash
python src/evaluation/filter_vlm_by_confidence.py
```

---

## Phase 3 — Recipe Retrieval and Recommendation

Normalized high-confidence ingredients are matched against a local recipe dataset to produce ranked recipe recommendations.

The recipe dataset contains 500 recipes sourced from the [RecipeNLG Lite](https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite) dataset on Hugging Face, filtered to recipes that overlap with common fridge ingredients found in VLM predictions (egg, milk, butter, cheese, tomato, carrot, apple, lemon, cucumber, yogurt, chicken, onion, garlic, pepper) and ranked by ingredient coverage. Each recipe includes a title, ingredient list, cuisine, meal type, prep time, and step-by-step instructions. The original 40 hand-written recipes are kept for reference in `data/recipes_manual_40.json`.

Recipes are ranked by ingredient coverage — the share of recipe ingredients confirmed in the fridge. Ties are broken by number of missing ingredients, then by prep time.

**Ingredient matching.** Recipe ingredient lines in `data/recipes.json` are raw descriptive phrases as scraped from RecipeNLG Lite (e.g. `"medium tomato"`, `"crumbled feta cheese"`, `"2% milk"`), not canonical ingredient names. Matching these against the normalized fridge ingredients (e.g. `"tomato"`, `"cheese"`, `"milk"`) with exact string equality almost never succeeds, which previously produced very low coverage (12-16%) and only 1-2 matched ingredients per recipe even when the right items were extracted. `ingredient_in_text()` in `src/recipe/retrieve_recipes.py` fixes this with whole-word, plural-tolerant token matching: a fridge ingredient is counted as present if its word(s) appear as a contiguous run inside the recipe ingredient phrase (after naive singularization), while still rejecting unrelated containment such as `"egg"` inside `"eggplant"` or `"milk"` inside `"buttermilk"`. This raised coverage to a realistic 30-40%+ range for the same extracted ingredients.

The recipe dataset is stored in:

```text
data/recipes.json
```

It is built by:

```text
scripts/build_recipe_dataset.py
```

Run it to regenerate the dataset:

```bash
pip install datasets
python scripts/build_recipe_dataset.py
```

The retrieval and ranking module is in:

```text
src/recipe/retrieve_recipes.py
```

Run a quick test:

```bash
python src/recipe/retrieve_recipes.py
```

---

## Streamlit UIs

### VLM Review UI

Used to inspect VLM predictions and support manual ground truth review:

```bash
streamlit run app_vlm_review.py
```

### Recipe Recommendation UI

Full end-to-end pipeline: select a fridge image, review extracted ingredients, and get ranked recipe recommendations with coverage percentage and missing ingredient list:

```bash
streamlit run app_recipe_ui.py
```

The UI uses a dark, card-based layout:

* **Step 1 card** — image selector with the fridge photo previewed underneath.
* **Step 2 card** — extracted ingredients shown as green pill badges (🟢 high confidence) and amber pills (🟡 medium confidence, only when "High confidence only" is off), with a "Filtered out" expander listing predictions dropped by the confidence threshold or the generic-term ignore list.
* **Step 3 cards** — one card per recommended recipe: large title, a big color-coded coverage number (green ≥50%, orange 30-49%, red <30%), "You have" ingredients as green pills, "You need" ingredients as red pills, a cuisine/meal-type/prep-time meta line, and an expander with the full numbered instructions.

The sidebar is organized into three sections:

* **⚙️ Filters** — high-confidence-only toggle, dietary filter (All / Vegetarian / Quick (<30min)), and a recipe-count slider (5-20).
* **🖼️ Dataset** — count of locally available images and a train/valid/test split breakdown, plus a warning if any ground truth images are pending upload (see [Manual Ground Truth](#manual-ground-truth)).
* **🐞 Debug** — a "Show debug info" toggle that reveals the extraction → normalization → matching pipeline trace per ingredient (raw name, normalized name, confidence, kept/filtered reason) and the matched ingredients behind the top 3 recipe results, useful for diagnosing low-coverage cases.

A search box to filter recipes by name sits above the recipe list in the main column.

### Gemma 4 Annotation Review UI

Used to manually review and correct a separate 100-image Gemma 4 annotation batch, with an optional Qwen auto-review pass:

```bash
streamlit run app_gemma_review.py
```

---

## Phase 4 — Gemma 4 Batch and Model Comparison

A second, non-overlapping 100-image batch was annotated with Gemma 4 (`gemma4-31b-it`) to compare against the Qwen baseline (`qwen35-397b`) used for the main evaluation above. The batch was generated with:

```text
src/vlm/run_gemma4_annotation_batch.py
```

and is stored under:

```text
data/annotations/gemma4_batch_100/gemma4_annotations_raw.csv
data/annotations/gemma4_batch_100/gemma4_annotations_reviewed.csv
```

Each image's Gemma prediction was reviewed using Qwen as an independent second model via `app_gemma_review.py`, producing a `corrected_visible_ingredients` column. Since this batch has no independent manual ground truth, the Qwen-reviewed column is used as a ground-truth proxy when scoring Gemma.

> Note: an earlier version of the review tool had a bug where the "Qwen auto-review" button called the Gemma model ID instead of Qwen's. This has been fixed (`app_gemma_review.py` now calls `qwen35-397b`), and the existing reviewed CSV was regenerated with genuine Qwen output.

### Gemma vs Qwen Ingredient Comparison

```bash
python src/evaluation/compare_gemma_vs_qwen.py
```

Gemma's raw predictions are scored against Qwen's review, used as a ground-truth proxy:

| Metric | Value |
|---|---:|
| Micro Precision | 0.7241 |
| Micro Recall | 0.2263 |
| Micro F1-score | 0.3448 |
| Macro F1-score | 0.3314 |
| Mean Jaccard Similarity | 0.2160 |
| Avg. ingredients detected — Gemma | 3.95 |
| Avg. ingredients detected — Qwen | 12.64 |

Gemma is precise but conservative: when it names an ingredient it is usually correct (precision 0.72), but it detects only about a third of the ingredient count Qwen confirms per image (recall 0.23). The most frequently missed ingredients are pantry/condiment items such as yogurt, butter, cheese, mustard, and pickles — items Gemma appears reluctant to report without very clear visual evidence. Full breakdown in `reports/gemma4_batch_100/gemma_vs_qwen_summary.md`.

### Gemma 4 Latency Analysis

```bash
python src/evaluation/analyze_gemma4_latency.py
```

| Metric | Value |
|---|---:|
| Mean inference time | 7.06s |
| Median inference time | 6.42s |
| Min / Max | 3.06s / 17.53s |

Latency scales with ingredient count: images with 10+ detected ingredients average 15.4s versus 5.4s for images with 0-3. Full breakdown in `reports/gemma4_batch_100/gemma4_latency_summary.md`.

### Recipe Prediction Prompt

A structured, few-shot recipe-prediction prompt (4 worked ingredient → recipe examples) is available for LLM-based recipe suggestion as an alternative to the deterministic coverage-ranking in `src/recipe/retrieve_recipes.py`:

```text
configs/recipe_prediction_prompt.txt
```

---

## VLM Inference Latency

The VLM was queried image-by-image through the InnKube endpoint. Latency analysis
was performed on 50 images.

| Metric | Value |
|---|---:|
| Min | 0.7s |
| Max | 292.9s |
| Mean | 49.0s |
| Median | 17.8s |
| Stdev | 67.2s |

| Range | Images |
|---|---:|
| Under 10s | 12 |
| 10 – 30s | 22 |
| 30 – 60s | 5 |
| 60 – 120s | 5 |
| Over 120s | 6 |

The large gap between mean (49s) and median (17.8s) indicates that latency is
driven by server load variability rather than image complexity. Six extreme
outliers (over 120s) account for most of the total inference time. When the
server is under low load, responses arrive in under 10 seconds.

Three strategies are implemented to reduce effective latency:

1. **Prediction caching** — predictions are stored by image hash in
   `reports/vlm_prediction_cache.json`. Repeated queries return instantly
   without hitting the API. See `src/vlm/vlm_prediction_cache.py`.
2. **Image resizing** — images are resized to 512px on the longest side
   before sending, reducing token count and inference time.
   See `src/vlm/run_vlm_cached.py`.
3. **Offline batch processing** — run inference overnight for all images,
   store results in cache. The Streamlit UI reads from cache with zero latency.

![Runtime per image](reports/latency_analysis/figures/vlm_runtime_per_image.png)
![Runtime distribution](reports/latency_analysis/figures/vlm_runtime_distribution.png)

---

## Preliminary VLM Trial

Earlier exploratory VLM outputs and visualizations are stored under:

```text
reports/preliminary_vlm_trial/
```

These files document the first open-vocabulary VLM run before final manual ground truth review, ingredient normalization, and 100-image evaluation.

---

## Interpretation of Final Results

The final 100-image evaluation shows that the VLM is useful for broad open-vocabulary ingredient discovery in fridge images.

The baseline model achieved:

```text
Micro Precision: 0.4789
Micro Recall:    0.6069
Micro F1-score:  0.5354
```

After Phase 2 confidence filtering:

```text
Micro Precision: 0.5520
Micro Recall:    0.5542
Micro F1-score:  0.5531
```

The higher recall than precision in the baseline indicates that the model detects many relevant ingredients but frequently over-predicts plausible items that are not visually confirmed. Confidence filtering addresses this by discarding medium-confidence predictions, which are the primary source of common fridge defaults and context guesses.

Exact match accuracy is low because it requires the complete predicted ingredient set to exactly match the ground truth set for an image. This is very strict for open-vocabulary multi-label ingredient extraction. Mean Jaccard similarity is more informative because it measures partial overlap between predicted and ground-truth ingredient sets.

---

## Limitations

* Some ingredients are difficult to verify due to occlusion, clutter, transparent packaging, or unreadable labels.
* VLM outputs may include plausible but unverifiable guesses even at high confidence.
* The recipe dataset (500 recipes from RecipeNLG Lite) does not cover all possible ingredient combinations, and cuisine/meal type are unlabeled in the source data ("unknown").
* Inference time is high because a large VLM is queried through an external endpoint.
* Recipe ingredient matching uses a naive suffix-stripping singularizer rather than a real lemmatizer, so a few irregular plurals (e.g. "leaves" → "leav" instead of "leaf") fail to match even when the ingredient is present.
* 50 of the 100 manual ground truth images are not yet committed to this repo, so local evaluation runs currently cover only half the set until they're pushed (see [Manual Ground Truth](#manual-ground-truth)).

---

## Next Steps

1. Re-run the full 100-image evaluation and confidence-filtering scripts once the teammate's 50 pending images are pushed, and refresh the headline metrics above.
2. Explore runtime reduction strategies for practical deployment.
3. Refine the VLM prompt to further reduce common fridge default predictions.
4. Label cuisine and meal type for the RecipeNLG-derived recipes (currently "unknown") to enable richer filtering.
5. Replace the naive singularization in `ingredient_in_text()` with a proper lemmatizer or a small synonym table to catch the remaining irregular-plural and synonym mismatches in recipe ingredient matching.
6. Visually QA the redesigned `app_recipe_ui.py` in a real browser session (current verification used Streamlit's headless `AppTest`, which confirms no runtime errors and correct data but not pixel-level styling).