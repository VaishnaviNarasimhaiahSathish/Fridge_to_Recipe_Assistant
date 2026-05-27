# Fridge-to-Recipe Assistant

This project builds a VLM-focused fridge-to-recipe assistant for the Applied Artificial Intelligence Lab.

The goal is to identify visible ingredients from fridge images using Vision-Language Models and use the detected ingredients as the basis for recipe recommendation.

## Project Idea

The assistant takes a fridge image as input, extracts visible ingredients using a Vision-Language Model, normalizes the detected ingredient names, and uses them to suggest relevant recipes.

The main focus of the current project stage is open-vocabulary ingredient extraction and evaluation.

## Main Pipeline

    Fridge image → VLM-based ingredient extraction → ingredient normalization → recipe retrieval or generation → recipe ranking → recipe recommendation

## Dataset

We use the Roboflow `fridge-detection-merged` dataset.

The raw dataset is not committed to Git and is stored locally under:

    data/raw/

### Dataset Example

The fridge images contain cluttered shelves, occluded objects, packaging, and partially visible food items.

![Sample fridge images](reports/figures/sample_grid_valid.png)

Dataset statistics, split sizes, class names, and class distribution are documented in:

    reports/dataset_audit_report.md

The dataset contains YOLO-style labels for a limited set of ingredient classes. These labels are used as partial reference information only. The main evaluation uses manually reviewed open-vocabulary ground truth labels created for a selected 50-image subset.

## Current Repository Structure

    fridge-to-recipe-assistant/
    ├── README.md
    ├── ai_tool_usage.md
    ├── project_plan.md
    ├── app_vlm_review.py
    ├── configs/
    │   ├── vlm_prompt.txt
    │   ├── vlm_prompt_with_counts.txt
    │   └── ingredient_normalization.json
    ├── data/
    │   └── raw/                         # local only, not committed
    ├── reports/
    │   ├── dataset_audit_report.md
    │   ├── dataset_visual_inspection.md
    │   ├── vlm_eval_subset.csv
    │   ├── vlm_predictions_v1.jsonl
    │   ├── manual_ground_truth_50.csv
    │   ├── evaluation/
    │   │   ├── vlm_per_image_evaluation.csv
    │   │   ├── vlm_false_positives.csv
    │   │   ├── vlm_false_negatives.csv
    │   │   ├── vlm_evaluation_summary.md
    │   │   └── figures/
    │   │       ├── overall_metrics.png
    │   │       ├── tp_fp_fn_counts.png
    │   │       ├── per_image_f1_sorted.png
    │   │       ├── f1_score_distribution.png
    │   │       ├── precision_vs_recall_scatter.png
    │   │       ├── top_false_positives.png
    │   │       ├── top_false_negatives.png
    │   │       ├── runtime_distribution.png
    │   │       ├── runtime_per_image.png
    │   │       └── visual_evaluation_summary.md
    │   ├── preliminary_vlm_trial/
    │   │   ├── app_trial_review.py
    │   │   ├── vlm_predictions_raw.jsonl
    │   │   ├── vlm_predictions_flat.csv
    │   │   ├── vlm_ingredient_frequencies.csv
    │   │   ├── vlm_uncertain_frequencies.csv
    │   │   ├── vlm_item_frequencies_combined.csv
    │   │   ├── vlm_output_analysis.md
    │   │   └── figures/
    │   │       ├── vlm_top_ingredients.png
    │   │       ├── vlm_top_uncertain_items.png
    │   │       ├── vlm_predictions_per_image.png
    │   │       └── vlm_dataset_vs_open_vocab.png
    │   └── figures/
    │       ├── sample_grid_train.png
    │       ├── sample_grid_valid.png
    │       └── sample_grid_test.png
    └── src/
        ├── data/
        │   ├── dataset_audit.py
        │   ├── create_sample_grid.py
        │   └── create_vlm_eval_subset.py
        ├── evaluation/
        │   ├── evaluate_vlm_predictions.py
        │   └── visualize_vlm_evaluation.py
        └── vlm/
            ├── test_innkube_vlm.py
            ├── run_vlm_baseline.py
            ├── run_vlm_counts_trial.py
            └── analyze_trial_vlm_outputs.py

## Preliminary VLM Trial

Earlier exploratory VLM outputs and visualizations are stored under:

    reports/preliminary_vlm_trial/

These files document the first open-vocabulary VLM run before manual ground truth review, ingredient normalization, and final evaluation.

A lightweight Streamlit UI is available for reviewing preliminary VLM outputs image-by-image:

    streamlit run app_trial_review.py

The UI shows the input fridge image, open-vocabulary VLM ingredient predictions, uncertain items, optional dataset reference labels, and raw VLM responses.


## Structured VLM Review UI

A second Streamlit UI is available for reviewing the final structured 50-image VLM run:

    streamlit run app_vlm_review.py

This UI shows:

- input fridge image
- manual open-vocabulary ground truth
- VLM extracted ingredients
- quantity, unit, confidence, and visual evidence fields
- uncertain VLM items
- quick overlap between manual ground truth and VLM predictions
- raw VLM response when needed

This UI was used to manually inspect extra VLM predictions and decide whether they should be added to the reviewed ground truth.

## Current Progress

Completed steps:

1. Repository initialized and connected to FIM Git.
2. Dataset downloaded and organized locally.
3. Initial dataset audit completed.
4. Class distribution analysis completed.
5. Project documentation added.
6. Sample image grids generated for train, validation, and test splits.
7. Visual dataset inspection completed.
8. Created a 50-image VLM evaluation subset.
9. Ran a preliminary open-vocabulary VLM trial on 50 images.
10. Analyzed preliminary VLM outputs using frequency tables and visualizations.
11. Added a Streamlit trial review UI for inspecting preliminary VLM outputs.
12. Refined the prompt for structured ingredient extraction with quantities/counts.
13. Ran a structured VLM extraction trial on the selected 50-image subset.
14. Created manual open-vocabulary ground truth labels for the 50 images.
15. Reviewed extra VLM predictions against the original images.
16. Added clearly visible/readable ingredients missed in the first annotation to the reviewed ground truth.
17. Kept guessed, unclear, or unverifiable VLM predictions as false positives.
18. Created ingredient normalization rules for spelling variants, plural forms, synonyms, and selected brand/product names.
19. Evaluated VLM predictions using normalized ingredient names.
20. Generated visualizations for metrics, error patterns, and runtime analysis.

## Manual Ground Truth Review

The 50-image subset was manually annotated with visible ingredients.

After the structured VLM run, extra VLM predictions were reviewed against the original fridge images. If an extra prediction was clearly visible in the image or readable from the label, it was added to the reviewed ground truth. If the prediction appeared to be caused by guessing, unclear packaging, unreadable labels, or unverifiable visual evidence, it was kept as a false positive.

This process avoids blindly accepting VLM predictions while reducing incompleteness in the manual ground truth.

The final ground truth file is:

    reports/manual_ground_truth_50.csv

## Ingredient Normalization

Because this is an open-vocabulary extraction task, the same ingredient may appear under different names.

Examples:

    eggs → egg
    soya sauce → soy sauce
    shredded cheese → cheese
    land o'lakes butter → butter

Normalization rules are stored in:

    configs/ingredient_normalization.json

The normalization step is applied to both manual ground truth labels and VLM predictions before evaluation.

Generic or uncertain labels such as `unknown bottle`, `unknown jar`, `beverage`, `condiment`, and `leftover food` are excluded from the main ingredient-level metrics.

## Evaluation

Evaluation was performed on 50 manually reviewed fridge images.

Since this is an open-vocabulary multi-label ingredient extraction task, standard classification accuracy is not the main metric. Precision, recall, and F1-score are used as the primary metrics. Exact match accuracy and mean Jaccard similarity are reported as additional image level metrics.

The final normalized evaluation results are:

| Metric | Value |
|---|---:|
| Micro Precision | 0.5311 |
| Micro Recall | 0.6240 |
| Micro F1-score | 0.5738 |
| Macro Precision | 0.5807 |
| Macro Recall | 0.6558 |
| Macro F1-score | 0.5960 |
| Exact Match Accuracy | 0.0200 |
| Mean Jaccard Similarity | 0.4398 |

The VLM achieved higher recall than precision. This means that it detected many visible ingredients, but it also produced additional predictions that were not confirmed in the reviewed ground truth.

## Evaluation Outputs

The evaluation outputs are stored in:

    reports/evaluation/vlm_per_image_evaluation.csv
    reports/evaluation/vlm_false_positives.csv
    reports/evaluation/vlm_false_negatives.csv
    reports/evaluation/vlm_evaluation_summary.md

The main evaluation script is:

    src/evaluation/evaluate_vlm_predictions.py

Run evaluation with:

    python src/evaluation/evaluate_vlm_predictions.py

## Evaluation Visualizations

Evaluation visualizations are generated using:

    src/evaluation/visualize_vlm_evaluation.py

Run visualization generation with:

    python src/evaluation/visualize_vlm_evaluation.py

Generated figures are saved under:

    reports/evaluation/figures/

### Figure Descriptions

`overall_metrics.png` shows the main metric scores: precision, recall, F1-score, mean Jaccard similarity, and exact match accuracy.

`tp_fp_fn_counts.png` shows the total number of true positives, false positives, and false negatives.

`per_image_f1_sorted.png` shows image-level F1-scores sorted from highest to lowest.

`f1_score_distribution.png` shows how the image-level F1-scores are distributed across the 50-image subset.

`precision_vs_recall_scatter.png` shows one point per image, making it easier to see whether the VLM is over-predicting or missing ingredients.

`top_false_positives.png` shows the most frequent extra VLM predictions that were not present in the reviewed ground truth.

`top_false_negatives.png` shows the most frequent ground-truth ingredients missed by the VLM.

`runtime_distribution.png` and `runtime_per_image.png` summarize inference time and show the practical runtime cost of VLM-based extraction.

## Interpretation of Current Results

The current results show that the VLM is useful for broad open-vocabulary ingredient discovery in fridge images. However, it still tends to over-predict in cluttered scenes, especially when objects are partially visible, packaging is ambiguous, or labels are difficult to read.

The higher recall than precision indicates that the VLM finds many relevant ingredients, but stricter prompting, better post-processing, or confidence filtering may be needed to reduce false positives.

Exact match accuracy is low because it requires the complete predicted ingredient set to exactly match the ground truth set for an image. This is very strict for open-vocabulary fridge scenes. Mean Jaccard similarity is more informative because it measures partial set overlap between predicted and ground-truth ingredient sets.

### VLM Inference Latency

The VLM was queried image-by-image through the InnKube endpoint. Runtime varied across images, which shows the practical cost of using a large VLM for fridge image analysis.

![Runtime per image](reports/evaluation/figures/runtime_per_image.png)

## Limitations

Current limitations:

- Evaluation is based on a 50-image subset. Planned to extend upto 200.
- Manual ground truth is image-level, not bounding-box-level.
- Some ingredients are difficult to verify due to occlusion, unreadable labels, transparent packaging, or clutter.
- VLM outputs may include plausible but unverifiable guesses.
- Quantity extraction is approximate and not yet evaluated separately.
- Inference time is high because large VLMs are used through an external endpoint.

## Next Steps

1. Conduct detailed error analysis.
2. Summarize the most common false positives and false negatives.
3. Compare the VLM-based approach with a YOLO/object-detection baseline if time permits.
5. Add confidence filtering or post-processing rules.
6. Extend evaluation to a larger manually annotated subset if time allows.
7. Build the recipe recommendation module using normalized extracted ingredients.