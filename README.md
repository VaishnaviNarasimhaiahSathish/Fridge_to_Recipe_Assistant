# Fridge-to-Recipe Assistant

This project builds a VLM-focused fridge-to-recipe assistant for the Applied Artificial Intelligence Lab.

The goal is to identify visible ingredients from fridge images using Vision-Language Models and recommend suitable recipes based on the detected ingredients.

This is not a traditional computer vision project. YOLO/object detection is used only as a comparison baseline, not as the main pipeline.

## Project Idea

The assistant takes a fridge image as input, extracts ingredients using a Vision-Language Model, normalizes the detected ingredients, and uses them to suggest relevant recipes.

## Main Pipeline

    Fridge image
    → VLM-based ingredient extraction
    → ingredient normalization
    → recipe retrieval or generation
    → recipe ranking
    → recipe recommendation

## Baseline Comparison

    Fridge image
    → YOLO object detection
    → detected ingredient labels
    → comparison with VLM output

## Dataset

We use the Roboflow `fridge-detection-merged` dataset.

The raw dataset is not committed to Git and is stored locally under:

    data/raw/

Dataset statistics, split sizes, class names, and class distribution are documented in:

    reports/dataset_audit_report.md

## Current Repository Structure

    fridge-to-recipe-assistant/
    ├── README.md
    ├── ai_tool_usage.md
    ├── project_plan.md
    ├── configs/
    │   └── vlm_prompt.txt
    ├── data/
    │   └── raw/                  # local only, not committed
    ├── reports/
    │   ├── dataset_audit_report.md
    │   ├── dataset_visual_inspection.md
    │   ├── vlm_eval_subset.csv
    │   ├── vlm_predictions_raw.jsonl
    │   ├── vlm_predictions_flat.csv
    │   ├── vlm_ingredient_frequencies.csv
    │   ├── vlm_uncertain_frequencies.csv
    │   ├── vlm_item_frequencies_combined.csv
    │   ├── vlm_output_analysis.md
    │   └── figures/
    │       ├── sample_grid_train.png
    │       ├── sample_grid_valid.png
    │       ├── sample_grid_test.png
    │       ├── vlm_top_ingredients.png
    │       ├── vlm_top_uncertain_items.png
    │       ├── vlm_predictions_per_image.png
    │       └── vlm_dataset_vs_open_vocab.png
    └── src/
        ├── data/
        │   ├── dataset_audit.py
        │   ├── create_sample_grid.py
        │   └── create_vlm_eval_subset.py
        └── vlm/
            ├── test_innkube_vlm.py
            ├── run_vlm_baseline.py
            └── analyze_trial_vlm_outputs.py

## Current Progress

- Repository initialized and connected to FIM Git
- Dataset downloaded and organized locally
- Initial dataset audit completed
- Class distribution analysis completed
- Project documentation added
- Sample image grids generated for train, validation, and test splits
- Visual dataset inspection completed
- Created a 50-image VLM evaluation subset
- Tested the InnKube VLM endpoint with one fridge image
- Ran a preliminary open-vocabulary VLM trial on 50 images
- Analyzed preliminary VLM outputs using frequency tables and visualizations


## Next Steps

1. Refine the VLM prompt to reduce guessing and improve structured ingredient extraction
2. Extend the prompt to extract ingredient quantities or approximate counts where visible
3. Rerun the VLM baseline with the improved prompt
4. Normalize ingredient names from open-vocabulary outputs
5. Analyze visible-but-unannotated items separately from hallucinations
6. Build the recipe recommendation module using extracted ingredients
7. Compare VLM-based extraction with a YOLO/object-detection baseline