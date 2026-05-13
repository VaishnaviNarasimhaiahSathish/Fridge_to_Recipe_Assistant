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
    ├── data/
    │   └── raw/                  # local only, not committed
    ├── reports/
    │   └── dataset_audit_report.md
    └── src/
        └── data/
            └── dataset_audit.py

## Current Progress

- Repository initialized
- Dataset downloaded and organized
- Initial dataset audit completed
- Class distribution analysis completed
- Project documentation added

## Next Steps

1. Create a small sample set for VLM testing
2. Run first VLM ingredient extraction baseline
3. Normalize VLM outputs to dataset class names
4. Evaluate VLM predictions against annotation labels
5. Compare VLM results with YOLO baseline
6. Build recipe recommendation module