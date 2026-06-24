## 1 - Project Setup and VLM-First Planning

**Date:** 2026-04-29

**Team member(s):** Vaishnavi Narasimhaiah Sathish

**AI Tool used:** ChatGPT

### Context

We had downloaded the Roboflow `fridge-detection-merged` dataset and needed to start the Fridge-to-Recipe Assistant project in a structured way. The project should follow a VLM-focused approach, where YOLO/object detection is only used later as a comparison baseline.

### Prompt / Task

"I have downloaded this dataset (https://universe.roboflow.com/recipe-recomendation-app/fridge-detection-merged). Guide me on priliminary steps i need to setup before I actually start the project for example Git repo setup along with Initial Dataset exploration."

### AI Output Summary

ChatGPT suggested starting with repository initialization, `.gitignore`, a clean `README.md`, and a clear project direction. It recommended not committing the raw dataset and keeping YOLO only as a later baseline, while the main pipeline remains VLM-based.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The suggested workflow was useful, but it was simplified to avoid too many steps at once. We first initialized the repository, fixed `.gitignore`, connected the local repository to FIM Git, and pushed the initial commit. The raw dataset was excluded from Git because it is large and should stay local.

### Impact

This helped create a clean project start and meaningful initial Git history. It also clarified the project direction early: VLM-first ingredient extraction, with YOLO only as a comparison baseline.


## 2 - Dataset Audit and Class Distribution Analysis

**Date:** 2026-05-07

**Team member(s):** Vaishnavi Narasimhaiah Sathish

**AI Tool used:** ChatGPT

### Context

After organizing the Roboflow dataset locally, we needed to inspect the dataset before running any VLM experiments. The goal was to check whether the dataset structure, splits, labels, and ingredient classes were suitable for later VLM evaluation.

### Prompt / Task

"Give me the next step after creating the repository. I have the dataset with train, valid, test folders, each containing images and labels. Give me what exploratory steps I need to carry out."

### AI Output Summary

ChatGPT suggested creating a Python dataset audit script that reads the Roboflow `data.yaml`, counts images and label files in train/valid/test splits, extracts class names, and later extends the script to count object annotations per ingredient class.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The script structure was accepted but adapted to our local folder structure. We verified the output manually by checking the generated `reports/dataset_audit_report.md`. The audit confirmed that the dataset contains 2566 training images, 366 validation images, 182 test images, and 22 ingredient classes. The class distribution analysis was added as a separate improvement step.

### Impact

This created a useful first technical output for the project. The dataset audit report now supports later evaluation because the annotations can be used as ground truth for checking VLM ingredient recognition. It also produced meaningful Git commits for the university repository.


## 3 - Sample Image Grid and Visual Dataset Inspection

**Date:** 2026-05-13

**Team member(s):**  Vaishnavi Narasimhaiah Sathish

**AI Tool used:** ChatGPT

### Context

After completing the dataset audit and class distribution analysis, we needed to visually inspect the dataset before starting VLM experiments. The goal was to understand whether the images are suitable for a VLM-focused fridge-to-recipe assistant and whether the dataset contains realistic fridge scenes.

### Prompt / Task

Asked ChatGPT what should be included in `src/data/create_sample_grid.py` and how to document the visual dataset inspection based on generated sample grids.

### AI Output Summary

ChatGPT provided a Python script to generate sample image grids for the train, validation, and test splits. It also suggested a structure for `reports/dataset_visual_inspection.md`.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The script was used to generate sample grids for all dataset splits. The generated images were manually inspected before writing the visual inspection notes. The written observations were based on the actual dataset grids, not only on the AI suggestion.

### Impact

This helped us inspect the dataset more efficiently instead of opening individual images manually. The visual inspection confirmed that the dataset mainly contains real fridge interior images with clutter, occlusion, multiple ingredients, and packaging variation. This supports the decision to use the dataset for VLM-based ingredient recognition evaluation.

## 4 - Preliminary Open-Vocabulary VLM Trial and Output Analysis

**Date:** 2026-05-17

**Team member(s):**  Vaishnavi Narasimhaiah Sathish

**AI Tool used:** ChatGPT

### Context

After completing dataset inspection, we created a 50-image evaluation subset and tested the university InnKube VLM endpoint for open-vocabulary ingredient extraction. The goal was to check whether a VLM can identify visible fridge items beyond the limited 22 annotated dataset classes.

### Prompt / Task

Asked ChatGPT how to run a preliminary VLM baseline on 50 images, and analyze the raw VLM outputs using CSV summaries and visualizations.

### AI Output Summary

ChatGPT suggested scripts srunning the VLM baseline on 50 selected images, and analyzing the output. The analysis included ingredient frequency counts, uncertain item counts, predictions per image, and comparison between dataset-class predictions and open-vocabulary predictions.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The suggested workflow was used as a starting point and adapted to the project direction. We kept the task open-vocabulary because the fridge images contain more visible items than the 22 annotated dataset classes. The results were manually checked to understand whether extra VLM predictions were useful visible items or possible over-inference.

### Impact

This created the first working VLM trial for the project. The InnKube endpoint successfully processed the selected fridge images and returned structured ingredient outputs. The analysis showed that open-vocabulary extraction is more suitable for the Fridge-to-Recipe Assistant than restricting the task to the 22 dataset labels. It also revealed next improvements: prompt refinement, better parsing, ingredient normalization, and possible extraction of visible quantities or counts.

## 5 - Streamlit UI for Preliminary VLM Trial Review

**Date:** 2026-05-19

**Team member(s):** Vaishnavi Narasimhaiah Sathish

**AI Tool used:** ChatGPT

### Context

After running the preliminary open-vocabulary VLM trial on 50 fridge images, we needed a more intuitive way to inspect the results. Reading raw JSONL and CSV files was not convenient for manual review, error analysis, or weekly progress presentation.

### Prompt / Task

Asked ChatGPT whether it would be useful to create a UI at this stage and then requested code for a lightweight Streamlit review interface. The UI should focus on open-vocabulary VLM outputs rather than treating the dataset’s 22 labels as complete ground truth.

### AI Output Summary

ChatGPT suggested building a trial review UI instead of the final recipe assistant UI. It provided a Streamlit script that loads the preliminary VLM output file, displays each fridge image, shows predicted ingredients and uncertain items, and optionally shows dataset labels as partial reference metadata. The UI also includes raw VLM response inspection and a table of trial outputs.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The UI idea was accepted because it supports manual inspection and makes the preliminary VLM results easier to understand. The wording was adapted to emphasize that the task is open-vocabulary ingredient extraction and that dataset labels are not complete ground truth. Dataset labels are hidden by default and only shown as optional partial reference.

### Impact

The UI makes the current VLM trial more interpretable and presentation-friendly. It helps inspect whether predicted ingredients are clearly visible, uncertain, or possibly over-inferred. This will support the next step of prompt refinement, especially for reducing guessing and later extracting visible quantities or approximate counts.

## 6 - Structured VLM Run with Count-Aware Prompt

**Date:** 2026-05-20

**Team member(s):** Vaishnavi Narasimhaiah Sathish

**AI Tool used:** ChatGPT

### Context

After the preliminary open-vocabulary VLM trial, we needed a more structured VLM output format for evaluation. The first trial showed that the VLM could identify many visible fridge ingredients, but the output needed to be more consistent for comparison with manually created ground truth labels.

### Prompt / Task

Asked ChatGPT how to refine the VLM workflow so that the model returns structured ingredient outputs with ingredient name, approximate quantity/count, unit, confidence, and visual evidence. Also asked for code logic to resume the VLM run after connection errors or timeout failures.

### AI Output Summary

ChatGPT suggested using a stricter count-aware prompt and saving each image response as one JSONL line. It also suggested adding resume logic so that already completed images are skipped if the script is restarted after timeout or connection errors.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The prompt and script logic were adapted to the InnKube VLM endpoint and local repository structure. The final run was performed on the same 50-image evaluation subset to keep the VLM results comparable with the manually reviewed ground truth.

### Impact

This produced the final structured VLM output file used for evaluation:

    reports/vlm_predictions_v1.jsonl

The resume logic made the VLM run more reliable because failed or timed-out images could be retried without rerunning all completed images.


## 7 - Manual Ground Truth Review and VLM Output Adjudication

**Date:** 2026-05-23

**Team member(s):** Vaishnavi Narasimhaiah Sathish

**AI Tool used:** Claude Sonnet 4.6

### Context

A manually created ground truth file was needed for evaluating open-vocabulary VLM ingredient extraction. The original Roboflow labels were not sufficient because they contain only a limited set of dataset classes and do not cover all visible fridge items.

### Prompt / Task

Asked Claude how to create and review a manual ground truth file for the 50-image subset, how to handle extra VLM predictions, and whether extra ingredients predicted by the VLM should be added to the ground truth.

### AI Output Summary

Claude suggested using the same 50-image subset for manual annotation and creating a ground truth CSV with image identifiers and visible ingredient labels. It also suggested reviewing extra VLM predictions manually: if an extra prediction was clearly visible or readable in the image, it could be added to the reviewed ground truth; if it was guessed, unclear, or visually unverifiable, it should remain a false positive.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The workflow was used, but all ground truth corrections were made manually by visually inspecting the images. VLM outputs were not accepted automatically. Only clearly visible ingredients or readable labels were added to the reviewed ground truth.

### Impact

This produced the reviewed manual ground truth used for final evaluation:

    reports/manual_ground_truth_50.csv

The adjudication step made the evaluation more reliable by reducing missing ground truth labels while still keeping guessed VLM predictions as model errors.


## 8 - Ingredient Normalization and Quantitative Evaluation

**Date:** 2026-05-26

**Team member(s):** Vaishnavi Narasimhaiah Sathish

**AI Tool used:** ChatGPT

### Context

The manually reviewed ground truth and VLM predictions contained naming variations, spelling differences, plural forms, and brand/product names. Direct string matching would incorrectly count many correct predictions as errors.

### Prompt / Task

Asked ChatGPT to help create an ingredient normalization strategy and an evaluation script for comparing normalized VLM predictions against normalized manual ground truth labels.

### AI Output Summary

ChatGPT suggested creating a normalization dictionary and applying it to both ground truth and VLM predictions before evaluation. It also provided a Python evaluation script that calculates true positives, false positives, false negatives, precision, recall, F1-score, exact match accuracy, and mean Jaccard similarity.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The normalization rules were manually reviewed and adjusted before final evaluation. Some mappings were accepted for spelling variants, plural forms, and clear synonyms, while overly broad mappings were avoided to prevent artificially improving the score.

### Impact

The normalization file and evaluation script were added:

    configs/ingredient_normalization.json
    src/evaluation/evaluate_vlm_predictions.py

Final normalized evaluation results on the 50-image subset:

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

The results show that the VLM detects many visible ingredients but also tends to over-predict extra items in cluttered fridge scenes.


## 9 - Evaluation Visualization and Error Analysis Support

**Date:** 2026-05-27

**Team member(s):** Vaishnavi Narasimhaiah Sathish

**AI Tool used:** ChatGPT

### Context

After generating numerical evaluation results, we needed visualizations to make the results easier to interpret for the final report and presentation. Tables alone were not enough to explain model behavior, error patterns, and latency.

### Prompt / Task

Asked ChatGPT what visualizations would be useful for VLM ingredient extraction evaluation and requested code to generate them from the evaluation CSV files.

### AI Output Summary

ChatGPT suggested visualizing overall metrics, TP/FP/FN counts, per-image F1-score, F1-score distribution, precision-vs-recall scatter, top false positives, top false negatives, and VLM runtime. It provided a Python visualization script using Matplotlib and Pandas.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The suggested visualization set was accepted because it supports both quantitative evaluation and qualitative interpretation. The figures were generated from the final evaluation files and selected for use in the README/report.

### Impact

The visualization script and figures were added:

    src/evaluation/visualize_vlm_evaluation.py
    reports/evaluation/figures/

The visualizations help explain that the VLM has higher recall than precision, meaning it finds many visible ingredients but also produces extra predictions. Runtime plots also show that large VLM inference through an external endpoint has practical latency limitations.

## 10. - Bootstrap Confidence Intervals for Evaluation Metrics

**Date:** 2026-05-27

**Team member(s):** Vaishnavi Narasimhaiah Sathish

**AI Tool used:** ChatGPT

### Context

After computing precision, recall, F1-score, exact match accuracy, and mean Jaccard similarity for the 50-image VLM evaluation subset, we wanted to estimate how stable these metrics are. Since the evaluation subset contains only 50 images, a single metric value does not fully show uncertainty.

### Prompt / Task

Asked ChatGPT how bootstrapping can be used to estimate uncertainty in recall and other evaluation metrics, and how to implement image-level bootstrap resampling using the existing per-image evaluation results.

### AI Output Summary

ChatGPT explained that image-level bootstrapping is appropriate because each image is one evaluation unit and ingredients within the same image are not independent. It suggested sampling 50 images with replacement, recalculating micro and macro metrics for each bootstrap sample, repeating this for 10,000 iterations, and using the 2.5th and 97.5th percentiles as 95% confidence intervals.

ChatGPT also provided a Python script to perform bootstrap resampling using the existing per-image evaluation file.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The approach was accepted because it directly addresses uncertainty in the evaluation results. The bootstrap was applied at image level rather than ingredient level to preserve dependencies between predictions from the same fridge image.


## 11 - Final Error Analysis Visualizations

**Date:** 2026-06-27

**Team member(s):** Vaishnavi Narasimhaiah Sathish

**AI Tool used:** ChatGPT

### Context

After running the final 100-image evaluation, visualizations were needed to interpret the model’s error patterns. The focus was on false positives, false negatives, and the precision-recall behavior across images.

### Prompt / Task

Asked ChatGPT to provide plotting code for the top false positives, top false negatives, best/worst performing images, and precision-vs-recall pattern. Later, the best/worst image visualizations were removed because not all raw images from the second batch were available locally.

### AI Output Summary

ChatGPT provided a Matplotlib-based visualization script. The final retained visualizations are the top false positives, top false negatives, and precision-vs-recall scatter plot. It also helped debug why some best/worst image grids could not display images and explained that the missing images were not available in the local `data/raw/` folder.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The initial visualization script was modified to keep only the plots that do not depend on missing raw image files. Best/worst image visualizations were postponed until all raw images are available locally.

### Impact

These figures support the final interpretation that the VLM often detects relevant ingredients but tends to over-predict extra plausible items.


## 12 - Repository Cleanup and Final Organization

**Date:** 2026-06-04

**Team member(s):** Vaishnavi Narasimhaiah Sathish

**AI Tool used:** ChatGPT

### Context

The repository needed to be organized so that the final 100-image results were clearly visible.

### Prompt / Task

Asked ChatGPT how to organize the repository, decide which files should remain in the main view, which files should be moved to archived folders, and which helper files should be deleted or ignored.

### AI Output Summary

ChatGPT suggested keeping final 100-image files directly under `reports/` and `reports/evaluation_100/`, moving the earlier 50-image evaluation into `reports/evaluation_50/`, archiving preliminary VLM trial files under `reports/preliminary_vlm_trial/`, and ignoring local intermediate files. It also suggested using separate Git commits for major cleanup steps.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The cleanup plan was followed but adjusted manually based on the actual local folder structure. Some files were moved rather than deleted to preserve reproducibility, while temporary helper files were removed or left uncommitted.

### Impact

The repository structure was cleaned so that the repository is easier to understand and keeps the final results clearly separated from earlier experiments.

---

## 13 - Gemma 4 Batch Restructuring, Model Comparison, and Recipe Prediction Prompt

**Date:** 2026-06-25

**Team member(s):** Vaishnavi Narasimhaiah Sathish

**AI Tool used:** Claude Sonnet 4.6

### Context

A new 100-image Gemma 4 (`gemma4-31b-it`) annotation batch and a Streamlit review tool had been added but were not yet integrated: annotation CSVs were loose files under `reports/`, and there was no comparison against the existing Qwen baseline or its inference latency.

### Prompt / Task

Asked Claude to clean up file naming, split ground-truth and Gemma annotations into separate `data/annotations/` folders, generate a Gemma-vs-Qwen ingredient comparison report and a Gemma 4 latency report, write a structured few-shot recipe-prediction prompt, and update hardcoded paths across the repo to match.

### AI Output Summary

While exploring the review tool, Claude found that the "Auto-Review with Qwen" button actually called the Gemma model ID instead of Qwen's, meaning the existing `corrected_visible_ingredients` column reflected Gemma reviewing itself rather than independent Qwen review. Claude proposed fixing the bug and regenerating that column with genuine Qwen output before building the comparison report, moved the annotation CSVs into `data/annotations/manual_ground_truth_100/` and `data/annotations/gemma4_batch_100/`, renamed `tools/review_gemma_annotations_streamlit.py` to `app_gemma_review.py` (matching the existing `app_*.py` Streamlit convention) and `src/vlm/test_innkube_vlm.py` to `check_innkube_connection.py` (removing a `test_` prefix that risked pytest auto-collection), and added `src/evaluation/compare_gemma_vs_qwen.py`, `src/evaluation/analyze_gemma4_latency.py`, and `configs/recipe_prediction_prompt.txt`.

### Decision

- [ ] Accepted as-is
- [x] Modified before use
- [ ] Rejected

### Reasoning

The renaming scope was deliberately kept light-touch: established, README-documented folders like `evaluation_50/100`, `error_analysis_100`, and `confidence_filtering_100` were left untouched to avoid breaking already-published metrics and cross-references, while only the new/confusing items were renamed or relocated.

### Impact

The Gemma 4 batch is now reproducible and comparable: `reports/gemma4_batch_100/` holds the ingredient-agreement and latency reports, the data/annotations split makes it clear which CSV is manual ground truth versus model-generated, and the recipe-prediction prompt gives the project an LLM-based alternative to the deterministic coverage ranking in `src/recipe/retrieve_recipes.py`.

---