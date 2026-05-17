## Entry #1 - Project Setup and VLM-First Planning

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


## Entry #2 - Dataset Audit and Class Distribution Analysis

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


## Entry #3 - Sample Image Grid and Visual Dataset Inspection

**Date:** 2026-05-13

**Team member(s):** Vaishnavi N.S

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