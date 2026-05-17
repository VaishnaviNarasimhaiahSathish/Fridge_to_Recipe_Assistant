# Dataset Visual Inspection

## Goal

The goal of this step is to manually inspect sample images from the train, validation, and test splits to understand whether the dataset is suitable for a VLM-focused fridge-to-recipe assistant.

## Sample Grids Generated

Sample image grids were generated for all three dataset splits:

- `reports/figures/sample_grid_train.png`
- `reports/figures/sample_grid_valid.png`
- `reports/figures/sample_grid_test.png`

Each grid contains 20 randomly selected images from the corresponding split.

## General Observation

The dataset mainly contains real fridge interior images. Most images show open refrigerators with visible food items placed on shelves, drawers, and door compartments. This makes the dataset relevant for the Fridge-to-Recipe Assistant task.

The images are more realistic than isolated ingredient images because they include clutter, different lighting conditions, packaging, partial visibility, and multiple objects in one scene.

## Image Type Observations

| Image Type | Observation |
|---|---|
| Real fridge images | Most images are real fridge interior images, which fits the project goal well. |
| Tabletop / cooking images | Not commonly observed in the sampled grids. |
| Isolated ingredient images | Not commonly observed and most ingredients appear inside fridge scenes. |
| Grocery-style images | Not common, although some images contain many packaged products similar to grocery style packaging|

## Visual Difficulty Observations

| Difficulty Factor | Observation |
|---|---|
| Occlusion | Some ingredients are partially hidden behind packaging, drawers, shelves, or other food items. |
| Small objects | Small fruits such as blueberries, strawberries, limes, and tomatoes may be difficult to detect in cluttered images. |
| Similar-looking ingredients | Items such as lemon/lime, tomato/bell pepper, and yogurt/mayonnaise/milk containers may be visually confusing. |
| Multiple objects in one image | Many images contain several ingredients, making the task suitable for multi-label ingredient recognition. |
| Poor lighting / blur | Some images have low light, reflections, or slight blur, especially in fridge corners and door compartments. |
| Packaging variation | Packaged items such as milk, yogurt, butter, mayonnaise, mustard, and ketchup appear in different shapes and brands, which may affect recognition. |

## Relevance for VLM Evaluation

The dataset is suitable for evaluating VLM-based ingredient recognition because it contains realistic fridge scenes rather than only clean object images. The VLM will need to handle clutter, packaging, occlusion, and multiple visible ingredients.

The dataset annotations can be used as ground truth to evaluate whether the VLM correctly identifies visible ingredients, misses ingredients, or hallucinates ingredients that are not present.

## Evaluation

The evaluation subset should include a mix of:

- simple fridge images with clearly visible ingredients
- cluttered fridge images
- images with multiple ingredients
- images with partially occluded ingredients
- images with packaged products
- images with visually similar classes

This will make the VLM evaluation more meaningful than testing only on easy examples.

