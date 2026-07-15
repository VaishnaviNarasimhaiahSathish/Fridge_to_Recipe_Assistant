# Fridge-to-Recipe Assistant

A Vision-Language Model based fridge-to-recipe assistant built for the Applied Artificial Intelligence Lab.

The system uses saved VLM predictions from fridge images to identify visible ingredients, filter and normalize them, recommend recipes, explain missing ingredients, and provide static grocery guidance for completing recipes.

```text
Fridge image → VLM ingredient extraction → confidence filtering → ingredient normalization → recipe retrieval
→ recipe ranking → missing ingredient analysis → grocery suggestion → React + FastAPI demo
```
The current demo uses saved predictions instead of live VLM inference to keep the app stable and reproducible.

---

## Key Features

- VLM-based open-vocabulary ingredient extraction
- 100-image manually reviewed ground truth
- Ingredient normalization and confidence filtering
- VLM evaluation, confidence calibration, and error analysis
- Hybrid recipe retrieval using local BM25-style search and ingredient coverage
- Missing ingredient difficulty scoring
- Static grocery guidance for missing items
- React + FastAPI demo interface
- Recipe recommendation evaluation

---

## Demo Application

The app is built with:

```text
Frontend: React + Vite
Backend: FastAPI
Recipe data: 500 RecipeNLG Lite recipes
Prediction mode: saved VLM predictions
```

The user can:

1. Select a fridge image.
2. View high-confidence detected ingredients.
3. Get recipe recommendations.
4. See matched and missing recipe ingredients.
5. View whether a recipe is easy or difficult to complete.
6. See static grocery suggestions for missing items.

The app does not use live grocery inventory, live prices, or live opening hours.

---

## Running the App

### 1. Start the backend

From the project root:

```bash
uvicorn backend.api:app --reload --port 8000
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

### 2. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite, usually:

```text
http://localhost:5173
```

---

## Repository Structure

```text
fridge-to-recipe-assistant/
├── backend/
│   └── api.py
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── styles.css
├── configs/
│   ├── ingredient_normalization.json
│   ├── vlm_prompt.txt
│   └── vlm_prompt_with_counts.txt
├── data/
│   ├── raw/
│   ├── recipes.json
│   ├── recipes_manual_40.json
│   └── annotations/
│       ├── final_images_200/
│       ├── manual_ground_truth_100/
│       └── gemma4_batch_100/
├── reports/
│   ├── vlm_predictions_100.jsonl
│   ├── evaluation_100/
│   ├── error_analysis_100/
│   ├── confidence_calibration/
│   ├── final_evaluation/
│   ├── gemma4_batch_100/
│   └── recipe_recommendation_evaluation/
├── scripts/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── grocery/
│   ├── ingredients/
│   ├── recipe/
│   └── vlm/
└── README.md
```

---

## VLM-Based Ingredient Extraction

The project uses a VLM-first approach because fridge images contain open-vocabulary items such as packaged food, jars, sauces, dairy products, vegetables, leftovers, and partially visible ingredients.

The VLM returns structured predictions with:

- ingredient name
- approximate quantity or count
- unit
- confidence label
- visual evidence
- uncertain items

Saved VLM predictions are stored in:

```text
reports/vlm_predictions_100.jsonl
```

The main prompt is stored in:

```text
configs/vlm_prompt_with_counts.txt
```

---

## Ground Truth and Normalization

The main evaluation uses manually reviewed image-level ground truth for 100 fridge images:

```text
data/annotations/manual_ground_truth_100/manual_ground_truth_100.csv
```

Manual review includes only visually confirmable ingredients. Plausible but unverifiable VLM guesses are kept as false positives.

Ingredient normalization is stored in:

```text
configs/ingredient_normalization.json
```

Examples:

```text
eggs → egg
green bell pepper → bell pepper
plain greek yogurt → yogurt
cherry tomatoes → tomato
```

The normalization is intentionally conservative.

---

## Evaluation Summary

### VLM Ingredient Extraction

Final high + medium confidence evaluation on 100 manually reviewed images:

| Metric | Value |
|---|---:|
| Images evaluated | 100 |
| True Positives | 460 |
| False Positives | 479 |
| False Negatives | 299 |
| Micro Precision | 0.4899 |
| Micro Recall | 0.6061 |
| Micro F1-score | 0.5418 |
| Mean Jaccard Similarity | 0.3960 |
| Exact Match Accuracy | 0.0200 |

The VLM detects many visible ingredients, but also over-predicts plausible fridge items.

### Confidence Calibration

The app uses high-confidence predictions only.

| Confidence | Predictions | True Positives | False Positives | Precision |
|---|---:|---:|---:|---:|
| High | 759 | 421 | 338 | 0.5547 |
| Medium | 186 | 44 | 142 | 0.2366 |

High-confidence predictions are more reliable, so medium-confidence predictions are filtered out in the demo app.

### Recipe Recommendation

Recipe recommendation evaluation uses the same high-confidence predictions as the app.

| Metric | Value |
|---|---:|
| Images evaluated | 100 |
| Mean detected ingredients per image | 7.59 |
| Mean top-1 recipe coverage | 0.3913 |
| Mean best top-5 recipe coverage | 0.4050 |
| Mean missing ingredients in top recipe | 4.35 |
| Images with at least one recipe >= 50% match | 29 |
| Images with at least one recipe >= 70% match | 2 |

The recipe module works as a prototype, but recommendation quality is limited by the local recipe database and VLM prediction noise.

---

## Recipe Retrieval

Recipes are retrieved from:

```text
data/recipes.json
```

The main retrieval logic is implemented in:

```text
src/recipe/retrieve_recipes_hybrid.py
```

The hybrid ranking uses:

1. BM25-style candidate search
2. Ingredient coverage
3. Missing ingredient count
4. Missing ingredient difficulty
5. Preparation time
6. Grocery suggestions for missing items

Run a quick recipe retrieval test:

```bash
python src/recipe/retrieve_recipes_hybrid.py
```

---

## Grocery Guidance

Missing ingredients are categorized as:

```text
basic_staple
common_grocery
specific_grocery
special_ingredient
```

The app uses these categories to provide static grocery suggestions, for example:

```text
basic staples → ALDI, Lidl, Penny, Netto
common groceries → REWE, EDEKA, Kaufland
specific ingredients → wider-selection supermarkets
```

This is static guidance only. The app does not claim real-time availability.

---

## Analysis and Experiments

Additional experiment outcomes:

- Gemma was tested as a comparison model. It was faster and more conservative, but detected fewer ingredients.
- Model ensembling was considered but not included because the available Qwen, Gemma, and ground-truth image IDs did not overlap.
- Ingredient taxonomy was attempted and rolled back because it reduced recipe recommendation quality.

---

## Limitations

- The app currently uses saved predictions, not live VLM inference.
- VLM predictions may include plausible but visually unverifiable ingredients.
- Fridge images are cluttered, occluded, and often contain unreadable packaging.
- Recipe recommendations depend on the limited 500-recipe local dataset.
- Grocery guidance is static and does not use live inventory or pricing.
- Online deployment is future work.

---

## Reproducibility

### Run VLM evaluation

```bash
python src/evaluation/evaluate_vlm_predictions.py
```

### Run confidence calibration

```bash
python src/evaluation/analyze_confidence_calibration.py
```

### Run error analysis

```bash
python src/evaluation/analyze_vlm_error_analysis.py
```

### Run recipe recommendation evaluation

```bash
python src/evaluation/evaluate_recipe_recommendations.py
```

### Run Gemma vs Qwen comparison

```bash
python src/evaluation/compare_gemma_vs_qwen.py
```

### Run recipe retrieval test

```bash
python src/recipe/retrieve_recipes_hybrid.py
```

### Run full demo

Terminal 1:

```bash
uvicorn backend.api:app --reload --port 8000
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

---

## Summary

This project demonstrates a complete VLM-based applied AI prototype for fridge image understanding and recipe recommendation.

The final system includes VLM ingredient extraction, manual evaluation, confidence filtering, recipe retrieval, missing ingredient analysis, grocery guidance, and a React + FastAPI demo interface.