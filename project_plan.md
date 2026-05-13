# Fridge-to-Recipe Assistant - Project Plan

## Goal
Build a VLM-focused assistant that identifies ingredients from fridge images and recommends recipes based on the detected ingredients.

## Main Approach
The main pipeline uses a Vision-Language Model for ingredient recognition. Object detection models such as YOLO are used only as a comparison baseline.

## Planned Pipeline
1. Dataset exploration
2. VLM-based ingredient extraction
3. Ingredient normalization
4. Recipe retrieval / generation
5. Recipe ranking
6. Evaluation
7. Web interface / demo

## Evaluation Plan
We will evaluate:
- Ingredient recognition quality
- Hallucinated ingredients
- Missed ingredients
- Recipe relevance
- Comparison between VLM-only and YOLO-assisted outputs

## Current Dataset
Roboflow fridge detection dataset:
- Source: Roboflow Universe
- Synthetically generate few samples
- Capture a few samples by ourselves