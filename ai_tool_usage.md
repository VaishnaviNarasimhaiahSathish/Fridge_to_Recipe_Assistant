## Entry #1 - Project Setup and VLM-First Planning

**Date:** 2026-04-29

**Team member(s):** Vaishnavi N.S

**AI Tool used:** ChatGPT

### Context

We had downloaded the Roboflow `fridge-detection-merged` dataset and needed to start the Fridge-to-Recipe Assistant project in a structured way. The project should follow a VLM-focused approach, where YOLO/object detection is only used later as a comparison baseline.

## Main Pipeline

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