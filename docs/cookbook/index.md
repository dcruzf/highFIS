# Cookbook

Short, task-focused recipes you can copy and run. Each recipe is self-contained and
uses small, seeded datasets so it runs in seconds. The code blocks are executed in
CI to keep them working; the `text` blocks below them show illustrative output and
may vary slightly across platforms and dependency versions. For conceptual
background see the [User Guides](../guides/tuning.md); for the full API see the
[API Reference](../api/index.md).

- [Quickstart: train, predict, and score](quickstart.md) — the basic workflow plus
  scikit-learn `Pipeline`, `cross_val_score`, and `GridSearchCV`.
- [Choosing a model](choosing-a-model.md) — the TSK families and how to swap between
  them.
- [Feature selection on high-dimensional data](high-dimensional-feature-selection.md)
  — embedded feature selection and rule extraction with `DGTSKClassifier`.
- [Persistence and the MF cache](persistence-and-cache.md) — `save`/`load` and
  managing the membership-function cache.
- [Inspecting a trained model](interpretability.md) — rules, membership functions,
  activations, and feature importance.
