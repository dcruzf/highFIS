# API Reference

Welcome to the **highFIS** API Reference. This section contains the complete reference documentation for all public modules, classes, and helper functions in the library.

The `highFIS` codebase is structured to decouple PyTorch model definitions from high-level `scikit-learn` estimators and optimization drivers.

---

## 1. Top-Level Estimators

For most machine learning tasks, you will interact with the scikit-learn compatible estimator wrappers. These wrappers handle the fit-predict cycle, metric evaluations, validation splits, and persistence.

*   [`highfis.estimators`](estimators.md) — Estimators Reference: Scikit-learn compatible classifiers (e.g., `HTSKClassifier`, `DGTSKClassifier`) and regressors (e.g., `HTSKRegressor`, `DGTSKRegressor`).

---

## 2. Core Architectures

If you are developing custom neural architectures or need direct access to the raw PyTorch models, use the core model definitions:

*   [`highfis.models`](models.md) — Model Architectures: PyTorch `nn.Module` classes representing TSK model variants (e.g., `HTSKModel`, `LogTSKModel`, `DGTSKModel`).
*   [`highfis.base`](base.md) — Base TSK: Unified PyTorch training hooks and shared base classes.

---

## 3. Layer Primitives

highFIS builds neuro-fuzzy systems by combining custom PyTorch layer modules:

*   [`highfis.layers`](layers.md) — Layer Modules: Core layers including `FuzzificationLayer`, `RuleLayer`, and consequent layers.
*   [`highfis.memberships`](memberships.md) — Membership Functions: Individual membership functions (e.g., `GaussianMF`, `GaussianPiMF`, `CompositeGMF`).
*   [`highfis.t_norms`](t_norms.md) — T-Norms: Logical antecedent aggregation (e.g., Product, Minimum, Dombi, Yager, and adaptive softmin T-norms).
*   [`highfis.defuzzifiers`](defuzzifiers.md) — Defuzzifiers: Normalization engines that map rule activation weights to output weights.
*   [`highfis.gates`](gates.md) — Gating Modules: Structural pruning gates used for feature selection and rule extraction.

---

## 4. Utilities and Protocols

*   [`highfis.optim`](optimisers.md) — Optimizers & Trainers: Decoupled training drivers like `GradientTrainer`, `DGTrainer`, and `FSRETrainer`.
*   [`highfis.clustering`](clustering.md) — Clustering & Initialization: Algorithms for initializing fuzzy membership centers (e.g. k-means, grid, P-FRB).
*   [`highfis.persistence`](persistence.md) — Checkpoint Serialization: Versioned, secure checkpoint saving and loading utilities.
*   [`highfis.metrics`](metrics.md) — Evaluation Metrics: Custom statistical metrics for regression and classification validation.
*   [`highfis.protocols`](protocols.md) — Type Protocols: Structural typing interfaces ensuring API compatibility.
