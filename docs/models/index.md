# Model Families

**highFIS** implements a diverse collection of concrete Takagi-Sugeno-Kang (TSK) neuro-fuzzy model architectures. These models are categorized below by their theoretical design, numerical stability properties, and structural sparsity capabilities.

Every model family exposes both a scikit-learn compatible **Classifier** (e.g., `HTSKClassifier`) and a **Regressor** (e.g., `HTSKRegressor`).

---

## 1. Baselines

These models implement standard, textbook fuzzy logic structures. They are ideal as simple baselines for low-dimensional problems.

*   [**TSK (Vanilla)**](tsk-vanilla.md) — Standard TSK fuzzy system with product antecedent aggregation and sum-based normalization.

---

## 2. High-Dimensional & Stable Models

Standard fuzzy inference suffers from the **saturation phenomenon** in high-dimensional spaces (rule weights underflow or overflow). These models introduce mathematical formulations designed specifically to maintain numerical stability on large feature sets.

*   [**HTSK**](htsk.md) — High-dimensional TSK system featuring geometric mean aggregation and log-space softmax normalization.
*   [**LogTSK**](logtsk.md) — Utilizes inverse-log normalization of log-domain rule weights to ensure stable inference under extreme dimensions.
*   [**HDFIS**](hdfis.md) — Implements both high-dimensional product-DMF (HDFIS-prod) and minimum frozen-antecedent (HDFIS-min) inference.

---

## 3. Parametric & Adaptive T-Norms

These models replace static T-norms (like standard product or minimum) with parametric or adaptive aggregation functions whose shape parameters are trained via gradient descent.

*   [**DombiTSK**](dombitsk.md) — Parametric antecedent aggregation based on the Dombi T-norm with a learnable shape parameter $\lambda$.
*   [**ADMTSK**](admtsk.md) — Adaptive Dombi TSK utilizing dimension-dependent Gaussian membership functions.
*   [**AYATSK**](ayatsk.md) — Flexible antecedent aggregation utilizing the Yager T-norm.
*   [**ADATSK**](adatsk.md) — Adaptive softmin aggregation offering dynamic rule weight scaling.
*   [**ADPTSK**](adptsk.md) — Double-parameter adaptive softmin aggregation for enhanced weight stabilization.

---

## 4. Sparse & Gated Models (Interpretability)

These architectures embed structural gate parameters to perform feature selection (identifying key inputs) and rule extraction (identifying key decision paths) during the training phase.

*   [**FSRE-ADATSK**](fsre-adatsk.md) — Features embedded feature selection and rule extraction gates in an adaptive softmin pipeline.
*   [**DGTSK**](dg-tsk.md) — Double-gated TSK utilizing separate structural gates to prune features and rules.
*   [**DGALETSK**](dg-aletsk.md) — Combines double-gated pruning with adaptive Ln-Exp softmin aggregation.
*   [**MHTSK**](mhtsk.md) — Multihead subantecedents designed for sparse high-dimensional rule extraction.
