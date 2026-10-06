# Model Families

**highFIS** implements a diverse collection of concrete Takagi-Sugeno-Kang (TSK) neuro-fuzzy model architectures. These models are categorized below by their theoretical design, numerical stability properties, and structural sparsity capabilities.

Every model family exposes both a scikit-learn compatible **Classifier** (e.g., `HTSKClassifier`) and a **Regressor** (e.g., `HTSKRegressor`).

---

## What each family fixes

A family is defined by its membership function, the way it aggregates the membership
degrees of a rule, and the way it normalizes the rule firing strengths. These are fixed in
the named estimators. Only the generic TSK lets you choose them; see
[Building Blocks of the Generic TSK](../guides/building-blocks.md).

| Family | Membership function | Aggregation | Normalization | Use this when |
|---|---|---|---|---|
| TSK | Gaussian (selectable) | product (selectable) | sum (selectable) | You have few features, want a baseline, or want to study one building block at a time. |
| HTSK | Gaussian | geometric mean | softmax in log space | You have many features and the product saturates. |
| LogTSK | Gaussian | geometric mean | inverse log | Same setting as HTSK, with a normalization that does not depend on the scale of the firing strengths. |
| HDFIS | dimension-dependent Gaussian (prod), Gaussian (min) | product or minimum | sum | You have very many features and want to keep the product or the minimum T-norm. |
| DombiTSK | Gaussian with a positive lower bound | Dombi, fixed parameter | sum | You want a parametric T-norm with a parameter you set yourself. |
| ADMTSK | Gaussian with a positive lower bound | Dombi, parameter set from the dimension | sum | You want the Dombi T-norm adapted to the number of features. |
| AYATSK | composite exponential | Yager, parameter set from the dimension | sum | You want the Yager T-norm adapted to the number of features. |
| ADATSK | Gaussian | adaptive softmin | sum | You want a smooth minimum that stays stable as the number of features grows. |
| ADPTSK | Gaussian with a positive lower bound | adaptive double-parameter softmin | sum | Same setting as ADATSK, with a second parameter for more stable normalized rule weights. |
| FSRE-ADATSK | Gaussian | adaptive softmin | softmax in log space | You also want feature selection and rule extraction. |
| DG-TSK | Gaussian | gated layer | softmax in log space | You want features and rules pruned by gates during training. |
| DG-ALETSK | Gaussian | gated layer with adaptive Ln-Exp softmin | softmax in log space | Same goal as DG-TSK, with an aggregation designed for high-dimensional data. |
| MHTSK | Gaussian, on feature subsets | product, per head | sum | You have many features and want rules that each use a small subset of them. |

---

## 1. Baselines

These models implement standard, textbook fuzzy logic structures. They are ideal as simple baselines for low-dimensional problems.

*   [**TSK**](tsk-vanilla.md) — Generic TSK fuzzy system. By default the standard (vanilla) system with product antecedent aggregation and sum-based normalization; the membership function, T-norm and defuzzifier can be chosen.

---

## 2. High-Dimensional & Stable Models

Standard fuzzy inference suffers from the **saturation phenomenon** in high-dimensional spaces (rule weights underflow or overflow). These models introduce mathematical formulations designed specifically to maintain numerical stability on large feature sets.

*   [**HTSK**](htsk.md) — High-dimensional TSK system featuring geometric mean aggregation and log-space softmax normalization.
*   [**LogTSK**](logtsk.md) — Utilizes inverse-log normalization of log-domain rule weights to ensure stable inference under extreme dimensions.
*   [**HDFIS**](hdfis.md) — Implements both high-dimensional product-DMF (HDFIS-prod) and minimum frozen-antecedent (HDFIS-min) inference.

---

## 3. Parametric & Adaptive T-Norms

These models replace static T-norms (like standard product or minimum) with parametric or adaptive aggregation functions. Their shape parameters are either fixed hyperparameters or computed from the input dimensionality and the membership values, rather than trained by gradient descent.

*   [**DombiTSK**](dombitsk.md) — Parametric antecedent aggregation based on the Dombi T-norm with a fixed shape parameter $\lambda$.
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
