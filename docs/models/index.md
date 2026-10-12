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

## Numerical Stability and the Published Models

highFIS computes several quantities in another way than the articles write them, and
departs from the articles in a few deliberate places. The two are different things.

**Same model, other arithmetic.** These changes compute the quantity that the article
defines; the result is the same wherever the article's own formula can be evaluated,
and remains defined where that formula would overflow or underflow.

| Quantity | In the article | In highFIS |
|---|---|---|
| Normalized product of HDFIS-prod | product of the membership degrees, in double precision | `softmax(D log(geometric mean))`, exact in single precision too |
| Ada-softmin, ADP-softmin | powers of the membership degrees, bounded for double precision | sums in the logarithmic domain |
| Dombi and Yager T-norms | power means | power means relative to the largest term |
| Normalization by the sum | division by the sum | the same, with strengths bounded below by the smallest positive number |
| Membership degrees | one function per fuzzy set | one batched operation |

**Positive spreads.** A spread is kept positive through a softplus transformation of
the trained parameter. The membership function is the one of the article; the path of
the optimizer differs from training the spread directly.

**Deliberate differences from the articles.** These do change the model or its
training, and each is stated on the page of the family:

| Difference | Families |
|---|---|
| Normalization of the consequent inputs (batch normalization) by default | ADATSK, FSRE-ADATSK |
| Regressors whose consequents start at zero start from the mean of the target | ADATSK, FSRE-ADATSK, HDFIS, MHTSK |
| Mean squared error where the article sums over the classes and halves (a constant factor under Adam) | ADMTSK, DombiTSK, AYATSK, ADPTSK, MHTSK |
| AdamW with a weight decay of `1e-8` where the article has Adam | TSK, HTSK, LogTSK, HDFIS, MHTSK |
| Initial spreads centred on the spread of the cluster (`sigma_init="constant"` gives the article's) | TSK, HTSK, LogTSK |
| Antecedents trained and number of heads from the feature coverage below 1000 features | MHTSK |
| Clustering with five rules by default instead of the partition of the article | DombiTSK |
| Regressors of families whose article only treats classification | all but HDFIS |

**Hyperparameters are not part of the model.** The learning rate, the number of
epochs and the batch size are often not stated in the articles, and where they are
they were chosen for the datasets of the article. The defaults of highFIS are a
starting point; see [The Defaults Are a Starting Point](../guides/tuning.md). The pages
under [Reproductions](../reproductions/index.md) give, for each article, the settings
that reproduce its results.

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
