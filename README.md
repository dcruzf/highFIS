# highFIS — High-Dimensional Fuzzy Inference Systems

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/backend-PyTorch-orange)](https://pytorch.org)
[![License: GPLv3](https://img.shields.io/badge/license-GPLv3-green)](https://www.gnu.org/licenses/gpl-3.0)
[![scikit-learn API](https://img.shields.io/badge/API-scikit--learn-yellow)](https://scikit-learn.org)

> **highFIS** is a PyTorch-based Python library for constructing, training, and evaluating
> Takagi–Sugeno–Kang (TSK) fuzzy inference systems under high-dimensional input spaces.
> The library exposes a scikit-learn–compatible estimator API and provides modular neural-network
> layers that implement each stage of the TSK fuzzy inference pipeline as differentiable
> PyTorch modules, enabling end-to-end gradient-based optimisation.

---

## Motivation

TSK fuzzy systems are universal approximators whose transparency and expressiveness
have long made them attractive for classification and regression in scientific domains.
However, their classical formulation suffers from an *exponential complexity* in the number
of fuzzy rules: given $d$ input variables each partitioned into $s$ linguistic terms, the
Cartesian product rule base yields $s^d$ rules.
This **curse of dimensionality** renders standard TSK architectures computationally
intractable for problems with moderate-to-high input dimensionality (Cui *et al.*, IJCNN).

highFIS addresses this limitation by

1. implementing the **HTSK defuzzification** strategy (geometric-mean firing strength),
	 which provably decouples rule-activation magnitude from input dimensionality;
2. providing alternative **compact rule-base strategies** (CoCo and En-FRB) whose number
	 of rules scales linearly in $d$;
3. exposing a fully differentiable pipeline so that all parameters — antecedent centres,
	 spreads, and consequent weights — are learned jointly via backpropagation.

---

## Installation

```bash
pip install highFIS
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.23, PyTorch ≥ 2.3, scikit-learn ≥ 1.7.

---

## Theoretical Background

### TSK Fuzzy Inference

A TSK fuzzy system with $R$ rules infers an output $\hat{y}$ as the normalised weighted
sum of per-rule consequents:

$$
\hat{y} = \sum_{r=1}^{R} \bar{w}_r \, f_r(\mathbf{x}), \qquad
\bar{w}_r = \frac{w_r}{\sum_{j=1}^{R} w_j}
$$

where $f_r(\mathbf{x}) = \mathbf{a}_r^\top \mathbf{x} + b_r$ is a first-order linear
consequent and $w_r$ is the **firing strength** of rule $r$, computed as a t-norm
aggregation of the scalar membership degrees:

$$
w_r = \mathcal{T}\!\left(\mu_{r,1}(x_1),\, \mu_{r,2}(x_2),\, \ldots,\, \mu_{r,d}(x_d)\right)
$$

### HTSK Defuzzification

Standard product-t-norm firing strengths vanish exponentially with $d$, causing numerical
instability and gradient starvation.  HTSK replaces the product with the **geometric mean**:

$$
w_r^{\mathrm{HTSK}} = \left(\prod_{i=1}^{d} \mu_{r,i}(x_i)\right)^{\!\!1/d}
= \exp\!\left(\frac{1}{d}\sum_{i=1}^{d}\ln\mu_{r,i}(x_i)\right)
$$

This normalisation confines firing strengths to $(0,1]$ regardless of $d$, restoring
stable gradient flow and meaningful rule competition in high-dimensional settings.

### Compact Rule Bases

| Strategy | Symbol | Rule count | Description |
|---|---|---|---|
| **Cartesian** | `"cartesian"` | $s^d$ | Full combinatorial product; exact but exponential. |
| **CoCo** | `"coco"` | $s$ | Co-occurrence rule base; each rule activates identical terms across all inputs. Linear in $s$. |
| **En-FRB** | `"en"` | $s(2d+1)$ | Enhanced Fuzzy Rule Base; includes diagonal and adjacent rules. Linear in $s$ and $d$. |
| **Custom** | `"custom"` | user-defined | Arbitrary rule sets provided as index sequences. |

### Uniform Regularisation

To prevent mode collapse where a single rule dominates inference, highFIS provides an
optional **Uniform Regularisation (UR)** penalty:

$$
\mathcal{L}_{\mathrm{UR}} = \frac{1}{R}\sum_{r=1}^{R}\!\left(\bar{w}_r^{\,\text{avg}} - \tau\right)^2,
\qquad \tau = \frac{1}{R} \text{ (default)}
$$

The total training objective becomes $\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda\,\mathcal{L}_{\mathrm{UR}}$,
where $\lambda$ is controlled by the `ur_weight` hyperparameter.

---

## Architecture Overview

The library decomposes the TSK inference pipeline into four composable PyTorch modules:

```
Input x ──► MembershipLayer ──► RuleLayer ──► NormalizationLayer ──► ConsequentLayer ──► Output
							(antecedents)    (t-norm)       (softmax-like)          (linear TSK)
```

| Module | Class | Responsibility |
|---|---|---|
| Antecedent | `MembershipLayer` | Evaluates each MF $\mu_{r,i}(x_i)$ for all inputs and terms. |
| Firing strength | `RuleLayer` | Aggregates antecedent terms via a t-norm into $w_r$. |
| Normalisation | `NormalizationLayer` | Computes $\bar{w}_r = w_r / \sum_j w_j$. |
| Consequent | `ClassificationConsequentLayer` | Computes class logits via $\sum_r \bar{w}_r f_r(\mathbf{x})$. |

---

## Public API

### Membership Functions — `highfis.memberships`

```python
from highfis import GaussianMF, MembershipFunction
```

| Class | Formula | Parameters |
|---|---|---|
| `GaussianMF` | $\mu(x) = \exp\!\left(-\tfrac{(x-c)^2}{2\sigma^2}\right)$ | `mean` $c$, `sigma` $\sigma > 0$ |

`sigma` is reparameterised through a softplus transform to guarantee positivity during
unconstrained gradient descent.  `MembershipFunction` is the abstract base class for
user-defined differentiable membership functions.

### T-Norms — `highfis.t_norms`

```python
from highfis.t_norms import t_norm_prod, t_norm_min, t_norm_gmean
```

| Function | Expression | Notes |
|---|---|---|
| `t_norm_prod` | $\prod_i \mu_i$ | Standard product t-norm. |
| `t_norm_min` | $\min_i \mu_i$ | Łukasiewicz minimum t-norm. |
| `t_norm_gmean` | $(\prod_i \mu_i)^{1/d}$ | Geometric mean; recommended for high-dimensional data (HTSK). |

Custom t-norms can be injected as any callable with signature `(Tensor[batch, d]) → Tensor[batch]`.

### Layers — `highfis.layers`

```python
from highfis.layers import (
		MembershipLayer,
		RuleLayer,
		NormalizationLayer,
		ClassificationConsequentLayer,
)
```

All layers are `torch.nn.Module` subclasses and participate in standard PyTorch training loops.
`RuleLayer` accepts `rule_base ∈ {"cartesian", "coco", "en", "custom"}` and an optional
`t_norm_fn` callable for custom aggregation strategies.

### Model — `highfis.models.HTSKClassifier`

A self-contained `nn.Module` that assembles the full four-stage TSK pipeline:

```python
from highfis import HTSKClassifier, GaussianMF

input_mfs = {
		"x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
		"x2": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}

model = HTSKClassifier(
		input_mfs=input_mfs,
		n_classes=3,
		rule_base="en",       # Enhanced Fuzzy Rule Base
		t_norm="gmean",       # HTSK geometric-mean aggregation
)

history = model.fit(x_train, y_train, epochs=300, learning_rate=1e-3, ur_weight=0.01)
logits   = model.forward(x_test)
proba    = model.predict_proba(x_test)
labels   = model.predict(x_test)
norm_w   = model.forward_antecedents(x_test)  # normalised rule strengths for inspection
```

**Key `fit` arguments:**

| Argument | Type | Description |
|---|---|---|
| `epochs` | `int` | Number of gradient-descent epochs (default: 200). |
| `learning_rate` | `float` | Adam optimiser step size (default: 1e-3). |
| `batch_size` | `int \| None` | Mini-batch size; `None` for full-batch training. |
| `ur_weight` | `float` | Weight $\lambda$ of the uniform regularisation penalty (default: 0.0). |
| `ur_target` | `float \| None` | Target activation $\tau$; defaults to $1/R$. |
| `criterion` | `Callable \| None` | Loss function; defaults to MSE on one-hot targets. |
| `optimizer` | `Optimizer \| None` | Custom PyTorch optimiser; defaults to Adam. |
| `consequent_batch_norm` | `bool` | Apply batch normalisation to consequent inputs (default: False). |

### Scikit-learn Estimator — `highfis.estimators.HTSKClassifierEstimator`

```python
from highfis import HTSKClassifierEstimator, InputConfig
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
		("scaler", StandardScaler()),
		("clf", HTSKClassifierEstimator(
				n_mfs=3,
				rule_base="en",
				epochs=300,
				learning_rate=5e-4,
				ur_weight=0.005,
				random_state=42,
		)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

For per-feature control, pass a list of `InputConfig` objects:

```python
from highfis import InputConfig, HTSKClassifierEstimator

configs = [
		InputConfig(name="age",    n_mfs=3, overlap=0.5, margin=0.10),
		InputConfig(name="income", n_mfs=5, overlap=0.4, margin=0.05),
]

clf = HTSKClassifierEstimator(input_configs=configs, epochs=500)
```

`InputConfig` parameters:

| Field | Default | Description |
|---|---|---|
| `n_mfs` | 3 | Number of Gaussian terms per input. |
| `overlap` | 0.5 | Overlap factor controlling MF width: $\sigma = \tfrac{\Delta(1+\text{overlap})}{2}$. |
| `margin` | 0.10 | Fractional padding added beyond the observed data range before grid initialisation. |

Membership function centres are initialised on a uniform grid over the padded input range.
`HTSKClassifierEstimator` is fully compatible with scikit-learn utilities including
`cross_val_score`, `GridSearchCV`, and `Pipeline`.

---

## Design Principles

- **Full differentiability.** All antecedent and consequent parameters are `nn.Parameter`
	objects trained jointly by backpropagation; no alternating least-squares or EM steps.
- **Positivity-constrained parameters.** Gaussian spreads are reparameterised through
	softplus to guarantee $\sigma > 0$ throughout training without projection or clipping.
- **Modular composition.** Each inference stage is an independent `nn.Module`.
	Custom membership functions, t-norms, and rule bases can be substituted without
	modifying any other component.
- **scikit-learn interoperability.** `HTSKClassifierEstimator` inherits from
	`BaseEstimator` and `ClassifierMixin`, providing `fit`, `predict`, `predict_proba`,
	`score`, and full pipeline/grid-search compatibility.

---

## Citation

If you use highFIS in academic work, please cite the relevant HTSK paper:

```bibtex
@inproceedings{cui2020curse,
	title     = {Curse of Dimensionality for {TSK} Fuzzy Neural Networks:
							 Explanation and Solutions},
	author    = {Cui, Yifan and others},
	booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
	year      = {2020}
}
```

---

## License

highFIS is distributed under the **GNU General Public License v3** (GPLv3).
See [LICENSE](LICENSE) for the full terms.
