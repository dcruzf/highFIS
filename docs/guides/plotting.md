# Diagnostic Plots

Every fitted estimator has a `plot` method. It draws one of seven diagnostic plots, chosen
with `kind`:

| `kind` | Shows | Needs data |
|---|---|---|
| `"memberships"` (default) | The learned fuzzy sets, one panel per feature | no |
| `"history"` | Loss or a metric per epoch | no |
| `"rules"` | The rule base as a table: the fuzzy set of each feature in each rule | no |
| `"rule_activation"` | Mean firing strength of each rule, by class when labels are given | `X`, optionally `y` |
| `"diagnostics"` | Confusion matrix and decision margin, or fit and residuals | `X` and `y` |
| `"feature_importance"` | The features that weigh most in the consequents | no |
| `"gates"` | Gates of the kept features and rules, with their thresholds (gated families) | no |

Each kind is also a method of its own (`plot_memberships`, `plot_history`, `plot_rules`,
`plot_rule_activation`, `plot_diagnostics`, `plot_feature_importance`, `plot_gates`),
which is easier to discover with autocompletion.

Plotting needs matplotlib, which is an optional dependency:

```bash
pip install highfis[plot]
```

In a notebook or at an interactive prompt, a plot is displayed as soon as it is called,
and nothing is returned: `clf.plot()` is the whole command. In a script nothing is
displayed, so that a batch run is never stopped by a window, and the plot returns the
`Axes` (single panel) or the `Figure` (several panels). `show` decides otherwise:

```python
clf.plot(kind="history")                     # displayed in a notebook, returned in a script
ax = clf.plot(kind="history", show=False)    # always returned, never displayed
clf.plot(kind="history", show=True)          # always displayed
```

Save a plot with `fig.savefig("plot.png")`, or `ax.figure.savefig("plot.png")` for a
single panel. The plots accept `ax` to draw on existing axes, and never change the global
settings of matplotlib. The examples below keep the figure, as a script does.

The examples below share one fitted model:

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)

scaler = MinMaxScaler().fit(X_fit)
X_fit, X_val, X_test = scaler.transform(X_fit), scaler.transform(X_val), scaler.transform(X_test)

clf = HTSKClassifier(n_mfs=3, epochs=100, random_state=0)
clf.fit(X_fit, y_fit, x_val=X_val, y_val=y_val, metrics=["accuracy"])

fig = clf.plot()                                              # memberships
ax = clf.plot(kind="history")
ax = clf.plot(kind="rule_activation", X=X_test, y=y_test)
fig = clf.plot(kind="diagnostics", X=X_test, y=y_test)
```

## Membership functions

```python
fig = clf.plot(kind="memberships")
```

![Learned membership functions of the six most important features](../assets/plots/memberships.png)

One panel per feature, one curve per fuzzy set, drawn from the trained model itself.

- The sets are named as in `rules_as_text()`: two, three and five sets are `low` to
  `high`. Pass `set_names=[...]` to use other names. A name has the same colour on every
  panel, and when every feature has the same sets one legend is drawn beside the panels.

- With more than `max_features` features (6 by default), the most important ones
  according to `feature_importance()` are shown. Pass `features=[...]` to choose them, by
  name or by column index.
- Panels are titled with the feature names when the model was fitted on a pandas
  `DataFrame`, and with `x1`, `x2`, ... otherwise.
- The horizontal range is taken from the fuzzy sets. Pass `X=` to span the data instead.
- Families that prune features (DG-TSK, DG-ALETSK, FSRE-ADATSK) show the features that
  survived.

What to look for: sets that overlap almost completely carry the same information; a set
much narrower than the others covers very few samples.

## Training history

```python
ax = clf.plot(kind="history")
ax = clf.plot(kind="history", metric="accuracy")
```

![Training and validation loss per epoch](../assets/plots/history.png)

The validation curve appears when a validation set was passed to `fit`, and the dashed
line marks the best epoch, whose weights the model keeps. For classifiers the best epoch
is the one with the highest validation accuracy, not the lowest validation loss, so
training can stop while the loss is still falling, as it does here. For regressors it is
the epoch with the lowest validation loss.

`metric` plots any metric requested in `fit(..., metrics=[...])`:

![Training and validation accuracy per epoch](../assets/plots/history-accuracy.png)

Families that train in phases get one panel per phase, on the same vertical scale. The
plot then returns the `Figure`, and `ax` takes one axes per phase:

![Training loss of a DG-ALETSK model across its two gradient phases](../assets/plots/history-phases.png)

What to look for: a training curve that keeps oscillating at the last epoch means the
model returned depends on where training happened to stop. Lower the learning rate or
use a scheduler.

## Rule base

```python
ax = clf.plot(kind="rules")
```

![The fuzzy set that each rule uses on each feature](../assets/plots/rules.png)

One row per rule and one column per feature. Each cell is the fuzzy set that the rule
uses on the feature, coloured from the lowest set (blue) to the highest (red) and, in
small rule bases, labelled with the initials of its name: `L`, `M`, `H`. It is the
complete antecedent part of the model in one figure, where `rules_as_text()` writes one
rule at a time. `features`, `max_features` and `set_names` work as for the membership
functions.

What to look for: rows that are almost equal describe rules that differ in one or two
features only; columns of a single colour are features on which the rules do not differ.

## Rule activation

```python
ax = clf.plot(kind="rule_activation", X=X_test, y=y_test)
```

![Mean rule activation by true class](../assets/plots/rule-activation.png)

Each cell is the mean normalized firing strength of a rule over the samples of a class.
Without `y`, or for a regressor, the plot is a bar per rule over all samples.

What to look for: in a model that uses its rules, each class is carried by one or a few
rules, as above. A plot where every cell has the same value means the rules do not
discriminate. That happens to the classical TSK when the product of many membership
degrees underflows: every rule gets the same weight and the model reduces to a single
linear model. Here is the same data fitted with `TSKClassifier`:

![Uniform rule activation of a saturated TSK model](../assets/plots/rule-activation-saturated.png)

Its accuracy is still high on this dataset, so the score alone does not reveal the
problem. The high-dimensional families exist to avoid it.

## Prediction diagnostics

```python
fig = clf.plot(kind="diagnostics", X=X_test, y=y_test)
```

![Confusion matrix and decision margin of the classifier](../assets/plots/diagnostics-classifier.png)

For a classifier, the left panel is the confusion matrix. The right panel shows the
decision margin of every sample, grouped by its true class: the probability given to the
true class minus the highest probability given to another class. A sample below zero is
misclassified, and samples just above zero are the ones the model is least sure about.

For a regressor, the panels are the predicted against the observed values and the
residuals:

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
x_scaler, y_scaler = MinMaxScaler().fit(X_train), MinMaxScaler().fit(y_train.reshape(-1, 1))

reg = HTSKRegressor(n_mfs=3, epochs=150, random_state=0)
reg.fit(x_scaler.transform(X_train), y_scaler.transform(y_train.reshape(-1, 1)).ravel())

fig = reg.plot(
    kind="diagnostics",
    X=x_scaler.transform(X_test),
    y=y_scaler.transform(y_test.reshape(-1, 1)).ravel(),
)
```

![Predicted against observed values and residuals of the regressor](../assets/plots/diagnostics-regressor.png)

What to look for: residuals that trend with the observed value, as here, mean the model
shrinks its predictions towards the mean.

## Feature importance

```python
ax = clf.plot(kind="feature_importance", top=8)
```

![The eight features with the largest importance](../assets/plots/feature-importance.png)

The bars are the values of `clf.feature_importance()`: the mean magnitude of the
consequent coefficients of each feature, normalized to sum to one, from the largest.
`top` sets how many features are drawn (20 by default, `None` for all). The features are
named as in `fit` when the data had names, for example the columns of a DataFrame.

What to look for: the inputs must be on a common scale for the coefficients to be
comparable. With thousands of features, a few bars far above the rest point to the
features to look at first.

## Gates

DG-TSK, DG-ALETSK and FSRE-ADATSK train a gate for every feature and for every rule, and
keep the features and rules whose gates pass a threshold:

```python
from highfis import DGALETSKClassifier

gated = DGALETSKClassifier(n_mfs=3, dg_epochs=40, finetune_epochs=60, random_state=0).fit(X_fit, y_fit)
fig = gated.plot(kind="gates")
```

![Gates of the features and rules that a DG-ALETSK model kept](../assets/plots/gates.png)

The left panel has one bar for each feature in `selected_features_` and the right panel
one for each rule of the fitted model; the dashed lines are the thresholds, whose values
are in the titles. The values
are those of `get_feature_gates()` and `get_rule_gates()`, at the end of the phase that
trained them. For FSRE-ADATSK, whose gate is an odd function, the magnitude is drawn.
The other families have no gates, and the plot raises a `ValueError` for them.

What to look for: gates just above the threshold belong to features or rules that a
slightly different threshold, or a shorter gate phase, would have removed.

## Colours

Every plot takes its colours from one theme, so that the figures of an analysis look
alike. The default colours are those of Okabe and Ito, which remain distinct for readers
with a colour vision deficiency.

```python
import highfis

highfis.set_plot_theme(
    colors=["#1b9e77", "#d95f02", "#7570b3"],  # fuzzy sets, curves, classes
    sequential="Greys",                         # confusion matrix, rule activation
    diverging="PuOr",                           # table of the rule base
)
highfis.reset_plot_theme()                      # back to the colours of the package
```

Only the arguments given are changed, and `highfis.get_plot_theme()` returns the theme in
use. For a single figure, pass `colors` or `cmap` to the plot:

```python
clf.plot(kind="rules", cmap="viridis")
clf.plot(kind="memberships", colors=["black", "grey", "silver"])
```

| Plot | `colors` | `cmap` |
|---|---|---|
| `"memberships"` | the fuzzy sets, from the lowest to the highest | |
| `"history"` | the training and the validation curve | |
| `"rules"` | | the cells, from the lowest set to the highest |
| `"rule_activation"` | the bars, without `y` | the table, with `y` |
| `"diagnostics"` | the classes, or the points of a regressor | the confusion matrix |
| `"feature_importance"`, `"gates"` | the bars | |

## Drawing on your own axes

```python
import matplotlib.pyplot as plt

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5))
clf.plot(kind="history", ax=left)
clf.plot(kind="rule_activation", X=X_test, y=y_test, ax=right)
```

Single-panel plots take one `Axes`. `"memberships"` takes one per feature shown, and
`"diagnostics"` and `"gates"` take two.

The same functions are available in `highfis.plotting` and take the estimator as their
first argument; see the [API reference](../api/plotting.md).
