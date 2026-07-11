# Feature selection on high-dimensional data

`DGTSKClassifier` performs embedded **feature selection** and **rule extraction**
while it trains: it starts from a point-based rule base and prunes uninformative
features and rules via learnable gates. This is useful when you have many more
features than samples.

The example below builds a small dataset with only a handful of informative
features among many, then inspects how many the model kept.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from highfis import DGTSKClassifier

X, y = make_classification(
    n_samples=80,
    n_features=40,
    n_informative=6,
    n_redundant=4,
    n_classes=2,
    random_state=0,
)
X = MinMaxScaler().fit_transform(X).astype("float32")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

clf = DGTSKClassifier(
    n_mfs=3,
    dg_epochs=5,
    finetune_epochs=10,
    learning_rate=0.05,
    random_state=0,
)
clf.fit(X_tr, y_tr)

print("features kept:", clf.model_.n_inputs, "of", X.shape[1])
print("rules kept   :", clf.model_.n_rules)
print("test accuracy:", round(clf.score(X_te, y_te), 3))
```

```text
features kept: 15 of 40
rules kept   : 2
test accuracy: 0.75
```

`clf.model_.n_inputs` and `clf.model_.n_rules` reflect the pruned model. The same
workflow applies to `FSREADATSKClassifier`, which runs feature selection and rule
extraction in separate phases.
