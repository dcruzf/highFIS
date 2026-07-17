"""Root pytest config — shared globals for the CI-tested documentation examples.

The guides under ``docs/guides`` and the recipes under ``docs/cookbook`` are executed as
tests by ``pytest-markdown-docs``. That plugin calls ``pytest_markdown_docs_globals`` and
injects the returned dict into every code fence, so a snippet can reference a common
dataset (and a pre-fitted ``clf``) without rebuilding it — which keeps the prose short
while still running end to end and catching drift between the docs and the code.

The hook is only defined when the plugin is installed (the ``cookbook`` Hatch env). The
default test env does not have it, and declaring an impl for an unknown hook there would
abort collection, so the guard keeps ``hatch test`` unaffected. Living at the repo root
(rather than under ``docs/``) also keeps this helper out of the published site, which the
static-site builder would otherwise copy verbatim.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import Any


@lru_cache(maxsize=1)
def _doc_globals() -> dict[str, Any]:
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    from highfis import HTSKClassifier

    x, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, n_redundant=0, random_state=42
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    scaler = MinMaxScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)
    clf = HTSKClassifier(n_mfs=3, mf_init="kmeans", epochs=5, random_state=42)
    clf.fit(x_train_scaled, y_train)
    return {
        "X": x, "y": y,
        "X_train": x_train_scaled, "X_test": x_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": x_train_scaled, "X_test_scaled": x_test_scaled,
        "X_val": x_test_scaled, "y_val": y_test,
        "clf": clf,
    }


if find_spec("pytest_markdown_docs") is not None:

    def pytest_markdown_docs_globals() -> dict[str, Any]:
        return dict(_doc_globals())
