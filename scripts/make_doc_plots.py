"""Regenerate the figures of docs/guides/plotting.md.

Run from the repository root:  hatch run python scripts/make_doc_plots.py
The code mirrors the examples on that page, so the page and its figures stay in step.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from sklearn.datasets import load_diabetes, load_wine  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from highfis import DGALETSKClassifier, HTSKClassifier, HTSKRegressor, TSKClassifier  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "plots"
DPI = 110


def save(artist: object, name: str) -> None:
    figure = getattr(artist, "figure", artist)
    figure.savefig(OUT / name, dpi=DPI)  # type: ignore[attr-defined]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)
    scaler = MinMaxScaler().fit(X_fit)
    X_fit, X_val, X_test = scaler.transform(X_fit), scaler.transform(X_val), scaler.transform(X_test)

    clf = HTSKClassifier(n_mfs=3, epochs=100, random_state=0)
    clf.fit(X_fit, y_fit, x_val=X_val, y_val=y_val, metrics=["accuracy"])

    save(clf.plot(), "memberships.png")
    save(clf.plot(kind="history"), "history.png")
    save(clf.plot(kind="history", metric="accuracy"), "history-accuracy.png")
    save(clf.plot(kind="rule_activation", X=X_test, y=y_test), "rule-activation.png")
    save(clf.plot(kind="diagnostics", X=X_test, y=y_test), "diagnostics-classifier.png")

    saturated = TSKClassifier(n_mfs=3, epochs=100, random_state=0).fit(X_fit, y_fit)
    save(saturated.plot(kind="rule_activation", X=X_test, y=y_test), "rule-activation-saturated.png")

    gated = DGALETSKClassifier(n_mfs=3, dg_epochs=40, finetune_epochs=60, random_state=0).fit(X_fit, y_fit)
    save(gated.plot(kind="history"), "history-phases.png")

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    x_scaler, y_scaler = MinMaxScaler().fit(X_train), MinMaxScaler().fit(y_train.reshape(-1, 1))
    reg = HTSKRegressor(n_mfs=3, epochs=150, random_state=0)
    reg.fit(x_scaler.transform(X_train), y_scaler.transform(y_train.reshape(-1, 1)).ravel())
    save(
        reg.plot(kind="diagnostics", X=x_scaler.transform(X_test), y=y_scaler.transform(y_test.reshape(-1, 1)).ravel()),
        "diagnostics-regressor.png",
    )


if __name__ == "__main__":
    main()
