"""FSRE-ADATSK reports the rule base it ends with, not the one it starts from."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from highfis import FSREADATSKClassifier, FSREADATSKRegressor


def _fit(cls: type, **params: Any) -> tuple[Any, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 4)).astype("float32")
    y = (x[:, 0] > 0).astype(int) if cls is FSREADATSKClassifier else (x[:, 0] - x[:, 1]).astype("float32")
    return cls(fs_epochs=2, re_epochs=2, finetune_epochs=2, random_state=0, **params).fit(x, y), x


@pytest.mark.parametrize("cls", [FSREADATSKClassifier, FSREADATSKRegressor])
def test_every_description_names_the_enhanced_rule_base(cls: type) -> None:
    est, _ = _fit(cls)

    report = est.summary(show=False)

    assert report.data["rule_base"] == "en"
    assert "(rule base: en)" in report
    assert est.rule_base_ == "en"
    assert est.inspect()["rule_base"] == "en"


@pytest.mark.parametrize("cls", [FSREADATSKClassifier, FSREADATSKRegressor])
@pytest.mark.parametrize("use_en_frb", [False, True])
def test_a_saved_model_keeps_its_rule_base_and_its_rules(cls: type, use_en_frb: bool, tmp_path: Path) -> None:
    est, x = _fit(cls, use_en_frb=use_en_frb)
    path = tmp_path / "model.pt"

    est.save(str(path))
    loaded = cls.load(str(path))

    assert loaded.rule_base_ == "en"
    assert "(rule base: en)" in loaded.summary(show=False)
    assert loaded.model_.n_rules == est.model_.n_rules
    np.testing.assert_allclose(loaded.predict(x), est.predict(x))
