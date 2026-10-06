"""Tests for the environment report."""

from __future__ import annotations

import pytest
import torch

import highfis
from highfis import _show_versions


def _parse(output: str) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in (line.split(" : ", 1) for line in output.splitlines())}


def test_show_versions_reports_every_field(capsys: pytest.CaptureFixture[str]) -> None:
    highfis.show_versions()

    fields = _parse(capsys.readouterr().out)

    assert list(fields) == [
        "highfis",
        "python",
        "platform",
        "torch",
        "numpy",
        "scikit-learn",
        "default dtype",
        "threads",
        "cpu",
    ]
    assert fields["highfis"] == highfis.__version__
    assert fields["torch"] == torch.__version__
    assert fields["threads"] == str(torch.get_num_threads())
    assert all(fields.values())


def test_show_versions_follows_the_default_dtype(capsys: pytest.CaptureFixture[str]) -> None:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        highfis.show_versions()
    finally:
        torch.set_default_dtype(previous)

    assert _parse(capsys.readouterr().out)["default dtype"] == "torch.float64"


def test_cpu_name_falls_back_without_procfs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_show_versions.Path, "is_file", lambda self: False)
    monkeypatch.setattr(_show_versions.platform, "processor", lambda: "")
    monkeypatch.setattr(_show_versions.platform, "machine", lambda: "x86_64")

    assert _show_versions._cpu_name() == "x86_64"
