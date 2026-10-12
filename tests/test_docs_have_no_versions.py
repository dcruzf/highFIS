"""The documentation is versioned with the package, so its text must not name versions."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"(?<![\w.])\d+\.\d{2}\.\d+(?![\w.])")


def _files() -> list[Path]:
    docs = sorted((ROOT / "docs").rglob("*.md")) if (ROOT / "docs").is_dir() else []
    code = [path for path in sorted((ROOT / "highfis").rglob("*.py")) if path.name != "version.py"]
    return docs + code


@pytest.mark.parametrize("path", _files(), ids=lambda path: str(path.relative_to(ROOT)))
def test_no_package_version_in_the_text(path: Path) -> None:
    found = VERSION.findall(path.read_text(encoding="utf-8"))
    assert not found, f"{path.relative_to(ROOT)} names a version: {sorted(set(found))}"
