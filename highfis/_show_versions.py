"""Report the software and hardware environment highFIS runs in."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch

from .version import __version__


def _cpu_name() -> str:
    """Return a human-readable CPU model, falling back to the machine architecture."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def _collect_versions() -> dict[str, str]:
    """Gather the fields printed by :func:`show_versions`, in display order."""
    return {
        "highfis": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "default dtype": str(torch.get_default_dtype()),
        "threads": str(torch.get_num_threads()),
        "cpu": _cpu_name(),
    }


def show_versions() -> None:
    """Print the versions and runtime settings that affect highFIS results.

    Lists the versions of highFIS, Python, PyTorch, NumPy and scikit-learn, the default
    floating-point type, the number of threads PyTorch uses and the CPU model. Include
    this output when reporting a bug or documenting how a result was produced.

    Examples:
        >>> import highfis
        >>> highfis.show_versions()  # doctest: +SKIP
    """
    info = _collect_versions()
    width = max(len(key) for key in info)
    sys.stdout.write("".join(f"{key:<{width}} : {value}\n" for key, value in info.items()))
