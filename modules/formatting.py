# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import math
import os
from pathlib import Path


def human_bytes(value: float | int) -> str:
    n = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    for unit in units:
        if abs(n) < 1024.0 or unit == units[-1]:
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.2f} PiB"


def human_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds): return "N/A"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def display_path(path: "Path | str") -> str:
    """A path as shown on the screen: relative to the program folder when it is inside it (`diagnostics\\benchmark\\2026-10-27_...`), else in full.
    Media files and anything else outside the program folder keep their whole path. Files and logs always keep the full path."""
    from . import constants                                  # read when called: the program folder is not known when this module loads
    try:
        full, root = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(constants.APP_ROOT))
        if full == root: return "."
        if full.startswith(root + os.sep): return os.path.relpath(path, constants.APP_ROOT)
    except (OSError, ValueError):
        pass
    return str(path)
