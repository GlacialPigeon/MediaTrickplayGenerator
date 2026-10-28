# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import os
from pathlib import Path

from .constants import KNOWN_NONVIDEO_EXTENSIONS


# Quote pairs a person may wrap a pasted or dropped path in (straight and typographic, nested ones too).
_QUOTE_PAIRS = {'"': '"', "'": "'", "\u201c": "\u201d", "\u2018": "\u2019"}


def remove_outer_quotes(value: object) -> str:
    """The text without surrounding whitespace and without matching quotes around it ("path", 'path', “path”, ‘path’, nested)."""
    text = "" if value is None else str(value).strip()
    while len(text) >= 2 and text[0] in _QUOTE_PAIRS and text[-1] == _QUOTE_PAIRS[text[0]]:
        text = text[1:-1].strip()
    return text


def normalize_path(path: Path | str) -> str:
    """Comparison key for a path. os.path supplies the platform's case and separator rules."""
    try: p = Path(path).expanduser().resolve(strict=False)
    except Exception: p = Path(path).expanduser().absolute()
    return os.path.normcase(os.path.normpath(str(p)))


def is_relative_to_path(child: str, parent: str) -> bool:
    try:
        common = os.path.commonpath([child, parent])
        return os.path.normcase(common) == os.path.normcase(parent)
    except ValueError: return False


def could_be_video(name: str) -> bool:
    """The scanner probes every file that is not clearly something else, so anything with an unknown extension may be a video."""
    return os.path.splitext(name)[1].casefold() not in KNOWN_NONVIDEO_EXTENSIONS
