# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import re
from typing import Callable, Optional

from .localization import english, lazy

# What FFmpeg prints after the real cause, once a filter or an encoder has failed.
_AFTERMATH = ("task finished", "terminating thread", "nothing was written", "could not open encoder", "error reinitializing",
              "failed to configure", "error initializing", "error opening")
_ADDRESS = re.compile(r"@ 0x[0-9a-fA-F]+|@ [0-9a-fA-F]{8,}")


def compact_ffmpeg_error(text: str) -> str:
    """The same text with lines that repeat (the same message from different decoder instances, only the address differs) shown once
    with a count, so one noisy file does not fill the log with hundreds of identical lines."""
    out: list[tuple[str, str, int]] = []                  # (line, line without addresses, repeats)
    for line in text.splitlines():
        key = _ADDRESS.sub("@", line).strip()
        if out and out[-1][1] == key and key: out[-1] = (out[-1][0], key, out[-1][2] + 1)
        else: out.append((line, key, 1))
    return "\n".join(line if count == 1 else f"{line}  [repeated {count} times]" for line, _key, count in out)


def error_cause(text: str) -> str:
    """The cause in one line: the first line FFmpeg printed that is not just the aftermath of another failure."""
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if not lines: return ""
    cause = next((l for l in lines if not any(marker in l.lower() for marker in _AFTERMATH)), lines[0])
    return re.sub(r"^(\[[^\]]*\]\s*)+", "", cause)[:200] or cause[:200]


def diagnostic_error_summary(error: str, *, localized: bool = False) -> str:
    """The cause in a few words. English (reports, logs and result files), or in the language of the person with `localized` (the screen)."""
    known = known_error_summary(error, localized=localized)
    if known: return known
    if error_cause(error): return error_cause(error)
    return _text(lazy("error.processing_failed"), localized)


def _text(message: Callable[[], str], localized: bool) -> str:
    if localized: return message()
    with english(): return message()


def known_error_summary(error: str, *, localized: bool = False) -> Optional[str]:
    """A plain explanation for the failures that are understood; None for the others."""
    low = error.lower()
    message: Optional[Callable[[], str]] = None
    if "source changed during generation" in low:
        message = lazy("error.source_changed")
    elif "source file disappeared" in low:
        message = lazy("error.source_disappeared")
    elif "impossible to convert between the formats" in low:
        message = lazy("error.cannot_convert")
    elif any(marker in low for marker in ("no device", "no capable devices", "cannot load nvcuda", "device setup failed")):
        message = lazy("error.device_unavailable")
    elif "unsupported" in low or "not supported" in low:
        message = lazy("error.unsupported")
    elif "no space left" in low or "disk full" in low:
        message = lazy("error.no_space")
    elif "being used by another process" in low or "winerror 32" in low:
        message = lazy("error.in_use")
    elif "permission denied" in low or "access is denied" in low:
        message = lazy("error.access_denied")
    return _text(message, localized) if message else None


_UNSUPPORTED_MARKERS = ("not supported", "unsupported", "no device", "device setup failed", "cannot load nvcuda", "cuda_error", "no capable devices",
                        "failed to initialise vaapi", "failed to initialize a qsv", "hardware does not support", "function not implemented",
                        "impossible to convert between the formats", "invalid hwframe context")


def classify_ffmpeg_error(text: str) -> str:
    """"unsupported" when FFmpeg says the hardware or the driver cannot do this (it will not work next time either), else "transient"."""
    low = text.lower()
    return "unsupported" if any(marker in low for marker in _UNSUPPORTED_MARKERS) else "transient"
