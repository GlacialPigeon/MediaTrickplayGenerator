# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""What the HDR check found out about the GPU paths, and how Generate uses it.

The person rates, for every kind of HDR and every GPU path, whether the colors look right. A path rated badly for a kind is
not used for that kind when generating (the CPU is used instead); everything not tested keeps working as before.
Stored in the configuration as acceleration.hdr_validation:

    {"ffmpeg_hash": ..., "gpu_names": [...], "results": {"cuda": {"HDR10": "PASS", "Dolby Vision P8": "FAIL"}}}

The results belong to one FFmpeg build and one set of GPUs; with another build or another card they are not applied.
"""
from __future__ import annotations

from typing import Any, Optional

from .localization import t

PASS, FAIL = "PASS", "FAIL"
_ASSESSMENT_RESULT = {"ACCEPTABLE": PASS, "TOO_BRIGHT": FAIL, "TOO_DARK": FAIL, "WRONG_COLORS": FAIL}   # PENDING decides nothing

Results = dict[str, dict[str, str]]          # backend -> HDR kind -> PASS | FAIL


def review_results(data: dict[str, Any]) -> Results:
    """The decided results of an HDR report: its GPU rows joined with the person's assessments."""
    results: Results = {}
    for row in data.get("rows", []):
        kind, backend = row.get("hdr_kind"), row.get("backend")
        if row.get("status") != "PASS" or not kind or not backend or backend == "cpu": continue
        assessment = data.get("visual", {}).get(f"{row.get('hdr_label')} | {backend}")
        if assessment in _ASSESSMENT_RESULT: results.setdefault(backend, {})[kind] = _ASSESSMENT_RESULT[assessment]
    return results


def failing(results: Results) -> list[tuple[str, str]]:
    return [(backend, kind) for backend, kinds in results.items() for kind, result in kinds.items() if result == FAIL]


def describe(results: Results) -> list[str]:
    return [f"{backend}: " + ", ".join(f"{kind} {t('hdr.ok') if result == PASS else t('hdr.not_ok')}" for kind, result in kinds.items())
            for backend, kinds in results.items()]


def same_system(validation_or_report: dict[str, Any], ffmpeg_hash: Optional[str], gpu_names: list[str]) -> bool:
    return (validation_or_report.get("ffmpeg_hash") == ffmpeg_hash
            and sorted(validation_or_report.get("gpu_names") or []) == sorted(gpu_names))


def merge(existing: Any, new: Results, ffmpeg_hash: Optional[str], gpu_names: list[str]) -> dict[str, Any]:
    """The new results on top of the saved ones - unless the saved ones are from another FFmpeg or GPU: those are dropped."""
    results: Results = {}
    if isinstance(existing, dict) and same_system(existing, ffmpeg_hash, gpu_names):
        results = {backend: dict(kinds) for backend, kinds in (existing.get("results") or {}).items() if isinstance(kinds, dict)}
    for backend, kinds in new.items(): results.setdefault(backend, {}).update(kinds)
    return {"ffmpeg_hash": ffmpeg_hash, "gpu_names": sorted(gpu_names), "results": results}


def is_blocked(validation: Any, ffmpeg_hash: Optional[str], gpu_names: list[str], backend: str, kind: str) -> bool:
    """True when this GPU path was rated badly for this kind of HDR on this FFmpeg / GPU."""
    if not isinstance(validation, dict) or not kind or not same_system(validation, ffmpeg_hash, gpu_names): return False
    kinds: Optional[Any] = (validation.get("results") or {}).get(backend)
    return isinstance(kinds, dict) and kinds.get(kind) == FAIL
