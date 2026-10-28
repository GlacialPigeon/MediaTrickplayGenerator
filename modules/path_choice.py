# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""Which path is the fastest for which kind of video, as the benchmark measured it, and how Generate uses it.

On a small SD video the CPU can beat the GPU (starting the GPU path costs more than it saves); on a big HEVC or AV1 video the GPU
wins by far. The benchmark compares the paths per kind of video (codec, bit depth, picture size, HDR type) and this module turns
that into a choice per kind: the CPU, or the fastest GPU path, and the CPU where no GPU path is at least `gpu_speed_threshold`
times faster. In acceleration mode "auto" Generate then goes straight to the chosen path for those kinds; kinds that were not
measured behave as before (GPU first, CPU as the fallback). Stored in the configuration as acceleration.path_choice:

    {"ffmpeg_hash": ..., "gpu_names": [...], "results": {"hevc 10-bit 4K SDR": {"best": "cuda", "speedup": 3.6}}}

The choice belongs to one FFmpeg build and one set of GPUs, like the other results.
"""
from __future__ import annotations

from typing import Any, Optional

from . import hdr_validation
from .library_sample import aligned_kinds, kind_key
from .localization import t, tp
from .models import VideoMeta

CPU = "cpu"


def kind_of(meta: VideoMeta) -> str: return kind_key(meta)


def from_benchmark(data: dict[str, Any], threshold: float) -> dict[str, dict[str, Any]]:
    """{kind: {"best": "cpu" or a GPU path, "speedup": CPU time / time of the best path, "files": n, "cpu": s, "paths": {path: s}}} from the rows
    of a benchmark. Only complete, comparable measurements count; a path that failed on any file of a kind is not offered for that kind."""
    per_kind: dict[str, dict[str, Any]] = {}
    rows = [r for r in data.get("rows", []) if r.get("kind")]
    for video in dict.fromkeys(r["video"] for r in rows):
        mine = [r for r in rows if r["video"] == video]
        cpu = [r for r in mine if r["backend"] == CPU]
        if not cpu or any(r["status"] != "PASS" for r in cpu): continue                    # nothing to compare with
        entry = per_kind.setdefault(mine[0]["kind"], {"files": 0, "cpu": 0.0, "paths": {}, "failed": set(), "video_seconds": 0.0})
        entry["files"] += 1
        entry["video_seconds"] += sum(r.get("duration_seconds") or 0.0 for r in cpu)
        entry["cpu"] += sum(r["extract_seconds"] + r["pack_seconds"] for r in cpu)
        for backend in data.get("backends", []):
            runs = [r for r in mine if r["backend"] == backend]
            if len(runs) != len(cpu) or any(r["status"] != "PASS" or not r.get("matching_frame_count") for r in runs): entry["failed"].add(backend)
            else: entry["paths"][backend] = entry["paths"].get(backend, 0.0) + sum(r["extract_seconds"] + r["pack_seconds"] for r in runs)
    results: dict[str, dict[str, Any]] = {}
    for kind, entry in per_kind.items():
        usable = {b: t for b, t in entry["paths"].items() if b not in entry["failed"]}
        best, speedup = CPU, 1.0
        for backend, seconds in usable.items():
            ratio = entry["cpu"] / max(seconds, 0.000001)
            if ratio >= threshold and ratio > speedup: best, speedup = backend, ratio
        best_seconds = entry["cpu"] if best == CPU else usable[best]
        results[kind] = {"best": best, "speedup": round(speedup, 2), "files": entry["files"], "cpu": round(entry["cpu"], 3),
                         "paths": {b: round(t, 3) for b, t in usable.items()},
                         "rate": round(entry["video_seconds"] / best_seconds, 2) if entry["video_seconds"] > 0 and best_seconds > 0 else 0.0}
    return results


def table_lines(results: dict[str, dict[str, Any]], backends: list[str]) -> list[str]:
    """The table of the fastest path per kind of video."""
    table = [[t("choice.col_kind"), t("choice.col_files"), "CPU (s)", *[f"{b} (s)" for b in backends], t("choice.col_fastest")]]
    shown = aligned_kinds(sorted(results))
    for kind in sorted(results):
        r = results[kind]
        table.append([shown[kind], str(r["files"]), f"{r['cpu']:.2f}", *[f"{r['paths'][b]:.2f}" if b in r["paths"] else t("choice.na") for b in backends],
                      "CPU" if r["best"] == CPU else f"{r['best']} ({r['speedup']:.2f}x)"])
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    out = [" | ".join(v.ljust(widths[i]) for i, v in enumerate(row)).rstrip() for row in table]
    out.insert(1, "-+-".join("-" * w for w in widths))
    return out


def lines(results: dict[str, dict[str, Any]], backends: list[str]) -> list[str]:
    if not results: return []
    return ["", "FASTEST PATH PER KIND OF VIDEO (a GPU path counts only when it is faster than the CPU by the benchmark threshold)", "",
            *table_lines(results, backends)]


def merged(existing: Any, results: dict[str, dict[str, Any]], ffmpeg_hash: Optional[str], gpu_names: list[str]) -> dict[str, Any]:
    """The new choices on top of the saved ones - unless the saved ones are from another FFmpeg or GPU: those are dropped."""
    kept: dict[str, Any] = {}
    if isinstance(existing, dict) and hdr_validation.same_system(existing, ffmpeg_hash, gpu_names):
        kept = {k: dict(v) for k, v in (existing.get("results") or {}).items() if isinstance(v, dict)}
    for kind, r in results.items(): kept[kind] = {"best": r["best"], "speedup": r["speedup"], "rate": r.get("rate", 0.0)}
    return {"ffmpeg_hash": ffmpeg_hash, "gpu_names": sorted(gpu_names), "results": kept}


def choice_for(stored: Any, ffmpeg_hash: Optional[str], gpu_names: list[str], meta: VideoMeta) -> Optional[str]:
    """"cpu" or the GPU path the benchmark found fastest for this kind of video; None when this kind was not measured (or on another FFmpeg / GPU)."""
    if not isinstance(stored, dict) or not hdr_validation.same_system(stored, ffmpeg_hash, gpu_names): return None
    entry = (stored.get("results") or {}).get(kind_of(meta))
    best = entry.get("best") if isinstance(entry, dict) else None
    return best if isinstance(best, str) and best else None


def rate_of(stored: Any, ffmpeg_hash: Optional[str], gpu_names: list[str], meta: VideoMeta) -> Optional[float]:
    """Seconds of video per second that one job gets through on the chosen path for this kind of video (all frames); None when unknown."""
    if not isinstance(stored, dict) or not hdr_validation.same_system(stored, ffmpeg_hash, gpu_names): return None
    entry = (stored.get("results") or {}).get(kind_of(meta))
    rate = entry.get("rate") if isinstance(entry, dict) else None
    return float(rate) if isinstance(rate, (int, float)) and rate > 0 else None


def benchmark_known(acceleration: dict[str, Any], ffmpeg_hash: Optional[str], gpu_names: list[str]) -> bool:
    """True when a benchmark has run to the end for this FFmpeg build and these GPUs, or has saved a choice for them."""
    seen = acceleration.get("benchmark_seen")
    if isinstance(seen, dict) and hdr_validation.same_system(seen, ffmpeg_hash, gpu_names): return True
    stored = acceleration.get("path_choice")
    return isinstance(stored, dict) and bool(stored.get("results")) and hdr_validation.same_system(stored, ffmpeg_hash, gpu_names)


def mark_benchmark_seen(acceleration: dict[str, Any], ffmpeg_hash: Optional[str], gpu_names: list[str]) -> None:
    acceleration["benchmark_seen"] = {"ffmpeg_hash": ffmpeg_hash, "gpu_names": sorted(gpu_names)}


def describe(stored: Any) -> list[str]:
    results = stored.get("results") if isinstance(stored, dict) else None
    if not results: return []
    return [f"{kind}: {'CPU' if r.get('best') == CPU else r.get('best')}" + ("" if r.get("best") == CPU else " " + t("choice.faster_than_cpu", speedup=f"{r.get('speedup', 0):.2f}")) for kind, r in sorted(results.items())]


def summary(results: dict[str, dict[str, Any]]) -> str:
    """"CPU for 5 kinds, GPU for 1 kind": what a choice per kind of video comes to."""
    cpu = sum(r["best"] == CPU for r in results.values())
    gpu = len(results) - cpu
    parts = [tp("choice.for_kinds", n, name=name) for name, n in (("CPU", cpu), ("GPU", gpu)) if n]
    return ", ".join(parts) or t("choice.nothing_chosen")
