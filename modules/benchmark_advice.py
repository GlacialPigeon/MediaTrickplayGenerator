# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""What the benchmark suggests changing in the settings, and how the extra measurements are shown.

Besides CPU against GPU the benchmark measures, for every file, the fastest path once more with key frames only (A/B), and
how much work the GPU gets through with 1, 2, 3 ... videos at the same time (on a mix of the videos where a GPU path won, so it is
the work of this computer and this library that is measured), and what kind of disks the library is on and how many videos are
processed at once on each. From that it can suggest three settings:
`trickplay.keyframe_only`, `trickplay.keyframe_repeat_limit_percent` and `acceleration.max_gpu_jobs`. Nothing is changed
without the person's confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import path_choice
from .localization import t, tp
from .prompts import fit, format_labelled

KEYFRAME_MIN_GAIN = 1.2      # key frames only must make the tested files at least this much faster (including the redo of sparse files)
MIN_KEYFRAME_FILES = 3       # fewer files say too little
PARALLEL_MIN_GAIN = 1.15     # one more GPU job must bring at least this much more work done per second
DEFAULT_REPEAT_LIMIT = 10.0
MAX_MOVES = 6                # kinds listed under a changed choice


@dataclass
class Advice:
    section: str
    key: str
    label: str
    current: Any
    proposed: Any
    reason: str
    payload: Any = None
    details: Any = ()            # lines under the change: what moves where


def path_results(data: dict[str, Any], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return path_choice.from_benchmark(data, float(config.get("acceleration", {}).get("gpu_speed_threshold", 1.15)))


def path_lines(data: dict[str, Any], config: dict[str, Any]) -> list[str]:
    return path_choice.lines(path_results(data, config), list(data.get("backends", [])))


def _yes_no(value: Any) -> str: return t("common.on") if value else t("common.off")


def _name(path: Any) -> str: return "CPU" if path == path_choice.CPU else str(path)


def keyframe_summary(data: dict[str, Any], limit: float) -> dict[str, Any]:
    rows = data.get("keyframe_rows", [])
    ok = [r for r in rows if r["status"] == "PASS"]
    sparse = [r for r in ok if limit > 0 and r["share"] > limit]          # Generate would redo these with every frame
    failed = [r for r in rows if r["status"] != "PASS"]
    all_time = sum(r["all_seconds"] for r in rows)
    effective = (sum(r["keyframe_seconds"] + (r["all_seconds"] if r in sparse else 0) for r in ok)
                 + sum(r.get("keyframe_seconds", 0) + r["all_seconds"] for r in failed))
    return {"files": len(rows), "ok": len(ok), "sparse": len(sparse), "failed": len(failed), "all_seconds": all_time,
            "effective_seconds": effective, "gain": all_time / effective if effective > 0 else 0.0}


def keyframe_detail(summary: dict[str, Any]) -> str:
    """"2.63x faster on 10 files; 1 would be redone with every frame, 1 cannot use key frames" (only what applies)."""
    gain = f"{summary['gain']:.2f}"
    text = tp("advice.detail_faster", summary["files"], gain=gain) if summary["gain"] >= 1 else tp("advice.detail_slower", summary["files"], gain=gain)
    extras = []
    if summary["sparse"]: extras.append(tp("advice.sparse", summary["sparse"]))
    if summary["failed"]: extras.append(tp("advice.failed", summary["failed"]))
    return text + ("; " + ", ".join(extras) if extras else "")


def keyframe_lines(data: dict[str, Any], limit: float) -> list[str]:
    rows = data.get("keyframe_rows", [])
    if not rows: return []
    table = [["File", "Path", "All frames (s)", "Key frames (s)", "Faster", "Repeated", "Result"]]
    for row in rows:
        if row["status"] == "PASS":
            share = row["share"]
            note = "key frames too far apart: Generate would redo it" if limit > 0 and share > limit else "ok"
            table.append([Path(row["video"]).name, row["backend"], f"{row['all_seconds']:.2f}", f"{row['keyframe_seconds']:.2f}",
                          f"{row['all_seconds'] / max(row['keyframe_seconds'], 0.000001):.2f}x", f"{share:.0f}%", note])
        else:
            table.append([Path(row["video"]).name, row["backend"], f"{row['all_seconds']:.2f}", "-", "-", "-",
                          "not possible for this file: " + (row.get("error_summary") or "failed")])
    widths = [max(len(r[i]) for r in table) for i in range(len(table[0]))]
    lines = [" | ".join(v.ljust(widths[i]) for i, v in enumerate(r)) for r in table]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    summary = keyframe_summary(data, limit)
    total = (f"Together: all frames {summary['all_seconds']:.1f} s, key frames only {summary['effective_seconds']:.1f} s "
             f"(including the redo of files whose key frames are too far apart) = {summary['gain']:.2f}x") if summary["effective_seconds"] > 0 else ""
    return ["", "KEY FRAMES ONLY vs ALL FRAMES (the fastest path of every file, run once more with key frames only)",
            "Repeated = share of thumbnails that are the same picture as the one before (key frames further apart than the interval).",
            "", *lines, "", *([total] if total else [])]


def parallel_lines(data: dict[str, Any]) -> list[str]:
    test = data.get("gpu_parallel")
    if not test: return []
    if test.get("error"): return ["", f"GPU JOBS AT THE SAME TIME: not measured ({test['error']})"]
    lines = ["", f"GPU JOBS AT THE SAME TIME (GPU/{test['backend']}, {test.get('segments', 1)} segment{'' if test.get('segments', 1) == 1 else 's'} of different kinds of video, "
                 f"{test['seconds']:g} s of video{', key frames only' if test.get('keyframes') else ''})"]
    if test.get("videos"): lines.append("  " + ", ".join(test["videos"]))
    base = test["runs"][0]["throughput"] if test["runs"] else 0
    for run in test["runs"]:
        lines.append(f"  {run['jobs']} job{'' if run['jobs'] == 1 else 's'}: {run['wall']:.1f} s, {run['throughput']:.2f} s of video per second"
                     + (f" = {run['throughput'] / base:.2f}x of one job" if base > 0 and run["jobs"] > 1 else ""))
    return lines


def storage_lines(data: dict[str, Any]) -> list[str]:
    """The disks of the library: kind, videos at once, read speed."""
    rows = data.get("storage") or []
    if not rows: return []
    lines = ["", "DISKS (the kind of disk decides how many videos are processed at once: a hard disk 1, an SSD 2 unless set by hand)"]
    for row in rows:
        speed = f", the copies were read at {row['mb_per_second']:g} MB/s" if row.get("mb_per_second") else ""
        lines.append(f"  {row['disk']} {row['kind'].upper()}: {row['workers']} video{'' if row['workers'] == 1 else 's'} at once{speed}")
    return lines


def videos_at_once(data: dict[str, Any]) -> Optional[int]:
    """How many videos are being processed at the same time in all, on the disks of the library (None when the benchmark did not look)."""
    rows = data.get("storage") or []
    return sum(int(row["workers"]) for row in rows) if rows else None


def remember_disk_speeds(config: dict[str, Any], data: dict[str, Any]) -> None:
    """Keep what the benchmark read from the drives of the library, by kind of disk, for the time Generate shows."""
    totals: dict[str, list[float]] = {}
    for row in data.get("storage") or []:
        if row.get("bytes") and row.get("seconds"):
            sums = totals.setdefault(str(row["kind"]), [0.0, 0.0])
            sums[0] += float(row["bytes"])
            sums[1] += float(row["seconds"])
    speeds = config.setdefault("disk_speed", {})
    for kind, (size, seconds) in totals.items():
        if seconds > 0 and size > 0: speeds[kind] = round(size / (1024 * 1024) / seconds, 1)


def best_gpu_jobs(test: dict[str, Any]) -> int:
    """The number of jobs after which one more brings less than PARALLEL_MIN_GAIN more work done per second."""
    runs = test.get("runs") or []
    best = 1
    for before, after in zip(runs, runs[1:]):
        if after["throughput"] >= before["throughput"] * PARALLEL_MIN_GAIN: best = after["jobs"]
        else: break
    return best


def keep_equal_choices(results: dict[str, dict[str, Any]], known: dict[str, dict[str, Any]], threshold: float) -> dict[str, dict[str, Any]]:
    """A saved choice stays when the new measurement finds no real difference: the path chosen before is within `threshold` of the new best one
    (0.86 s against 0.86 s says nothing, and a suggestion that flips back and forth between runs would only be noise)."""
    kept: dict[str, dict[str, Any]] = {}
    for kind, r in results.items():
        old = known.get(kind, {}).get("best")
        old_time = r["cpu"] if old == path_choice.CPU else r["paths"].get(old) if isinstance(old, str) else None
        new_time = r["cpu"] if r["best"] == path_choice.CPU else r["paths"].get(r["best"])
        if old and old != r["best"] and old_time is not None and new_time is not None and old_time <= new_time * threshold:
            r = {**r, "best": old, "speedup": 1.0 if old == path_choice.CPU else round(r["cpu"] / max(old_time, 0.000001), 2)}
        kept[kind] = r
    return kept


def build_advice(data: dict[str, Any], config: dict[str, Any]) -> list[Advice]:
    trickplay, acceleration = config.get("trickplay", {}), config.get("acceleration", {})
    advice: list[Advice] = []
    limit = float(trickplay.get("keyframe_repeat_limit_percent", DEFAULT_REPEAT_LIMIT))
    current_only = bool(trickplay.get("keyframe_only", False))

    summary = keyframe_summary(data, limit)
    final_only = current_only
    if summary["files"] >= MIN_KEYFRAME_FILES:
        worth_it = summary["gain"] >= KEYFRAME_MIN_GAIN
        detail = keyframe_detail(summary)
        final_only = worth_it
        if worth_it != current_only:
            advice.append(Advice("trickplay", "keyframe_only", t("advice.key_frames_only"), _yes_no(current_only), _yes_no(worth_it),
                                 detail if worth_it else detail + "; " + t("advice.not_worth"), payload=worth_it))
    if final_only and limit == 0 and any(r["status"] == "PASS" and r["share"] > DEFAULT_REPEAT_LIMIT for r in data.get("keyframe_rows", [])):
        advice.append(Advice("trickplay", "keyframe_repeat_limit_percent", t("advice.repeat_share"), t("advice.no_limit"), f"{DEFAULT_REPEAT_LIMIT:g} %",
                             t("advice.repeat_reason")))

    results = path_results(data, config) if data.get("backends") else {}            # without a GPU path there is nothing to choose between
    if results:
        stored = acceleration.get("path_choice")
        same = (isinstance(stored, dict) and stored.get("ffmpeg_hash") == data.get("ffmpeg_hash")
                and sorted(stored.get("gpu_names") or []) == sorted(data.get("gpu_names") or []))
        known = {k: {"best": v.get("best")} for k, v in (stored.get("results") or {}).items()} if same and isinstance(stored, dict) else {}
        results = keep_equal_choices(results, known, float(acceleration.get("gpu_speed_threshold", 1.15)))
        if any(known.get(kind, {}).get("best") != r["best"] for kind, r in results.items()):
            moves = [f"{kind}: {_name(known[kind]['best'])} -> {_name(r['best'])}" + (" " + t("choice.faster_than_cpu", speedup=f"{r['speedup']:.2f}") if r["best"] != path_choice.CPU else "")
                     for kind, r in sorted(results.items()) if kind in known and known[kind].get("best") != r["best"]]
            if len(moves) > MAX_MOVES: moves = moves[:MAX_MOVES] + [t("advice.more_moves", count=len(moves) - MAX_MOVES)]
            advice.append(Advice("acceleration", "path_choice", t("advice.which_processor"), path_choice.summary(known) if known else t("advice.gpu_first"),
                                 path_choice.summary({**known, **results}),
                                 t("advice.path_reason"),
                                 payload={"results": results, "ffmpeg_hash": data.get("ffmpeg_hash"), "gpu_names": list(data.get("gpu_names") or [])}, details=moves))

    test = data.get("gpu_parallel")
    if test and not test.get("error") and len(test.get("runs") or []) >= 2:
        best = best_gpu_jobs(test)
        current = int(acceleration.get("max_gpu_jobs", 2))
        at_once = videos_at_once(data)
        proposed = min(best, at_once) if at_once else best          # a job needs a video: more jobs than videos at once would never be used
        # Only more jobs are ever suggested: fewer would only hold a video back, and a job that has no video to work on costs nothing.
        if proposed > current:
            first = test["runs"][0]["throughput"]
            gain = next((r["throughput"] for r in test["runs"] if r["jobs"] == proposed), first) / first if first > 0 else 1.0
            advice.append(Advice("acceleration", "max_gpu_jobs", t("advice.gpu_jobs"), current, proposed, t("advice.jobs_gain", jobs=proposed, gain=f"{gain:.2f}")))
    return advice


def advice_lines(advice: list[Advice]) -> list[str]:
    """One numbered block per suggestion: what it is about, how it is now, what is suggested, what moves, and why."""
    lines: list[str] = []
    for number, a in enumerate(advice, 1):
        if number > 1: lines.append("")
        rows = [(t("advice.now"), str(a.current)), (t("advice.suggested"), str(a.proposed))]
        if a.details: rows.append((t("advice.moves"), "\n".join(a.details)))
        rows.append((t("advice.why"), a.reason))
        lines += [f"[{number}] {a.label}", *format_labelled(rows, indent=4, minimum=12)]
    return lines


def screen_lines(data: dict[str, Any], config: dict[str, Any], report: Path) -> list[str]:
    """What the benchmark shows on the screen when it has finished: the deciding facts. Every measurement is in the report."""
    limit = float(config.get("trickplay", {}).get("keyframe_repeat_limit_percent", DEFAULT_REPEAT_LIMIT))
    lines = ["", t("diag.result_heading")]
    results = path_results(data, config) if data.get("backends") else {}
    if results: lines += ["  " + line for line in path_choice.table_lines(results, list(data["backends"]))] + [""]
    summary = keyframe_summary(data, limit)
    if summary["files"] and summary["effective_seconds"] > 0:
        lines.append(fit("  " + t("advice.screen_key_frames", detail=keyframe_detail(summary))))
    for row in data.get("storage") or []:
        lines.append("  " + t("preflight.workers_disk", disk=row["disk"], kind=row["kind"].upper(), count=row["workers"])
                     + (t("advice.disk_speed", speed=f"{row['mb_per_second']:.0f}") if row.get("mb_per_second") else ""))
    test = data.get("gpu_parallel")
    if test and not test.get("error") and test.get("runs"):
        base = test["runs"][0]["throughput"]
        parts = [tp("advice.jobs_count", run["jobs"]) + (f" = {run['throughput'] / base:.2f}x" if run["jobs"] > 1 and base > 0 else "") for run in test["runs"]]
        lines.append(fit("  " + t("advice.screen_parallel", backend=test["backend"]) + ": " + ", ".join(parts) + (" " + t("advice.of_one_video") if len(parts) > 1 else "")))
    if videos_at_once(data) == 1 and test and test.get("runs"): lines.append(fit("  " + t("advice.one_at_a_time")))
    speeds = data.get("read_speed") or []
    if speeds and not data.get("storage"):
        mb, seconds = sum(r["bytes"] for r in speeds) / (1024 * 1024), sum(r["seconds"] for r in speeds)
        lines.append("  " + t("advice.drive_reads", speed=f"{mb / max(seconds, 0.001):.0f}"))
    tagged = {r["video"] for r in data.get("rows", []) if r.get("color_tag_ignored")}
    if tagged: lines.append("  " + tp("advice.tagged", len(tagged)))
    if data.get("direct_files"): lines.append("  " + tp("advice.direct", len(data["direct_files"])))
    failed = sorted({r["backend"] for r in data.get("rows", []) if r["status"] not in {"PASS", "CANCELED"}})
    if failed: lines.append(fit("  " + t("advice.failed_paths", paths=", ".join(failed))))
    lines += ["", fit(t("advice.every_measurement", report=report), 0)]
    return lines


def apply_advice(config: dict[str, Any], advice: list[Advice]) -> None:
    for item in advice:
        if item.key == "path_choice":
            p = item.payload
            config.setdefault("acceleration", {})["path_choice"] = path_choice.merged(config.get("acceleration", {}).get("path_choice"), p["results"], p["ffmpeg_hash"], p["gpu_names"])
            continue
        if item.key == "keyframe_only": value: Any = bool(item.payload)
        elif item.key == "keyframe_repeat_limit_percent": value = DEFAULT_REPEAT_LIMIT
        else: value = int(item.proposed)
        config.setdefault(item.section, {})[item.key] = value
