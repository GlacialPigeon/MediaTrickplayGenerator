# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform as host_info
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from . import benchmark_advice, library_sample, storage_workers, system_info
from .activity_log import ACTIVITY
from .console_progress import CONSOLE_PROGRESS
from .constants import APP_NAME, APP_VERSION, DIRS
from .datetime_helpers import local_stamp, utc_now_iso
from .error_summary import diagnostic_error_summary
from .file_io import atomic_write_json, atomic_write_text
from .formatting import human_duration
from .extraction import COLOR_TAG_ERROR
from .generator import TrickplayGenerator, disk_label, library_disks
from .localization import english, t, tp
from .models import ExtractResult, VideoMeta
from .prompts import fit, heading
from .storage import physical_storage_group
from .tiles import PIL_AVAILABLE, hdr_kind, hdr_label, validate_tile_directory

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]


def diagnostic_segments(duration: float, mode: str) -> list[tuple[float, float]]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("A finite positive video duration is required")
    if mode not in {"quick", "full", "sample"}:
        raise ValueError("Unknown diagnostic mode")
    if mode == "full" or duration <= 120:
        return [(0.0, duration)]
    if mode == "sample":                      # one two-minute stretch from the middle
        return [((duration - 120) / 2, 120.0)]
    count = min(3, int(duration // 120))
    if count == 1:
        return [((duration - 120) / 2, 120.0)]
    # Centers are near 20%, 50%, 80%, constrained to disjoint complete windows.
    positions = [0.2, 0.8] if count == 2 else [0.2, 0.5, 0.8]
    starts: list[float] = []
    for i, position in enumerate(positions):
        lower = starts[-1] + 120 if starts else 0
        upper = duration - 120 * (count - i)
        starts.append(max(lower, min(position * duration - 60, upper)))
    return [(round(start, 3), 120.0) for start in starts]


_HDR_KIND_ORDER = {"HDR10 / PQ": 0, "HDR10+": 1, "HLG": 2, "Dolby Vision": 3}


def select_hdr_candidates(videos: list[VideoMeta]) -> list[VideoMeta]:
    """The videos the HDR check should test: one for every kind of HDR found (HDR10 / PQ, HDR10+, HLG, each Dolby Vision
    profile, and HDR whose type is not confirmed), in that order. Per kind the best file wins: confirmed HDR before
    "uncertain", then HEVC, the largest picture, the deepest colors, the longest."""
    best: dict[tuple[str, str], VideoMeta] = {}
    for meta in videos:
        if not (meta.hdr and math.isfinite(meta.duration) and meta.duration > 0): continue
        kind = (meta.hdr_type, meta.dv_profile if meta.hdr_type == "Dolby Vision" else "")
        rank = lambda m: (m.hdr_pool == "HDR", m.codec == "hevc", m.width * m.height, m.bit_depth, m.duration)
        if kind not in best or rank(meta) > rank(best[kind]): best[kind] = meta
    return [best[kind] for kind in sorted(best, key=lambda k: (_HDR_KIND_ORDER.get(k[0], len(_HDR_KIND_ORDER)), k[1]))]


def benchmark_totals(data: dict[str, Any]) -> list[str]:
    def key(row: dict[str, Any]) -> tuple:
        return row["video"], row["window"], row["start_seconds"], row["duration_seconds"]

    def elapsed(row: dict[str, Any]) -> float:
        return row.get("extract_seconds", 0) + row.get("pack_seconds", 0)

    cpu = {key(r): r for r in data["rows"] if r["backend"] == "cpu" and r["status"] == "PASS"}
    table = [["Path", "Total time (s)", "Passed", "Failed", "Canceled", "Speed vs CPU", "Matched"]]
    for backend in ["cpu", *data["backends"]]:
        rows = [r for r in data["rows"] if r["backend"] == backend]
        passed = sum(r["status"] == "PASS" for r in rows)
        failed = sum(r["status"] in {"FAILED", "MISMATCH"} for r in rows)
        canceled = sum(r["status"] == "CANCELED" for r in rows)
        matched = [r for r in rows if r["status"] == "PASS" and key(r) in cpu
                   and (backend == "cpu" or r.get("matching_frame_count", False))]
        denominator = sum(elapsed(r) for r in matched)
        speed = f"{sum(elapsed(cpu[key(r)]) for r in matched) / denominator:.2f}x" if denominator > 0 else "-"
        table.append(["CPU" if backend == "cpu" else f"GPU/{backend}",
                      f"{sum(elapsed(r) for r in rows):.2f}", str(passed), str(failed), str(canceled), speed, str(len(matched))])
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    lines = [" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in table]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    return ["", "TOTAL CPU VS GPU", *lines,
            "Total time: extraction + packing, including recorded failed attempts.",
            "Passed/failed counts are measurements (video segments), not unique files.",
            "Speed: matched successful segments only; >1 GPU faster, <1 CPU faster."]


def benchmark_speed_table(data: dict[str, Any]) -> list[str]:
    backends = ["cpu", *data["backends"]]
    header = ["File", "Codec", "CPU (s)"]
    for backend in backends[1:]: header.extend([f"{backend} (s)", f"{backend} vs CPU"])
    header.append("Fastest")
    table = [header]
    files = list(dict.fromkeys(r["video"] for r in data["rows"]))
    for number, video in enumerate(files, 1):
        rows = [r for r in data["rows"] if r["video"] == video]
        cpu = [r for r in rows if r["backend"] == "cpu"]
        windows = {(r["window"], r["start_seconds"], r["duration_seconds"]) for r in cpu}
        times = {}
        cells = {}
        for backend in backends:
            runs = [r for r in rows if r["backend"] == backend]
            keys = {(r["window"], r["start_seconds"], r["duration_seconds"]) for r in runs}
            if not runs or keys != windows or len(runs) != len(cpu): cells[backend] = "INCOMPLETE"
            elif any(r["status"] != "PASS" for r in runs): cells[backend] = next(r["status"] for r in runs if r["status"] != "PASS")
            elif backend != "cpu" and any(not r.get("matching_frame_count") for r in runs): cells[backend] = "NOT COMPARABLE"
            else:
                times[backend] = sum(r["extract_seconds"] + r["pack_seconds"] for r in runs)
                cells[backend] = f"{times[backend]:.2f}"
        row = [str(number), rows[0].get("codec", "unknown"), cells["cpu"]]
        for backend in backends[1:]:
            ratio = f"{times['cpu'] / max(times[backend], .000001):.2f}x" if "cpu" in times and backend in times else "-"
            row.extend([cells[backend], ratio])
        fastest = min(times, key=lambda name: times[name]) if "cpu" in times and data["status"] == "COMPLETE" else "-"
        row.append(fastest.upper() if fastest == "cpu" else fastest)
        table.append(row)
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    formatted = [" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in table]
    formatted.insert(1, "-+-".join("-" * width for width in widths))
    return ["CPU VS GPU SPEED COMPARISON", "Times = extraction + packing, summed over the same tested windows (seconds).",
            "Ratio = CPU time / GPU-path time: >1 GPU path faster; <1 CPU faster.",
            "Fastest = measured minimum; recommendations below may retain CPU for small gains.", "", *formatted, "",
            *[f"{i}. {video}" for i, video in enumerate(files, 1)], ""]


def clock_text(start: float) -> str:
    seconds = int(start)
    if seconds >= 3600: return f"{seconds // 3600}h{seconds // 60 % 60:02d}m{seconds % 60:02d}s"
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def safe_stem(meta: VideoMeta) -> str:
    """File name of the video without characters Windows forbids and at most 40 characters long."""
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", meta.path.stem)
    stem = re.sub(r"\s+", " ", stem).strip(" ._")
    if len(stem) > 40: stem = stem[:39].rstrip(" ._") + "~"
    return stem or "video"


def report_name(video_no: int, meta: VideoMeta, window_no: int, start: float) -> str:
    """Readable, filesystem-safe name of one tested segment: number, kind of HDR, file, segment and start time."""
    stem = safe_stem(meta)
    return f"{video_no:02d} {hdr_label(meta) or 'SDR'} - {stem} - seg{window_no} @{clock_text(start)}"


class DiagnosticRunner:
    """Sequential, isolated runs: one input window and settings for every backend."""

    def __init__(self, generator: TrickplayGenerator, job: "DiagnosticJob") -> None:
        self.generator = generator
        self.job = job
        self.name_width = self.NAME_WIDTH
        self.kind_texts: dict[str, str] = {}    # benchmark: the kinds of video of this run in columns that line up
        self.compact = False                    # benchmark: one table row per tested segment on the screen, the details in the report
        self.cells: dict[str, str] = {}         # the cells of the row being tested: the time of each path, "kf" = key frames only
        self.current = ""
        self.current_name = ""
        self.started = time.monotonic()
        self.test_no = 0
        self.total_tests = 0

    PARALLEL_SECONDS = 60.0
    PARALLEL_MAX_JOBS = 4

    def run(self, videos: list[VideoMeta], mode: str, backends: list[str], kind: str, *, local_copy: bool = False,
            sample: Optional[list[dict[str, str]]] = None) -> dict[str, Any]:
        """`local_copy`: every tested segment is first copied (stream copy) into the program's temp folder and measured there, so the
        drive the video is on does not distort the timings; the copy also measures how fast that drive reads."""
        gen = self.generator
        data: dict[str, Any] = {
            "kind": kind, "mode": mode, "created_at": utc_now_iso(), "status": "RUNNING",
            "settings": asdict(gen.settings), "ffmpeg_hash": gen.tools.candidate.sha256,
            "ffmpeg_version": gen.tools.candidate.version_line, "gpu_names": gen.tools.gpu_names,
            "cpu": host_info.processor() or host_info.machine(),
            "system": system_info.as_dict(system_info.host(), gen.tools.gpu_names, system_info.all_disks(), library_disks(gen.config.get("media_roots", []))), "backends": backends, "rows": [], "visual": {},
            "local_copy": local_copy, "sample": sample or [], "read_speed": [], "direct_files": [], "keyframe_rows": [],
            "notes": ["Single sequential pass; OS cache and thermal state may affect timings.",
                      "Backend labels identify requested paths, not proof of hardware decoding.",
                      "HDR images use current generation filters; no automatic HDR-to-SDR correction is applied.",
                      "CPU/GPU frame slots use identical windows and sampling settings; visual alignment still requires review."],
        }
        if not any(meta.hdr for meta in videos):
            data["notes"].append("No HDR video detected in this benchmark folder.")
        if local_copy:
            data["notes"].append("Every tested segment was copied (stream copy) into the temp folder first; the timings do not include reading the library drive, "
                                 "except for the files listed under direct_files, whose segment could not be copied and was tested on the library drive.")
        data["warmup"] = {}
        self.save(data)
        extra = 1 if kind == "benchmark" else 0          # the run with key frames only (A/B)
        self.total_tests = sum(len(diagnostic_segments(meta.duration, mode)) * (1 + len([b for b in backends if not self._skipped(meta, kind, b)]) + extra) for meta in videos)
        self.test_no = 0
        self.started = time.monotonic()
        self.compact = kind in {"benchmark", "hdr"}       # one row per tested segment on the screen; the details of every test go to the report
        self.name_width = self.NAME_WIDTH if kind == "benchmark" else self.NAME_WIDTH + 14
        self.kind_texts = library_sample.aligned_kinds([library_sample.kind_key(m) for m in videos]) if kind == "benchmark" else {}
        if self.kind_texts: self.name_width = max(len(t("diag.col_kind")), *(len(text) for text in self.kind_texts.values()))
        self.cells = {}
        self.current = ""
        try:
            if self.compact:
                heading(t("diag.benchmark_heading") if kind == "benchmark" else t("diag.hdr_test_heading"))
                self._status(t("diag.status_warming_up"))
            self._warm_up(videos[0], mode, backends, data)
            if self.compact:
                self._clear_status()
                print(self._table_row("#", t("diag.col_kind") if kind == "benchmark" else t("diag.col_hdr_kind"), ["CPU", *backends], t("diag.col_key_frames") if extra else ""))
            for video_no, meta in enumerate(videos, 1):
                windows = diagnostic_segments(meta.duration, mode)
                self.job.log(f"\nFILE {video_no}/{len(videos)}: {meta.path.name}", echo=not self.compact)
                for window_no, (start, duration) in enumerate(windows, 1):
                    self.cells = {}
                    self.current = f"{video_no}/{len(videos)}" + ("" if len(windows) == 1 else f" s{window_no}")
                    self.current_name = meta.path.name
                    self._status(t("diag.status_copying"))
                    self.job.log(f"  Segment {window_no}/{len(windows)} | {human_duration(start)} - {human_duration(start + duration)} | length {human_duration(duration)}", echo=not self.compact)
                    window: dict[str, Any] = {"video_no": video_no, "videos": len(videos), "window_no": window_no, "windows": len(windows),
                              "start": start, "duration": duration, "source": meta, "source_start": start}
                    staged: Optional[library_sample.StagedSegment] = None
                    if local_copy:
                        try:
                            staged = library_sample.stage_segment(gen.tools.ffmpeg, meta, start, duration, DIRS["temp"])
                        except (KeyboardInterrupt, InterruptedError): raise
                        except Exception as exc:
                            self.job.log(f"    {exc}", echo=not self.compact)
                            self.job.log("    The segment is tested directly on the library drive instead (its timings include reading it).", echo=not self.compact)
                            data["direct_files"].append(str(meta.path))
                    if staged:
                        window.update(source=staged.meta, source_start=staged.start, copied=True)
                        speed = staged.read_speed / (1024 * 1024)
                        data["read_speed"].append({"video": str(meta.path), "bytes": staged.read_bytes, "seconds": round(staged.read_seconds, 3), "mb_per_second": round(speed, 2)})
                        self.job.log(f"    Copied the segment locally: {staged.read_bytes / (1024 * 1024):.0f} MB in {staged.read_seconds:.1f} s ({speed:.1f} MB/s)", echo=not self.compact)
                    try:
                        for backend in [None, *backends]:
                            if self._skipped(meta, kind, backend):
                                self.cells[backend or "cpu"] = t("diag.cell_skipped")
                                self.job.log(f"    {backend}: not tested, the backend compatibility check found that it does not decode this kind of video on the GPU", echo=False)
                                data.setdefault("skipped_paths", []).append({"video": str(meta.path), "backend": backend})
                                continue
                            self._measure(data, meta, kind, mode, window, backend)
                        if extra: self._measure_keyframes(data, meta, mode, window)
                    finally:
                        if staged: staged.remove()
                    done = self.test_no
                    left = (time.monotonic() - self.started) / max(done, 1) * max(self.total_tests - done, 0)
                    self.job.log(f"    Elapsed {human_duration(time.monotonic() - self.started)} | about {human_duration(left)} left", echo=not self.compact)
                    if self.compact:
                        self._clear_status()
                        print(self._table_row(self.current, self._kind_text(meta, kind), [self.cells.get(b or "cpu", "") for b in [None, *backends]], self.cells.get("kf", "")))
                if kind == "hdr": self._all_paths_sheet(data, meta, video_no, backends)
            if self.compact:
                self._clear_status()
                print("\n" + t("diag.all_done", time=human_duration(time.monotonic() - self.started)))
                if kind == "benchmark":
                    for legend_line in t("diag.table_legend", skipped=t("diag.cell_skipped")).split("\n"): print(fit(legend_line))
            data["storage"] = self._storage_summary(data, videos)
            if extra: self._parallel_mix(data, videos, local_copy)
            data["status"] = "COMPLETE"
        except (KeyboardInterrupt, InterruptedError):
            gen.stop_event.set()
            data["status"] = "CANCELED"
        finally:
            if self.compact: self._clear_status()
            self.save(data)
        return data

    def _comparison_path(self, name: str) -> Path:
        folder = self.job.path / "comparisons"
        folder.mkdir(exist_ok=True)
        return folder / name

    def _all_paths_sheet(self, data: dict[str, Any], meta: VideoMeta, video_no: int, backends: list[str]) -> None:
        """One picture per HDR file: a column per path (CPU, then every GPU path), a row per segment, the middle frame of each."""
        paths = ["cpu", *backends]
        rows: list[tuple[str, list[Optional[Path]]]] = []
        for window_no in sorted({r["window"] for r in data["rows"] if r["video"] == str(meta.path)}):
            cells: list[Optional[Path]] = []
            for path in paths:
                row = next((r for r in data["rows"] if r["video"] == str(meta.path) and r["window"] == window_no
                            and r["backend"] == path and r["status"] == "PASS" and r["samples"]), None)
                cells.append(self.job.path / row["samples"][len(row["samples"]) // 2] if row else None)
            start = next(r["start_seconds"] for r in data["rows"] if r["video"] == str(meta.path) and r["window"] == window_no)
            rows.append((f"seg{window_no} @{clock_text(start)}", cells))
        if not any(cell for _, cells in rows for cell in cells): return
        name = f"{video_no:02d} {hdr_label(meta) or 'SDR'} - {safe_stem(meta)} - all paths.jpg"
        try:
            create_paths_sheet(["CPU" if p == "cpu" else p for p in paths], rows, self.job.path / name,
                               f"{hdr_label(meta) or 'SDR'} | {meta.path.name}\nMiddle frame of every segment; a column per path")
        except Exception as exc:
            self.job.log(f"Could not create {name}: {exc}", echo=False)
            return
        data.setdefault("overview", []).append({"video": str(meta.path), "hdr_label": hdr_label(meta), "picture": name})
        self.save(data)

    NAME_WIDTH = 30
    CELL_WIDTH = 9

    def _skipped(self, meta: VideoMeta, kind: str, backend: Optional[str]) -> bool:
        """The benchmark does not time a GPU path that the backend compatibility check found not to decode this kind of video on the GPU."""
        return kind == "benchmark" and backend is not None and self.generator._gpu_decode_blocked(meta, backend)

    def _kind_text(self, meta: VideoMeta, kind: str) -> str:
        """The second column: the kind of video (benchmark) or the kind of HDR and the file (HDR test), cut to the width of the column."""
        text = self.kind_texts.get(library_sample.kind_key(meta), library_sample.kind_key(meta)) if kind == "benchmark" else f"{hdr_label(meta)} | {meta.path.name}"
        return text if len(text) <= self.name_width else text[:self.name_width - 3] + "..."

    def _table_row(self, number: str, kind_text: str, times: list[str], keyframes: str) -> str:
        """One line of the table: the number of the file (and segment), its kind, the time of every path and what key frames only brought."""
        return f"  {number:<8}{kind_text:<{self.name_width}}  " + "".join(f"{t:<{self.CELL_WIDTH}}" for t in times) + f" {keyframes}".rstrip()

    def _status(self, text: str) -> None:
        """A line that is written over again while a file is tested (what is running, the time so far and the time left). Only on a real terminal."""
        if not (self.compact and sys.stdout.isatty() and CONSOLE_PROGRESS.enabled): return
        elapsed = time.monotonic() - self.started
        done = max(self.test_no - 1, 0)
        left = elapsed / done * max(self.total_tests - done, 0) if done else None
        where = f"{self.current} {self.current_name}" if self.current else ""
        line = f"  > {where[:50] + '...' if len(where) > 53 else where} | {text} | {human_duration(elapsed)}" + (" | " + t("diag.status_left", time=human_duration(left)) if left is not None else "")
        sys.stdout.write("\r\x1b[2K" + line[:CONSOLE_PROGRESS._terminal_width() - 1])
        sys.stdout.flush()

    def _clear_status(self) -> None:
        if self.compact and sys.stdout.isatty() and CONSOLE_PROGRESS.enabled:
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()

    def _extract(self, source: VideoMeta, frames_dir: Path, method: str, backend: Optional[str], **kwargs: Any) -> ExtractResult:
        """One measured FFmpeg run, with the one retry Generate also makes: a file whose colour space tag FFmpeg refuses (bt2020c) is
        run again with the tag ignored. Without it such a file would fail on every path here and succeed in Generate. Only the run that
        decides is returned, so the time of the refused first try is not part of the measurement. Other retries and fallbacks stay out."""
        gen = self.generator
        first = gen._run_ffmpeg_extract(source, frames_dir, method, backend, **kwargs)
        if first.success or first.canceled or first.stalled or kwargs.get("ignore_color_tag") or COLOR_TAG_ERROR not in first.error.lower():
            first.color_tag_ignored = bool(kwargs.get("ignore_color_tag")) and first.success
            return first
        kwargs["ignore_color_tag"] = True
        second = gen._run_ffmpeg_extract(source, frames_dir, method, backend, **kwargs)
        second.color_tag_ignored = second.success
        return second

    def _new_row(self, meta: VideoMeta, window: dict[str, Any], backend: Optional[str]) -> dict[str, Any]:
        return {
            "video": str(meta.path), "codec": meta.codec, "hdr": meta.hdr, "hdr_type": meta.hdr_type, "hdr_pool": meta.hdr_pool,
            "hdr_label": hdr_label(meta), "hdr_kind": hdr_kind(meta), "kind": library_sample.kind_key(meta),
            "source_size": meta.size, "source_mtime_ns": meta.mtime_ns,
            "source_duration": meta.duration, "pix_fmt": meta.pix_fmt,
            "source_width": meta.width, "source_height": meta.height,
            "window": window["window_no"], "start_seconds": window["start"], "duration_seconds": window["duration"],
            "backend": backend or "cpu", "status": "FAILED", "samples": [],
        }

    def _measure(self, data: dict[str, Any], meta: VideoMeta, kind: str, mode: str, window: dict[str, Any], backend: Optional[str]) -> None:
        gen = self.generator
        source: VideoMeta = window["source"]
        start, duration, window_no = window["start"], window["duration"], window["window_no"]
        video_no, windows = window["video_no"], window["windows"]
        gen._check_canceled()
        self.test_no += 1
        test_no, total_tests = self.test_no, self.total_tests
        method = "GPU" if backend else "CPU"
        path_label = f"GPU/{backend}" if backend else "CPU"
        test_label = f"[{test_no}/{total_tests}] {path_label}"
        label = report_name(video_no, meta, window_no, start)
        path_name = f"GPU {backend}" if backend else "CPU"
        self._status(t("diag.status_testing", name=path_label))
        row = self._new_row(meta, window, backend)
        row["local_copy"] = bool(window.get("copied"))
        self.job.log(f"{test_label} | {meta.path.name} | start={start:.3f}s duration={duration:.3f}s", echo=False)
        try:
            before = source.path.stat()
            if (before.st_size, before.st_mtime_ns) != source.signature:
                raise RuntimeError("Source changed since scan")
            with tempfile.TemporaryDirectory(prefix="work_", dir=self.job.path) as work:
                work_path = Path(work)
                frames_dir = work_path / "frames"
                # A direct call intentionally bypasses production retries, fallback and failure cache (only the colour tag retry is kept).
                result = self._extract(source, frames_dir, method, backend,
                    start_time=window["source_start"], duration_limit=duration if mode != "full" else None,
                    progress_label=f"Test {test_no}/{total_tests} | file {video_no}/{window['videos']} | segment {window_no}/{windows}")
                row.update(extract_seconds=result.elapsed, frames=result.frames, color_tag_ignored=result.color_tag_ignored)
                if result.canceled:
                    row["status"] = "CANCELED"
                    raise InterruptedError("Canceled")
                if not result.success:
                    raise RuntimeError(result.error)
                pack_start = time.monotonic()
                frames, tiles, _ = gen._pack_frames(frames_dir, work_path / "sprites")
                row["pack_seconds"] = time.monotonic() - pack_start
                valid = validate_tile_directory(work_path / "sprites", gen.settings, duration,
                                                expected_exact_tiles=tiles, verify_all=True)
                if not valid.valid:
                    raise RuntimeError(valid.reason)
                after = source.path.stat()
                if (after.st_size, after.st_mtime_ns) != source.signature:
                    raise RuntimeError("Source changed during test")
                sample_dir = self.job.path / "samples" / f"{label} - {path_name}"
                sample_dir.mkdir(parents=True)
                paths = sorted(frames_dir.glob("frame_*.jpg"))
                indices = sorted({0, len(paths) // 2, len(paths) - 1})
                for index in indices:
                    dest = sample_dir / paths[index].name
                    shutil.copy2(paths[index], dest)
                    row["samples"].append(str(dest.relative_to(self.job.path)))
                row.update(status="PASS", tiles=tiles, frames=frames, output_bytes=valid.output_bytes)
                if kind == "hdr" and not backend:
                    sheet = self._comparison_path(f"{label} - CPU only.jpg")
                    create_contact_sheet([(self.job.path / p, None) for p in row["samples"]], sheet,
                                         f"{row['hdr_label'] or 'SDR'} | {meta.path.name}\nCPU only | start {start:g}s")
                    row["comparison"] = sheet.relative_to(self.job.path).as_posix()
                cpu = next((r for r in data["rows"] if r["video"] == str(meta.path)
                            and r["window"] == window_no and r["backend"] == "cpu" and r["status"] == "PASS"), None)
                if backend and cpu:
                    row["matching_frame_count"] = frames == cpu["frames"]
                    if frames != cpu["frames"]:
                        row.update(status="MISMATCH", error="CPU/GPU frame counts differ; not comparable")
                    else:
                        row["extraction_speedup"] = cpu["extract_seconds"] / max(result.elapsed, 0.000001)
                        row["total_speedup"] = (cpu["extract_seconds"] + cpu["pack_seconds"]) / max(result.elapsed + row["pack_seconds"], 0.000001)
                        if kind == "hdr":
                            sheet = self._comparison_path(f"{label} - GPU {backend} vs CPU.jpg")
                            pairs = [(self.job.path / a, self.job.path / b) for a, b in zip(cpu["samples"], row["samples"])]
                            create_contact_sheet(pairs, sheet, f"{row['hdr_label'] or 'SDR'} | {meta.path.name}\nCPU vs {backend} | start {start:g}s | identical frame slots")
                            row["comparison"] = sheet.relative_to(self.job.path).as_posix()
        except (KeyboardInterrupt, InterruptedError):
            row["status"] = "CANCELED"
            self.job.log(f"    {test_label:<28} CANCELED", echo=not self.compact)
            data["rows"].append(row)
            raise
        except Exception as exc:
            row["status"] = "FAILED"
            row["error"] = str(exc)
            row["error_summary"] = diagnostic_error_summary(str(exc))
            self.job.log(f"Full error for {label} | {meta.path}:\n{exc}", echo=False)
        elapsed = row.get("extract_seconds", 0) + row.get("pack_seconds", 0)
        message = f"    {test_label:<28} {row['status']:<9} {elapsed:7.2f}s"
        if row.get("error"):
            message += " | " + (row.get("error_summary") or diagnostic_error_summary(row["error"]))
        elif "total_speedup" in row:
            ratio = row["total_speedup"]
            message += f" | {ratio:.2f}x faster than CPU" if ratio >= 1 else f" | CPU is {1 / max(ratio, .000001):.2f}x faster"
        if row.get("color_tag_ignored"): message += " | colour space tag ignored, as Generate does"
        self.job.log(message, echo=not self.compact)
        self.cells[backend or "cpu"] = f"{elapsed:.2f}s" if row["status"] == "PASS" else (t("diag.cell_differs") if row["status"] == "MISMATCH" else t("diag.cell_fail"))
        data["rows"].append(row)
        self.save(data)

    def _measure_keyframes(self, data: dict[str, Any], meta: VideoMeta, mode: str, window: dict[str, Any]) -> Optional[str]:
        """A/B: the fastest path of this segment once more, decoding key frames only. Returns that path ("cpu" or a GPU backend), or None."""
        gen = self.generator
        source: VideoMeta = window["source"]
        window_no, duration = window["window_no"], window["duration"]
        self.test_no += 1
        mine = [r for r in data["rows"] if r["video"] == str(meta.path) and r["window"] == window_no and r["status"] == "PASS"
                and (r["backend"] == "cpu" or r.get("matching_frame_count"))]
        if not any(r["backend"] == "cpu" for r in mine): return None             # no reference to compare with
        best = min(mine, key=lambda r: r["extract_seconds"] + r["pack_seconds"])
        backend = None if best["backend"] == "cpu" else str(best["backend"])
        all_seconds = best["extract_seconds"] + best["pack_seconds"]
        row: dict[str, Any] = {"video": str(meta.path), "window": window_no, "backend": best["backend"], "status": "FAILED", "all_seconds": all_seconds, "all_frames": best["frames"]}
        test_label = f"[{self.test_no}/{self.total_tests}] KEY FRAMES on {'GPU/' + backend if backend else 'CPU'}"
        self.job.log(f"{test_label} | {meta.path.name}", echo=False)
        self._status(t("diag.status_testing_key_frames"))
        try:
            with tempfile.TemporaryDirectory(prefix="keyframes_", dir=self.job.path) as work:
                frames_dir = Path(work) / "frames"
                result = self._extract(source, frames_dir, "GPU" if backend else "CPU", backend, start_time=window["source_start"],
                                                 duration_limit=duration if mode != "full" else None, keyframes=True,
                                                 ignore_color_tag=bool(best.get("color_tag_ignored")),
                                                 progress_label=f"Key frames {self.test_no}/{self.total_tests} | file {window['video_no']}/{window['videos']}")
                row["keyframe_seconds"] = result.elapsed
                if result.canceled: raise InterruptedError("Canceled")
                if not result.success: raise RuntimeError(result.error)
                pack_start = time.monotonic()
                frames, _tiles, _ = gen._pack_frames(frames_dir, Path(work) / "sprites")
                repeated, total = gen.repeated_frames(frames_dir)
                row.update(status="PASS", keyframe_seconds=result.elapsed + (time.monotonic() - pack_start), frames=frames, repeated=repeated,
                           share=100.0 * repeated / max(total, 1), matching_frame_count=abs(frames - best["frames"]) <= 2)
        except (KeyboardInterrupt, InterruptedError): raise
        except Exception as exc:
            row.update(status="FAILED", error=str(exc), error_summary=diagnostic_error_summary(str(exc)))
            self.job.log(f"Full error for the key frame test of {meta.path}:\n{exc}", echo=False)
        if row["status"] == "PASS":
            ratio = all_seconds / max(row["keyframe_seconds"], 0.000001)
            self.job.log(f"    {test_label:<44} PASS {row['keyframe_seconds']:6.2f}s | {ratio:.2f}x faster than all frames | "
                         f"{row['share']:.0f}% of the thumbnails repeat the one before", echo=not self.compact)
            path = backend or "CPU"
            shown = t("diag.cell_kf_faster", ratio=f"{ratio:.1f}", path=path) if ratio >= 1 else t("diag.cell_kf_slower", ratio=f"{1 / max(ratio, 0.000001):.1f}", path=path)
            self.cells["kf"] = shown + (" " + t("diag.cell_repeated", share=f"{row['share']:.0f}") if row["share"] >= 5 else "")
        else:
            self.job.log(f"    {test_label:<44} FAILED | {row['error_summary']} (every frame will be used)", echo=not self.compact)
            self.cells["kf"] = t("diag.cell_all_frames")
        data["keyframe_rows"].append(row)
        self.save(data)
        return str(best["backend"])

    PARALLEL_SEGMENTS = 8
    PARALLEL_BUDGET_SECONDS = 60.0       # the chosen segments take about this long one after the other (with one job)
    PARALLEL_MIN_SEGMENT = 10.0          # a shorter stretch would only measure how long FFmpeg takes to start

    def _storage_summary(self, data: dict[str, Any], videos: list[VideoMeta]) -> list[dict[str, Any]]:
        """The disks the tested videos are on: the kind, how many videos are processed at once on each under the settings, and how fast
        the drive read the copies (when the segments were copied)."""
        setting = self.generator.config.get("workers", {}).get("per_storage_group", storage_workers.AUTO)
        group_of = {str(meta.path): physical_storage_group(meta.path) for meta in videos}
        groups: dict[str, dict[str, Any]] = {}
        for meta in videos:
            g = groups.setdefault(group_of[str(meta.path)], {"disk": disk_label(meta.path), "paths": [], "bytes": 0, "seconds": 0.0})
            g["paths"].append(meta.path)
        for row in data.get("read_speed", []):
            owner = groups.get(group_of.get(row["video"], ""))
            if owner is not None:
                owner["bytes"] += int(row["bytes"])
                owner["seconds"] += float(row["seconds"])
        summary: list[dict[str, Any]] = []
        for group, g in groups.items():
            kind = storage_workers.group_kind(g["paths"])
            speed = g["bytes"] / (1024 * 1024) / g["seconds"] if g["seconds"] > 0 else None
            summary.append({"group": group, "disk": g["disk"], "kind": kind, "workers": storage_workers.per_disk(setting, kind),
                            "bytes": g["bytes"], "seconds": round(g["seconds"], 3), "mb_per_second": round(speed, 1) if speed else None})
        return summary

    def _parallel_mix(self, data: dict[str, Any], videos: list[VideoMeta], local_copy: bool) -> None:
        """How much work the GPU gets through with 1, 2, 3 ... videos at the same time, on the real mix: the segments where a GPU path won,
        one of every kind of video, the longest first. One job is not a good test: how much a second one adds depends on how much the
        first leaves unused (starting FFmpeg, reading the index, a video the graphics card decodes faster than the drive delivers it),
        and that differs from one kind of video and from one computer to the next."""
        gen = self.generator
        metas = {str(m.path): m for m in videos}
        by_row = {(r["video"], r["window"], r["backend"]): r for r in data["rows"]}
        candidates: list[tuple[VideoMeta, dict[str, Any], dict[str, Any]]] = []
        for kf in sorted((r for r in data["keyframe_rows"] if r["backend"] != "cpu"), key=lambda r: r["all_seconds"], reverse=True):
            row, meta = by_row.get((kf["video"], kf["window"], kf["backend"])), metas.get(kf["video"])
            if row is not None and meta is not None and row["duration_seconds"] >= self.PARALLEL_MIN_SEGMENT: candidates.append((meta, row, kf))
        chosen: list[tuple[VideoMeta, dict[str, Any], dict[str, Any]]] = []
        kinds: set[str] = set()
        spent = 0.0
        for first_pass in (True, False):                       # one of every kind of video first, then the longest of the rest
            for candidate in candidates:
                if len(chosen) >= self.PARALLEL_SEGMENTS or spent >= self.PARALLEL_BUDGET_SECONDS: break
                if any(candidate is c for c in chosen) or (first_pass and candidate[1]["kind"] in kinds): continue
                kinds.add(candidate[1]["kind"])
                chosen.append(candidate)
                spent += candidate[2]["all_seconds"]
        if not chosen: return
        limit = float(gen.config.get("trickplay", {}).get("keyframe_repeat_limit_percent", benchmark_advice.DEFAULT_REPEAT_LIMIT))
        summary = benchmark_advice.keyframe_summary(data, limit)
        keyframes = bool(gen.keyframe_only) or (summary["files"] >= benchmark_advice.MIN_KEYFRAME_FILES and summary["gain"] >= benchmark_advice.KEYFRAME_MIN_GAIN)
        staged: list[library_sample.StagedSegment] = []
        segments: list[dict[str, Any]] = []
        test: dict[str, Any] = {"backend": "", "segments": 0, "seconds": 0.0, "keyframes": keyframes, "videos": [], "runs": []}
        data["gpu_parallel"] = test
        try:
            for meta, row, kf in chosen:
                source, start = meta, float(row["start_seconds"])
                if local_copy and str(meta.path) not in data["direct_files"]:
                    try:
                        copy = library_sample.stage_segment(gen.tools.ffmpeg, meta, start, float(row["duration_seconds"]), DIRS["temp"] / f"parallel_{len(staged)}")
                        staged.append(copy)
                        source, start = copy.meta, copy.start
                    except (KeyboardInterrupt, InterruptedError): raise
                    except Exception: pass                           # the original is used
                segments.append({"source": source, "start": start, "duration": float(row["duration_seconds"]), "backend": str(kf["backend"]),
                                 "ignore_tag": bool(row.get("color_tag_ignored")), "name": meta.path.name})
            total = sum(seg["duration"] for seg in segments)
            test.update(backend=", ".join(sorted({seg["backend"] for seg in segments})), segments=len(segments), seconds=round(total, 1), videos=[seg["name"] for seg in segments])
            if self.compact: print("\n" + fit(t("diag.parallel_heading", backend=test["backend"], count=len(segments), seconds=f"{total:g}"), 0))
            self.job.log(f"    GPU jobs at the same time: {len(segments)} segments, {total:g} s of video{', key frames only' if keyframes else ''}: "
                         + ", ".join(seg["name"] for seg in segments), echo=not self.compact)
            previous = 0.0
            if len(segments) < 2: raise RuntimeError("only one segment was tested on the GPU, too few to try more than one job at a time")
            for jobs in range(1, min(self.PARALLEL_MAX_JOBS, len(segments)) + 1):          # a job needs a segment of its own
                gen._check_canceled()
                todo: queue.Queue[dict[str, Any]] = queue.Queue()
                for seg in sorted(segments, key=lambda x: x["duration"], reverse=True): todo.put(seg)
                failures: list[Any] = []
                canceled = threading.Event()

                def worker(index: int) -> None:
                    while not failures and not canceled.is_set():
                        try: seg = todo.get_nowait()
                        except queue.Empty: return
                        try:
                            with tempfile.TemporaryDirectory(prefix="parallel_", dir=self.job.path) as work:
                                result = self._extract(seg["source"], Path(work) / "frames", "GPU", seg["backend"], start_time=seg["start"], duration_limit=seg["duration"],
                                                       keyframes=keyframes, ignore_color_tag=seg["ignore_tag"], progress_label=f"{jobs} GPU job(s) at once | job {index + 1}")
                        except BaseException as exc:             # reported below, not lost in the thread
                            failures.append(exc)
                            return
                        if result.canceled: canceled.set()
                        elif not result.success: failures.append(result.error)

                began = time.monotonic()
                threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(jobs)]
                try:
                    for thread in threads: thread.start()
                    for thread in threads: thread.join()
                except KeyboardInterrupt:
                    gen.stop_event.set()
                    for thread in threads: thread.join()
                    raise
                wall = time.monotonic() - began
                if canceled.is_set(): raise InterruptedError("Canceled")
                if failures:
                    test["error"] = diagnostic_error_summary(str(failures[0]))
                    self.job.log(f"    {jobs} job{'' if jobs == 1 else 's'}: failed: {failures[0]}", echo=False)
                    self.job.say("diag.parallel_failed", count=jobs, error=test["error"])
                    break
                throughput = total / max(wall, 0.001)
                test["runs"].append({"jobs": jobs, "wall": round(wall, 3), "throughput": round(throughput, 3)})
                self.job.say("diag.parallel_result", count=jobs, wall=f"{wall:.1f}", throughput=f"{throughput:.2f}")
                if jobs >= 2 and throughput < previous * benchmark_advice.PARALLEL_MIN_GAIN: break       # one more job brings nothing
                previous = throughput
        except (KeyboardInterrupt, InterruptedError): raise
        except Exception as exc:
            test["error"] = diagnostic_error_summary(str(exc))
            self.job.say("diag.parallel_not_measured", error=exc)
        finally:
            for copy in staged:
                copy.remove()
                with suppress(OSError): copy.path.parent.rmdir()
        self.save(data)

    WARMUP_SECONDS = 10.0

    def _warm_up(self, meta: VideoMeta, mode: str, backends: list[str], data: dict[str, Any]) -> None:
        """One short throw-away GPU run per backend, so one-time driver start-up cost is not charged to the first measured file.

        Results are only logged (data["warmup"]); they are never part of the measurements, and a failure here is not an error.
        """
        gen = self.generator
        windows = diagnostic_segments(meta.duration, mode)
        if not backends or not windows: return
        start, duration = windows[0]
        for backend in backends:
            gen._check_canceled()
            try:
                with tempfile.TemporaryDirectory(prefix="warmup_", dir=self.job.path) as work:
                    result = self._extract(meta, Path(work) / "frames", "GPU", backend, start_time=start,
                                           duration_limit=min(duration, self.WARMUP_SECONDS), progress_label=f"Warm-up GPU/{backend}")
                if result.canceled: raise InterruptedError("Canceled")
                data["warmup"][backend] = {"status": "PASS" if result.success else "FAILED", "seconds": result.elapsed}
                self.job.log(f"Warm-up GPU/{backend}: {'ok' if result.success else 'failed'} in {result.elapsed:.2f}s (not counted)", echo=not self.compact)
            except (KeyboardInterrupt, InterruptedError): raise
            except Exception as exc:
                data["warmup"][backend] = {"status": "FAILED", "error": str(exc)}
                self.job.log(f"Warm-up GPU/{backend}: failed ({exc}); measurements continue", echo=not self.compact)

    def save(self, data: dict[str, Any]) -> None:
        with english(): self._save(data)                  # the report and result files are English whatever the screen shows

    def _save(self, data: dict[str, Any]) -> None:
        counts: dict[str, int] = {}
        for row in data["rows"]: counts[row["status"]] = counts.get(row["status"], 0) + 1
        data["technical_summary"] = counts
        lines = [f"{data['kind'].upper()} | {data['mode']} | {data['status']}",
                 "Technical results: " + ", ".join(f"{k}={v}" for k, v in counts.items()), "",
                 *system_info.report_lines(data.get("system") or {}), ""]
        if data["kind"] == "benchmark":
            lines.extend(benchmark_speed_table(data))
            lines.extend(["DETAILED MEASUREMENTS", ""])
        for row in data["rows"]:
            line = (f"{Path(row['video']).name} | {row['backend']} | start={row['start_seconds']:g}s "
                    f"duration={row['duration_seconds']:g}s | {row['status']} | "
                    f"extract={row.get('extract_seconds', 0):.2f}s pack={row.get('pack_seconds', 0):.2f}s "
                    f"frames={row.get('frames', 0)}")
            if "total_speedup" in row:
                line += f" | CPU/path speedup={row['total_speedup']:.2f}x"
            if row.get("error"):
                line += f" | {row.get('error_summary') or diagnostic_error_summary(row['error'])} Details: diagnostic log / result.json"
            lines.append(line)
        data["recommendations"] = {}
        if data["status"] == "COMPLETE" and data["kind"] == "benchmark":
            for video in dict.fromkeys(r["video"] for r in data["rows"]):
                rows = [r for r in data["rows"] if r["video"] == video]
                cpu = [r for r in rows if r["backend"] == "cpu"]
                if not cpu or any(r["status"] != "PASS" for r in cpu): continue
                cpu_time = sum(r["extract_seconds"] + r["pack_seconds"] for r in cpu)
                best, best_ratio = "cpu", 1.0
                threshold = float(self.generator.config["acceleration"].get("gpu_speed_threshold", 1.15))
                for backend in data["backends"]:
                    runs = [r for r in rows if r["backend"] == backend]
                    if len(runs) != len(cpu) or any(r["status"] != "PASS" or not r.get("matching_frame_count") for r in runs): continue
                    ratio = cpu_time / max(sum(r["extract_seconds"] + r["pack_seconds"] for r in runs), 0.000001)
                    if ratio >= threshold and ratio > best_ratio: best, best_ratio = backend, ratio
                data["recommendations"][video] = {"requested_path": best, "speedup": best_ratio,
                    "scope": "Speed recommendation for tested file and settings only"}
                lines.append(f"Recommendation: {Path(video).name} | {best} | {best_ratio:.2f}x (tested file/settings only)")
        if data["kind"] == "hdr":
            lines.extend(["", "Visual assessments: " + json.dumps(data["visual"], ensure_ascii=False)])
        else:
            data["notes"] = [note for note in data["notes"] if not note.startswith(("HDR images", "CPU/GPU frame slots", "No HDR video"))]
        lines.extend(["", *data["notes"]])
        if data["kind"] == "benchmark":
            lines.extend(benchmark_totals(data))
            limit = float(self.generator.config.get("trickplay", {}).get("keyframe_repeat_limit_percent", benchmark_advice.DEFAULT_REPEAT_LIMIT))
            lines.extend(benchmark_advice.path_lines(data, self.generator.config))
            lines.extend(benchmark_advice.keyframe_lines(data, limit))
            lines.extend(benchmark_advice.storage_lines(data))
            lines.extend(benchmark_advice.parallel_lines(data))
            lines.extend(library_sample.read_speed_lines(data.get("read_speed", [])))
            if data.get("direct_files"):
                lines.extend(["", "TESTED ON THE LIBRARY DRIVE (the segment could not be copied; the timings include reading it): "
                              + ", ".join(Path(p).name for p in data["direct_files"])])
            tagged = list(dict.fromkeys(Path(r["video"]).name for r in data["rows"] if r.get("color_tag_ignored")))
            if tagged:
                lines.extend(["", "COLOUR SPACE TAG IGNORED (FFmpeg refuses the tag in these files; Generate ignores it in the same way): " + ", ".join(tagged)])
            if data["status"] == "COMPLETE":
                advice = benchmark_advice.build_advice(data, self.generator.config)
                data["suggested_settings"] = [{"setting": f"{a.section}.{a.key}", "now": str(a.current), "suggested": str(a.proposed), "why": a.reason} for a in advice]
                if advice:
                    lines.extend(["", "SUGGESTED SETTINGS (nothing is changed without your confirmation)", *benchmark_advice.advice_lines(advice)])
        self.job.save("\n".join(lines), data)

class DiagnosticJob:
    def __init__(self, kind: str) -> None:
        base = DIRS["diagnostics"] / kind
        base.mkdir(parents=True, exist_ok=True)
        existing_nums: list[int] = []
        for p in base.iterdir():
            if p.is_dir():
                m = re.search(r"_job_(\d+)$", p.name)
                if m: existing_nums.append(int(m.group(1)))
        num = max(existing_nums, default=0) + 1
        self.path = base / f"{local_stamp()}_job_{num:04d}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.log_path = self.path / f"{kind}.log"
        self.result_txt = self.path / "result.txt"
        self.result_json = self.path / "result.json"
        self._fh = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self.kind = kind
        self.log(f"{APP_NAME} {APP_VERSION} | diagnostic={kind}", echo=False)            # for the log file; the screen has its header
        ACTIVITY.log("DIAGNOSTIC", f"{kind} started | report: {self.path}")

    def log(self, message: str, echo: bool = True) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {message}"
        self._fh.write(line + "\n")
        self._fh.flush()
        if echo: print(message)

    def say(self, key: str, *, echo: bool = True, prefix: str = "", count: Optional[int] = None, **values: Any) -> str:
        """A line for the screen and the log file: the screen has it in the language of the person, the file always in English.
        With `count` the text is a plural one. Returns the English text (reports and result files are English as well)."""
        def make() -> str: return prefix + (t(key, **values) if count is None else tp(key, count, **values))
        with english(): in_english = make()
        self.log(in_english, echo=False)
        if echo: print(make())
        return in_english

    def save(self, text: str, data: dict[str, Any]) -> None:
        atomic_write_text(self.result_txt, text.rstrip() + "\n")
        atomic_write_json(self.result_json, data)

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except Exception: pass
        self._fh.close()
        try: outcome = self.result_txt.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError): outcome = "no result saved"
        ACTIVITY.log("DIAGNOSTIC", f"{self.kind} finished | {outcome} | report: {self.path}")

def create_contact_sheet(pairs: list[tuple[Path, Optional[Path]]], dest: Path, title: str) -> None:
    """CPU frames on the left and GPU frames on the right; with no GPU frames (None) only the CPU column is drawn."""
    if not PIL_AVAILABLE: raise RuntimeError("Pillow is required")
    images: list[tuple[Any, Any]] = []
    max_w = 0
    row_heights: list[int] = []
    for cpu, gpu in pairs:
        with Image.open(cpu) as source: a = source.convert("RGB")
        b = None
        if gpu is not None:
            with Image.open(gpu) as source: b = source.convert("RGB")
        images.append((a, b))
        max_w = max(max_w, a.width, b.width if b else 0)
        row_heights.append(max(a.height, b.height if b else 0) + 26)
    columns = 2 if any(b is not None for _, b in images) else 1
    header = 70
    canvas = Image.new("RGB", (max_w * columns + 10 + 10 * columns, header + sum(row_heights)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black")
    draw.text((10, 35), "CPU", fill="black")
    if columns == 2: draw.text((max_w + 20, 35), "GPU", fill="black")
    y = header
    for idx, ((a, b), rh) in enumerate(zip(images, row_heights), 1):
        draw.text((2, y), f"{idx:02d}", fill="black")
        canvas.paste(a, (10, y + 20))
        if b is not None: canvas.paste(b, (max_w + 20, y + 20))
        y += rh
        a.close()
        if b is not None: b.close()
    canvas.save(dest, "JPEG", quality=92)


def create_paths_sheet(columns: list[str], rows: list[tuple[str, list[Optional[Path]]]], dest: Path, title: str) -> None:
    """A column per path and a row per segment; a path without a picture (it failed) is marked instead."""
    if not PIL_AVAILABLE: raise RuntimeError("Pillow is required")
    images: list[list[Any]] = []
    cell_w = cell_h = 0
    for _, cells in rows:
        line: list[Any] = []
        for cell in cells:
            if cell is None: line.append(None)
            else:
                with Image.open(cell) as source: line.append(source.convert("RGB"))
        images.append(line)
        for image in line:
            if image is not None: cell_w, cell_h = max(cell_w, image.width), max(cell_h, image.height)
    cell_w, cell_h = cell_w or 160, cell_h or 90
    label_w, header, pad = 130, 70, 10
    canvas = Image.new("RGB", (label_w + (cell_w + pad) * len(columns) + pad, header + (cell_h + 26) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black")
    for index, name in enumerate(columns):
        draw.text((label_w + pad + (cell_w + pad) * index, 45), name, fill="black")
    for row_no, ((label, _), line) in enumerate(zip(rows, images)):
        y = header + (cell_h + 26) * row_no
        draw.text((10, y + 20), label, fill="black")
        for index, image in enumerate(line):
            x = label_w + pad + (cell_w + pad) * index
            if image is None: draw.text((x, y + 20), "no picture (failed)", fill="red")
            else:
                canvas.paste(image, (x, y + 20))
                image.close()
    canvas.save(dest, "JPEG", quality=92)
