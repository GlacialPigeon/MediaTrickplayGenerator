# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""Test files taken from the person's own library.

The benchmark does not need a special test folder: it picks a handful of files from the media folders, one segment of
each from the middle, and copies that segment (stream copy, nothing is re-encoded) into the program's own `temp` folder
before measuring. That way the drive the library lives on (a NAS, a slow disk) does not distort the CPU / GPU timings, and
the time of the copy is the read speed of that drive, which is worth knowing as well.
"""
from __future__ import annotations

import math
import random
import shutil
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from .error_summary import compact_ffmpeg_error, error_cause
from .models import VideoMeta
from .tiles import hdr_label

MIN_SAMPLE_FILES = 10      # at least this many files are tested (a library with few kinds of video gets several files of the same kind)
MAX_KINDS = 30             # one file per kind of video found in the library, up to this many kinds (the most common ones first)
RANDOM_FILES = 2           # on top of those: random files, which show whether the kinds stand for the whole library
SAMPLE_FILES = MIN_SAMPLE_FILES
SEGMENT_SECONDS = 120.0    # the part of each file that is tested
PREFER_LONGER_THAN = 180.0 # a file shorter than this is taken only when nothing longer of its kind exists


def kind_key(meta: VideoMeta) -> str:
    """What makes videos behave differently for a decoder: codec, bit depth, picture size and HDR type."""
    return f"{meta.codec} {meta.bit_depth}-bit {meta.resolution_class} {hdr_label(meta) if meta.hdr else 'SDR'}"


def aligned_kinds(kinds: list[str]) -> dict[str, str]:
    """The kinds of video written in columns that line up in a table: codec, bit depth, picture size, HDR type
    ("h264  8-bit   SD    SDR" under "av1   10-bit  1080p  SDR"). A text that is not a kind stays as it is."""
    split = {kind: kind.split(" ", 3) for kind in kinds}
    widths = [max((len(parts[i]) for parts in split.values() if len(parts) == 4), default=0) for i in range(3)]
    return {kind: ("  ".join(part.ljust(widths[i]) for i, part in enumerate(parts[:3])) + "  " + parts[3]) if len(parts) == 4 else kind for kind, parts in split.items()}


@dataclass
class SampleFile:
    meta: VideoMeta
    why: str                   # the kind of video it stands for, or "random"


def _usable(meta: VideoMeta) -> bool:
    return math.isfinite(meta.duration) and meta.duration > 0 and meta.width > 0 and meta.height > 0


def _one_of(rng: random.Random, candidates: list[VideoMeta]) -> VideoMeta:
    longer = [m for m in candidates if m.duration >= PREFER_LONGER_THAN]
    return rng.choice(sorted(longer or candidates, key=lambda m: str(m.path).casefold()))


def pick_sample(videos: list[VideoMeta], count: Optional[int] = None, stratified: Optional[int] = None,
                rng: Optional[random.Random] = None) -> list[SampleFile]:
    """One file for every kind of video in the library (every codec first, then the most common kinds; at most MAX_KINDS), plus RANDOM_FILES random
    ones, and at least MIN_SAMPLE_FILES files in all. `count` and `stratified` set the numbers by hand."""
    rng = rng or random.Random()
    usable = [m for m in videos if _usable(m)]
    groups: dict[str, list[VideoMeta]] = {}
    for meta in usable: groups.setdefault(kind_key(meta), []).append(meta)
    ranked = sorted(groups, key=lambda key: (-len(groups[key]), key))
    if stratified is None: stratified = min(len(groups), MAX_KINDS)
    if count is None: count = max(MIN_SAMPLE_FILES, stratified + RANDOM_FILES)
    order: list[str] = []
    seen_codecs: set[str] = set()
    for key in ranked:                                   # the most common kind of every codec first, so a rare codec is not left out
        codec = groups[key][0].codec
        if codec not in seen_codecs:
            seen_codecs.add(codec)
            order.append(key)
    order += [key for key in ranked if key not in order]
    chosen = [SampleFile(_one_of(rng, groups[key]), key) for key in order[:min(stratified, count)]]
    taken = {s.meta.path for s in chosen}
    rest = [m for m in usable if m.path not in taken]
    preferred = [m for m in rest if m.duration >= PREFER_LONGER_THAN]
    pool = preferred if len(preferred) >= count - len(chosen) else rest
    extra = rng.sample(sorted(pool, key=lambda m: str(m.path).casefold()), min(count - len(chosen), len(pool)))
    chosen += [SampleFile(meta, "random: " + kind_key(meta)) for meta in extra]
    return chosen


def segment_of(meta: VideoMeta) -> tuple[float, float]:
    """(start, length) of the tested part: two minutes from the middle, or the whole file when it is shorter."""
    length = min(SEGMENT_SECONDS, meta.duration)
    return round((meta.duration - length) / 2, 3), length


def segment_bytes(meta: VideoMeta, length: float) -> int:
    return int(meta.size * length / meta.duration) if meta.duration > 0 else meta.size


def estimate_seconds(files: list[VideoMeta], path_count: int) -> tuple[int, int]:
    """A rough range for the whole benchmark: every segment is decoded on every path (and once more with key frames only); a path gets
    through between about 6 and 60 seconds of video per second (small SD files are fast, big 4K files slow), and the copy reads 40 to
    200 MB per second."""
    video_seconds = sum(segment_of(m)[1] for m in files) * (path_count + 1)
    copy_bytes = sum(segment_bytes(m, segment_of(m)[1]) for m in files)
    return (int(video_seconds / 60 + copy_bytes / (200 * 1024 * 1024)), int(video_seconds / 6 + copy_bytes / (40 * 1024 * 1024)))


@dataclass
class StagedSegment:
    meta: VideoMeta            # the local copy, with the properties of the original
    start: float               # where to start in the copy (always the beginning)
    duration: float
    read_seconds: float
    read_bytes: int
    path: Path

    @property
    def read_speed(self) -> float:
        return self.read_bytes / max(self.read_seconds, 0.001)

    def remove(self) -> None:
        with suppress(OSError): self.path.unlink()


def stage_segment(ffmpeg: Path, meta: VideoMeta, start: float, length: float, folder: Path, timeout: float = 1800.0) -> StagedSegment:
    """Copy `length` seconds from `start` of the video stream into `folder` (stream copy). Raises RuntimeError when that is not possible."""
    folder.mkdir(parents=True, exist_ok=True)
    need = int(segment_bytes(meta, length) * 1.3) + 64 * 1024 * 1024
    free = shutil.disk_usage(folder).free
    if free < need: raise RuntimeError(f"Not enough free space for the copy of the segment: need about {need // (1024 * 1024)} MB, free {free // (1024 * 1024)} MB.")
    dest = folder / "benchmark_segment.mkv"
    with suppress(OSError): dest.unlink()
    cmd = [str(ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "error", "-fflags", "+genpts", "-ss", f"{start:.3f}", "-i", str(meta.path), "-t", f"{length:.3f}",
           "-map", "0:v:0", "-c", "copy", "-an", "-sn", "-dn", "-avoid_negative_ts", "make_zero", "-y", str(dest)]
    began = time.monotonic()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        with suppress(OSError): dest.unlink()
        raise RuntimeError("Copying the segment took too long (is the drive or share answering?).") from exc
    except OSError as exc:
        raise RuntimeError(f"Copying the segment failed: {exc}") from exc
    elapsed = time.monotonic() - began
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size <= 0:
        with suppress(OSError): dest.unlink()
        raise RuntimeError("Copying the segment failed: " + (error_cause(compact_ffmpeg_error(proc.stderr.strip())) or f"FFmpeg exited with code {proc.returncode}"))
    stat = dest.stat()
    local = replace(meta, path=dest, size=stat.st_size, mtime_ns=stat.st_mtime_ns, duration=length)
    return StagedSegment(local, 0.0, length, elapsed, stat.st_size, dest)


def read_speed_lines(rows: list[dict[str, Any]]) -> list[str]:
    """The read speed of the drive the library is on, from the time the copies took."""
    if not rows: return []
    lines = ["", "SOURCE READ SPEED (time to copy the tested segment from the library; includes starting FFmpeg and writing the copy)"]
    for row in rows:
        lines.append(f"  {Path(row['video']).name} | {row['bytes'] / (1024 * 1024):.0f} MB in {row['seconds']:.1f} s = {row['mb_per_second']:.1f} MB/s")
    total_bytes, total_seconds = sum(r["bytes"] for r in rows), sum(r["seconds"] for r in rows)
    lines.append(f"  Together: {total_bytes / (1024 * 1024):.0f} MB in {total_seconds:.1f} s = {total_bytes / (1024 * 1024) / max(total_seconds, 0.001):.1f} MB/s")
    return lines
