# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""How long Generate will take, roughly: from what the benchmark measured per kind of video, else from the size of the pictures.

One job gets through `rate` seconds of video per second on its path (the benchmark measures it per kind of video). Without a
measurement a rate typical for the picture size is used, which makes the range wide. Key frames only skip most of the work;
how much depends on the videos, so the lower end of the range counts them and the upper end does not. A second job at the same
time helps little on a graphics card (the benchmark finds out how little), so at most two count for the lower end.

A hard disk can be the slower part: a video is read whole whatever the graphics card does, and a hard disk delivers only part of
the speed it has on a long, straight read (the heads move between the files and the tiles written next to them; in a run of 66 GB
from a hard disk reading at up to 137 MB/s the average was 67 MB/s). So the time can not be shorter than the size of the videos on
the disk divided by that average. An SSD is never the limit.
"""
from __future__ import annotations

from typing import Callable, Optional

from .localization import t, tp
from .models import VideoMeta

# Seconds of video per second for one job, by picture height: typical for the CPU and the GPU in the benchmark of a mid-range system.
CPU_RATES = ((576, 100.0), (720, 30.0), (1080, 13.0), (1440, 8.0), (10_000, 3.0))
GPU_FACTOR = 3.0
OVERHEAD_SECONDS = 1.0           # starting FFmpeg, reading the video's index, writing the files
SLOW_MARGIN = 1.3                # network drives, antivirus and the like
DEFAULT_KEYFRAME_GAIN = 2.0
DEFAULT_HDD_MB_PER_SECOND = 110.0   # what a hard disk reads at on a straight read, when the benchmark has not measured the drive
HDD_EFFECTIVE_SHARE = 0.6           # the part of it that a run gets on average
DISK_MARGIN = 1.15                  # the upper end counts a disk that is a little slower than the average


def fallback_rate(meta: VideoMeta) -> float:
    rate = next(rate for height, rate in CPU_RATES if meta.height <= height)
    return rate * GPU_FACTOR


def disk_seconds(kind: str, size_bytes: float, measured_mb_per_second: Optional[float] = None) -> float:
    """The least time it takes to read this much from one disk of this kind: nothing for an SSD, for a hard disk the size divided by
    the average speed of a run (`measured_mb_per_second` is what the benchmark read from the drive, else a typical speed)."""
    if kind != "hdd" or size_bytes <= 0: return 0.0
    speed = measured_mb_per_second if measured_mb_per_second and measured_mb_per_second > 0 else DEFAULT_HDD_MB_PER_SECOND
    return size_bytes / (speed * HDD_EFFECTIVE_SHARE * 1024 * 1024)


def estimate(videos: list[VideoMeta], rate_of: Callable[[VideoMeta], Optional[float]], jobs: int, *, keyframes: bool,
             keyframe_gain: float = DEFAULT_KEYFRAME_GAIN, disk_floor: float = 0.0) -> tuple[float, float, bool]:
    """(fastest, slowest, measured) in seconds; `measured` is False when some of it rests on typical rates, not on the benchmark.
    `disk_floor`: the time the slowest disk needs to read its videos (disks work side by side, so the slowest one counts)."""
    total, measured = 0.0, True
    for meta in videos:
        rate = rate_of(meta)
        if rate is None:
            rate, measured = fallback_rate(meta), False
        total += max(meta.duration, 0.0) / rate + OVERHEAD_SECONDS
    fast = total / (max(1.0, keyframe_gain) if keyframes else 1.0) / max(1, min(jobs, 2))
    slow = total * SLOW_MARGIN
    return max(fast, disk_floor), max(slow, disk_floor * DISK_MARGIN), measured


def describe(fast: float, slow: float, measured: bool) -> str:
    """"about 3 h 20 min" when the two ends are close, "about 2 to 6 hours" when they are not."""
    def rounded(seconds: float) -> tuple[float, str]:
        if seconds < 90: return max(1.0, round(seconds)), "seconds"
        if seconds < 5400: return max(1.0, round(seconds / 60)), "minutes"
        return seconds / 3600, "hours"
    def word(unit: str, amount: float) -> str:
        count = int(round(amount))
        if unit == "seconds": return tp("time.unit_seconds", count)
        if unit == "minutes": return tp("time.unit_minutes", count)
        return tp("time.unit_hours", count)
    if slow < 60: return t("time.less_than_minute")
    if measured and slow <= fast * 1.5:
        mid = (fast + slow) / 2
        if mid >= 3600:
            minutes = int(round(mid / 600)) * 10
            return t("time.about_hours_minutes", hours=minutes // 60, minutes=f"{minutes % 60:02d}")
        return t("time.about_minutes", count=max(1, round(mid / 60)))
    low, unit = rounded(fast)
    high, high_unit = rounded(slow)
    if unit != high_unit:
        fast_text = f"{low:g} {word(unit, low)}" if unit != "hours" else f"{low:.0f} {word('hours', low)}"
        return t("time.about_range_units", low=fast_text, high=f"{high:.0f} {word(high_unit, high)}")
    low_text, high_text = (f"{low:.0f}", f"{high:.0f}")
    if low_text == high_text: return t("time.about_same", value=high_text, unit=word(high_unit, high))
    return t("time.about_range_same", low=low_text, high=high_text, unit=word(high_unit, high))
