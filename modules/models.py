# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from . import media_classification

if TYPE_CHECKING:
    from .verify import Restorable


@dataclass
class VideoMeta:
    path: Path
    size: int
    mtime_ns: int
    duration: float
    codec: str
    profile: str
    pix_fmt: str
    width: int
    height: int
    hdr: bool                    # HDR, including "HDR Uncertain" (HDR with a few details missing)
    bits: int = 0                # component bit depth as probed (0 = unknown, use the pixel format)
    chroma: str = ""
    resolution: str = ""         # resolution class as probed (empty = compute from the size)
    hdr_pool: str = "SDR"        # SDR | SDR Uncertain | HDR | HDR Uncertain
    hdr_type: str = ""           # SDR | HDR10 / PQ | HDR10+ | HLG | Dolby Vision | "" (HDR, type not confirmed)
    dv_profile: str = ""
    hdr_note: str = ""           # why the class is "Uncertain"

    @property
    def bit_depth(self) -> int:
        return self.bits or media_classification.bit_depth(None, self.pix_fmt) or 8

    @property
    def resolution_class(self) -> str:
        return self.resolution or media_classification.resolution_class(self.width, self.height) or "SD"

    @property
    def signature(self) -> tuple[int, int]: return self.size, self.mtime_ns

@dataclass(frozen=True)
class TrickplaySettings:
    width: int
    interval: float
    cols: int
    rows: int
    qscale: int
    jpeg_quality: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("Trickplay interval must be finite and positive")
        if self.width <= 0 or self.cols <= 0 or self.rows <= 0:
            raise ValueError("Trickplay width and grid dimensions must be positive")
        if not 1 <= self.qscale <= 31 or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("Invalid JPEG quality settings")

    @property
    def folder_name(self) -> str: return f"{self.width} - {self.cols}x{self.rows}"

    @property
    def hash(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @property
    def capacity(self) -> int: return self.cols * self.rows

@dataclass
class ValidationResult:
    valid: bool
    reason: str
    tiles: int = 0
    output_bytes: int = 0
    tile_size: Optional[tuple[int, int]] = None
    unexpected: list[str] = field(default_factory=list)

@dataclass
class ExtractResult:
    success: bool
    method: str
    backend: Optional[str]
    elapsed: float
    frames: int
    error: str = ""
    unsupported: bool = False
    canceled: bool = False
    fallback: bool = False
    stalled: bool = False          # FFmpeg stopped making progress and was killed: not retried, not tried on the CPU either
    keyframe_only: bool = False      # only key frames were decoded (the option in the trickplay settings; off after a failure of that way)
    color_tag_ignored: bool = False   # the extraction that worked had to ignore the colour space tag (a later extraction of the same file does the same)
    attempts: list[str] = field(default_factory=list)

@dataclass
class PlanItem:
    meta: VideoMeta
    action: str
    reason: str
    validation: Optional[ValidationResult] = None
    source_changed: bool = False
    other_widths: list[Path] = field(default_factory=list)
    storage_group: str = ""
    other_policy: Optional[str] = None
    restorable: Optional["Restorable"] = None      # RESTORE: the valid previous output that takes the place of a missing/damaged one
