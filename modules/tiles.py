# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from .models import TrickplaySettings, ValidationResult, VideoMeta

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore[assignment]
    PIL_AVAILABLE = False


def config_settings(config: dict[str, Any]) -> TrickplaySettings:
    t = config["trickplay"]
    return TrickplaySettings(width=int(t["width"]), interval=float(t["interval_seconds"]), cols=int(t["tile_columns"]),
                             rows=int(t["tile_rows"]), qscale=int(t["ffmpeg_qscale"]), jpeg_quality=int(t["jpeg_quality"]))

def trickplay_root_for(video: Path) -> Path: return video.with_name(video.stem + ".trickplay")
def target_dir_for(video: Path, settings: TrickplaySettings) -> Path: return trickplay_root_for(video) / settings.folder_name
def building_dir_for(video: Path, settings: TrickplaySettings) -> Path: return trickplay_root_for(video) / (settings.folder_name + ".__building__")

def parse_trickplay_width_dir(path: Path) -> Optional[tuple[int, int, int]]:
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)x(\d+)\s*", path.name, re.IGNORECASE)
    if not m: return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def list_other_widths(video: Path, settings: TrickplaySettings) -> list[Path]:
    root = trickplay_root_for(video)
    if not root.is_dir(): return []
    out: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir() or child.name.endswith(".__building__"): continue
        parsed = parse_trickplay_width_dir(child)
        if parsed and child.name != settings.folder_name: out.append(child)
    return out

def expected_foreign_tile_counts(duration: float, settings: TrickplaySettings) -> set[int]:
    if duration <= 0: return set()
    samples = max(1, int(math.floor(duration / settings.interval)))
    counts = {max(1, int(math.ceil(samples / settings.capacity)))}
    for delta in (-1, 1):
        s = max(1, samples + delta)
        counts.add(max(1, int(math.ceil(s / settings.capacity))))
    return counts

def frame_count_deviation(frames: int, duration: float, interval: float) -> Optional[tuple[int, int]]:
    """None when the number of thumbnails fits the length of the video; (expected, frames) when it clearly does not.

    A thumbnail is taken every `interval` seconds, so a video of `duration` seconds should give about duration / interval of them
    (one more or less is normal). Jellyfin keeps no time for a single thumbnail, only the interval, so a different number puts every
    later thumbnail at the wrong time. Typical causes are a video stream shorter than the length the container states, and
    damaged time stamps (the FFmpeg `fps` filter then repeats or drops pictures). Both are in the file, not in this program."""
    if duration <= 0 or interval <= 0: return None
    expected = max(1, round(duration / interval))
    if abs(frames - expected) <= max(3, math.ceil(expected * 0.01)): return None
    return expected, frames

def describe_frame_deviation(expected: int, frames: int, interval: float) -> str:
    """One line for people: "449 thumbnails instead of about 559 (the video seems to end about 4,490 s in, 110 thumbnails short)"."""
    if frames < expected:
        return f"{frames:,} thumbnails instead of about {expected:,} (the video stream seems to end about {frames * interval:,.0f} s in; {expected - frames:,} short)"
    return f"{frames:,} thumbnails instead of about {expected:,} ({frames - expected:,} too many; the time stamps in the file are probably damaged)"

def validate_tile_directory(directory: Path, settings: TrickplaySettings, duration: float, *, expected_exact_tiles: Optional[int] = None, verify_all: bool = True) -> ValidationResult:
    if not PIL_AVAILABLE: return ValidationResult(False, "Pillow is not installed")
    if not directory.is_dir(): return ValidationResult(False, "missing directory")
    numeric: dict[int, Path] = {}
    unexpected: list[str] = []
    try: children = list(directory.iterdir())
    except OSError as exc: return ValidationResult(False, f"cannot read directory: {exc}")
    for child in children:
        if child.is_file() and re.fullmatch(r"\d+\.jpg", child.name, re.IGNORECASE): numeric[int(child.stem)] = child
        else: unexpected.append(child.name)
    if unexpected: return ValidationResult(False, "unexpected files/folders present", unexpected=sorted(unexpected))
    if not numeric: return ValidationResult(False, "no numeric JPG tiles")
    indices = sorted(numeric)
    if indices != list(range(len(indices))): return ValidationResult(False, "missing or non-sequential tile numbers", tiles=len(indices))
    if expected_exact_tiles is not None:
        if len(indices) != expected_exact_tiles: return ValidationResult(False, f"expected {expected_exact_tiles} tiles, found {len(indices)}", tiles=len(indices))
    else:
        allowed = expected_foreign_tile_counts(duration, settings)
        if allowed and len(indices) not in allowed: return ValidationResult(False, f"tile count {len(indices)} outside expected {sorted(allowed)}", tiles=len(indices))
    inspect_paths = [numeric[i] for i in indices] if verify_all else [numeric[indices[0]], numeric[indices[-1]]]
    expected_size: Optional[tuple[int, int]] = None
    output_bytes = 0
    for p in numeric.values():
        try:
            st = p.stat()
            if st.st_size <= 0: return ValidationResult(False, f"zero-byte tile: {p.name}", tiles=len(indices))
            output_bytes += st.st_size
        except OSError as exc: return ValidationResult(False, f"cannot stat tile {p.name}: {exc}", tiles=len(indices))
    for p in inspect_paths:
        try:
            with Image.open(p) as im: im.verify()
            with Image.open(p) as im:
                size = im.size
                if size[0] != settings.width * settings.cols: return ValidationResult(False, f"wrong sprite width in {p.name}: {size[0]}", tiles=len(indices))
                if size[1] <= 0 or size[1] % settings.rows != 0: return ValidationResult(False, f"invalid sprite height in {p.name}: {size[1]}", tiles=len(indices))
                if expected_size is None: expected_size = size
                elif size != expected_size: return ValidationResult(False, f"inconsistent sprite dimensions: {p.name}", tiles=len(indices))
        except Exception as exc:
            why = "not a valid image" if type(exc).__name__ == "UnidentifiedImageError" else str(exc)            # Pillow's own text repeats the whole path
            return ValidationResult(False, f"unreadable JPEG {p.name}: {why}", tiles=len(indices))
    return ValidationResult(True, "OK", tiles=len(indices), output_bytes=output_bytes, tile_size=expected_size)

def quick_validate_tile_directory(directory: Path, *, tiles: int, output_bytes: int) -> Optional[ValidationResult]:
    """Cheap check of an output this program wrote itself: only the names and sizes of the files are looked at, no picture is
    read. Valid only when the folder holds exactly the tiles 0.jpg .. N-1.jpg and their total size is the recorded one.
    None when that cannot be confirmed (the caller then does the full check, which also says what is wrong)."""
    if tiles <= 0 or output_bytes <= 0: return None
    names: set[str] = set()
    total = 0
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False): return None
                size = entry.stat(follow_symlinks=False).st_size
                if size <= 0: return None
                names.add(entry.name)
                total += size
    except OSError:
        return None
    if names != {f"{i}.jpg" for i in range(tiles)} or total != output_bytes: return None
    return ValidationResult(True, "OK", tiles=tiles, output_bytes=total)

def hdr_kind(meta: VideoMeta) -> str:
    """The kind of HDR as a short name: "HDR10", "HDR10+", "HLG", "Dolby Vision P8", "HDR" (type not confirmed).
    "" for SDR. Test results of the HDR check are kept per kind."""
    if not meta.hdr: return ""
    if meta.hdr_type == "Dolby Vision": return "Dolby Vision" + (f" P{meta.dv_profile}" if meta.dv_profile else "")
    return {"HDR10 / PQ": "HDR10"}.get(meta.hdr_type, meta.hdr_type) or "HDR"


def hdr_label(meta: VideoMeta) -> str:
    """hdr_kind for lists; "(uncertain)" is added when a few details of the HDR signalling are missing or unusual."""
    return hdr_kind(meta) + (" (uncertain)" if meta.hdr_pool == "HDR Uncertain" else "")


def classify_video(meta: VideoMeta) -> str:
    codec = meta.codec.lower()
    if codec in {"h264", "avc1"}: family = "H.264"
    elif codec in {"hevc", "h265", "hev1", "hvc1"}: family = "HEVC"
    elif codec in {"av1"}: family = "AV1"
    else: family = codec.upper() if codec else "UNKNOWN"
    depth = meta.bit_depth
    res = meta.resolution_class
    if meta.hdr: tail = " " + hdr_label(meta)
    elif depth >= 10 or res in {"4K", "8K", ">8K"}: tail = " SDR"                    # where HDR would be expected, say it is not
    else: tail = ""
    return f"{family} {depth}-bit {res}{tail}"

def check_path_writeable(root: Path) -> tuple[bool, str]:
    if not root.exists(): return False, "missing/offline"
    test = root / f".trickplay_write_test_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    try:
        with open(test, "wb") as fh:
            fh.write(b"ok")
            fh.flush()
            os.fsync(fh.fileno())
        test.unlink()
        return True, "OK"
    except Exception as exc:
        try: test.unlink(missing_ok=True)
        except Exception: pass
        return False, str(exc)

def folder_size(path: Path) -> int:
    total = 0
    if not path.exists(): return 0
    for root, dirs, files in os.walk(path):
        for name in files:
            try: total += (Path(root) / name).stat().st_size
            except OSError: pass
    return total
