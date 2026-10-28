# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import copy
import platform as host_info
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from . import platform_api
from .constants import APP_VERSION, DIRS
from .datetime_helpers import utc_now_iso
from .diagnostics import DiagnosticJob
from .ffmpeg_registry import FFmpegCandidate
from .ffmpeg_tools import FFmpegTools
from .generator import TrickplayGenerator
from .logger import NullRunLogger
from .state_db import StateDB
from .tiles import PIL_AVAILABLE, check_path_writeable, validate_tile_directory

SYNTHETIC_SECONDS = 25


def run_self_test(config: dict[str, Any], state: StateDB, candidate: Optional[FFmpegCandidate],
                  tools: Optional[FFmpegTools], job: DiagnosticJob,
                  progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Environment and pipeline checks. Returns {"checks": {name: {status, detail}}, "passed": n, "failed": n}."""
    results: dict[str, Any] = {"version": APP_VERSION, "date": utc_now_iso(), "checks": {}}

    def check(name: str, fn: Callable[[], Any], shown: Optional[str] = None) -> bool:
        """`shown`: what the screen says for a pass instead of the detail (paths belong in the report, not on the screen)."""
        try:
            value = fn()
            results["checks"][name] = {"status": "PASS", "detail": str(value)}
            job.log(f"PASS {name}: {value}", echo=False)
            progress(f"  PASS  {name}" + (f": {shown if shown is not None else value}" if shown != "" else ""))
            return True
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            results["checks"][name] = {"status": "FAIL", "detail": detail}
            job.log(f"FAIL {name}: {detail}", echo=False)
            progress(f"  FAIL  {name}: {detail}")
            return False

    def need(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    check("platform", lambda: f"{platform_api.platform().name} | {host_info.platform()} | Python {host_info.python_version()}")

    def pillow() -> str:
        need(PIL_AVAILABLE, "Pillow is not installed (python -m pip install pillow)")
        return "available"
    check("Pillow", pillow)

    for name, path in DIRS.items():
        def writable(p: Path = path) -> str:
            ok, message = check_path_writeable(p)
            need(ok, message)
            return str(p)
        check(f"directory:{name}", writable, shown="")

    check("config", lambda: f"schema={config.get('schema_version')}")

    def database() -> str:
        row = state._execute("PRAGMA quick_check").fetchone()
        need(row is not None and row[0] == "ok", f"quick_check returned {row[0] if row else None}")
        return "quick_check ok"
    check("state_db", database)

    if candidate is None or tools is None:
        results["checks"]["ffmpeg"] = {"status": "FAIL", "detail": "FFmpeg / FFprobe not found"}
        job.log("FAIL ffmpeg: not found", echo=False)
        progress("  FAIL  ffmpeg: not found")
    else:
        check("ffmpeg", lambda: (candidate.version_line or "unknown version").split(" Copyright")[0])
        check("ffprobe", lambda: str(candidate.ffprobe), shown="")
        check("hardware backends", lambda: ", ".join(tools.backends()) or "none (CPU only)")
        _pipeline_check(config, state, tools, job, check, need)

    passed = sum(1 for c in results["checks"].values() if c["status"] == "PASS")
    failed = sum(1 for c in results["checks"].values() if c["status"] == "FAIL")
    results["passed"], results["failed"] = passed, failed
    return results


def _pipeline_check(config: dict[str, Any], state: StateDB, tools: FFmpegTools, job: DiagnosticJob,
                    check: Callable[[str, Callable[[], Any]], bool], need: Callable[[bool, str], None]) -> None:
    """Create a short synthetic video, probe it, then run the real CPU extract -> pack -> validate path on it."""
    work = DIRS["temp"] / f"selftest_{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=True)
    synthetic = work / "synthetic.mp4"
    holder: dict[str, Any] = {}

    def make_video() -> str:
        cmd = [str(tools.ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi",
               "-i", "testsrc2=size=640x360:rate=10", "-t", str(SYNTHETIC_SECONDS), "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-y", str(synthetic)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", timeout=120)
        need(proc.returncode == 0, proc.stderr[-800:] or f"ffmpeg exited with {proc.returncode}")
        meta = tools.probe(synthetic)
        if meta is None:
            raise RuntimeError("ffprobe could not read the synthetic test video")
        holder["meta"] = meta
        return f"synthetic video OK ({meta.width}x{meta.height}, {meta.duration:.1f}s)"

    try:
        if not check("synthetic video", make_video):
            return

        def trickplay_cpu() -> str:
            cfg = copy.deepcopy(config)
            cfg["trickplay"].update(width=160, interval_seconds=5.0, tile_columns=3, tile_rows=3, ffmpeg_qscale=4, jpeg_quality=85)
            cfg["acceleration"]["mode"] = "cpu"
            logger = NullRunLogger("self-test")
            try:
                gen = TrickplayGenerator(tools, state, cfg, logger)
                frames_dir, sprites = work / "frames", work / "sprites"
                result = gen._run_ffmpeg_extract(holder["meta"], frames_dir, "CPU", None, progress_label="self-test")
                need(result.success, result.error[-600:] or "extraction failed")
                frames, tiles, thumb = gen._pack_frames(frames_dir, sprites)
                validation = validate_tile_directory(sprites, gen.settings, holder["meta"].duration,
                                                     expected_exact_tiles=tiles, verify_all=True)
                need(validation.valid, validation.reason)
                need(frames >= SYNTHETIC_SECONDS // 5, f"expected at least {SYNTHETIC_SECONDS // 5} frames, got {frames}")
                return f"CPU extract+pack+validate OK ({frames} frames, {tiles} sprite(s), thumbnail {thumb[0]}x{thumb[1]})"
            finally:
                logger.close()
        check("trickplay CPU pipeline", trickplay_cpu)
    finally:
        shutil.rmtree(work, ignore_errors=True)
