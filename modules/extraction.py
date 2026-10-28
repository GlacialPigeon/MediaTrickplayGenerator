# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import hashlib
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import hardware_decode, hdr_validation, path_choice, platform_api
from .console_progress import CONSOLE_PROGRESS
from .error_summary import classify_ffmpeg_error, compact_ffmpeg_error
from .ffmpeg_tools import FFmpegTools
from .logger import RunLogger
from .models import ExtractResult, TrickplaySettings, VideoMeta
from .state_db import StateDB
from .tiles import PIL_AVAILABLE, Image, frame_count_deviation, hdr_kind

# Some files carry a colour space tag (bt2020c, for one) that newer FFmpeg builds refuse in the scale filter ("Impossible to convert
# between the formats supported by the filter ... and the filter 'auto_scale_0'"). The tag does not matter for small JPEG
# thumbnails, so the second try ignores it. The first try is unchanged, so every other file is processed exactly as before.
COLOR_TAG_ERROR = "impossible to convert between the formats"
IGNORE_COLOR_TAG = "setparams=colorspace=unknown"


class FrameExtraction:
    """Making the thumbnails of one video: the FFmpeg run (GPU or CPU, with its retries), the choice of method and packing the frames into tiles.
    Mixed into TrickplayGenerator, which sets the attributes below in its constructor."""

    tools: FFmpegTools
    state: StateDB
    config: dict[str, Any]
    logger: RunLogger
    settings: TrickplaySettings
    gpu_sem: threading.Semaphore
    keyframe_only: bool
    keyframe_repeat_limit: float
    stop_event: threading.Event

    # FFmpeg that makes no progress for this long is killed (a drive that stopped answering would otherwise block its storage group forever).
    # Progress is the position in the video; slow decoding (a big file on the CPU) still advances every few seconds.
    STALL_SECONDS: float = 600.0

    def _check_canceled(self) -> None:
        if self.stop_event.is_set():
            raise InterruptedError("Canceled by user")

    def _preferred_backends(self) -> list[str]:
        available = self.tools.backends()
        preferred = self.config["acceleration"].get("preferred_backend", "auto")
        if preferred and preferred != "auto" and preferred in available: return [preferred] + [b for b in available if b != preferred]
        return available

    def _backends_for(self, meta: VideoMeta, mode: str) -> list[str]:
        """The GPU paths to try for this video, in order. In "auto" mode the benchmark's choice for this kind of video comes first: the CPU
        (no GPU path at all) where no GPU path was clearly faster, else the fastest GPU path first. Kinds it did not measure: as before."""
        backends = self._preferred_backends()
        if mode != "auto": return backends
        chosen = path_choice.choice_for(self.config.get("acceleration", {}).get("path_choice"), self.tools.candidate.sha256, list(self.tools.gpu_names), meta)
        if chosen is None: return backends
        if chosen == path_choice.CPU: return []
        return [chosen] + [b for b in backends if b != chosen] if chosen in backends else backends

    def _hdr_gpu_blocked(self, meta: VideoMeta, backend: Optional[str]) -> bool:
        """True when the HDR check found this GPU path wrong for this kind of HDR (on this FFmpeg build and GPU)."""
        if not meta.hdr or not backend: return False
        return hdr_validation.is_blocked(self.config.get("acceleration", {}).get("hdr_validation"),
                                         self.tools.candidate.sha256, list(self.tools.gpu_names), backend, hdr_kind(meta))

    def _gpu_decode_blocked(self, meta: VideoMeta, backend: str) -> bool:
        """True when the GPU decoding check found that this GPU path decodes this codec and bit depth on the CPU after all."""
        return hardware_decode.is_blocked(self.config.get("acceleration", {}).get("gpu_decoding"),
                                          self.tools.candidate.sha256, list(self.tools.gpu_names), backend, meta)

    def _frame_pattern(self, temp_frames: Path) -> str: return str(temp_frames / "frame_%08d.jpg")

    def _classify_ffmpeg_error(self, text: str) -> str: return classify_ffmpeg_error(text)

    def _run_ffmpeg_extract(self, meta: VideoMeta, temp_frames: Path, method: str, backend: Optional[str], *, start_time: Optional[float] = None, duration_limit: Optional[float] = None, progress_label: str = "", ignore_color_tag: bool = False, normalize_fps: Optional[float] = None, keyframes: bool = False) -> ExtractResult:
        self._check_canceled()
        temp_frames.mkdir(parents=True, exist_ok=True)
        for old in temp_frames.glob("frame_*.jpg"): old.unlink(missing_ok=True)
        cmd: list[str] = [str(self.tools.ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats", "-progress", "pipe:1"]
        if start_time is not None: cmd += ["-ss", f"{start_time:.3f}"]

        if method == "GPU" and backend:
            hwargs, vf = self.tools.backend_filter(backend, self.settings, meta)
            cmd += hwargs
        else:
            vf = self.tools.cpu_filter(self.settings)
        if normalize_fps: vf = f"setpts=N/{normalize_fps:.3f}/TB,{vf}"     # time stamps from the frame count, as Jellyfin does for files whose own are not usable
        if ignore_color_tag: vf = f"{IGNORE_COLOR_TAG},{vf}"
        if keyframes: cmd += ["-skip_frame", "nokey"]          # the decoder drops every picture that is not a key frame; `fps` still fills every slot
        cmd += ["-i", str(meta.path)]

        if duration_limit is not None: cmd += ["-t", f"{duration_limit:.3f}"]
        cmd += ["-an", "-sn", "-dn", "-vf", vf, "-q:v", str(self.settings.qscale), "-fps_mode", "passthrough", "-y", self._frame_pattern(temp_frames)]

        start = time.monotonic()
        stderr_lines: list[str] = []
        # Keep console Ctrl+C in the parent; the second interrupt explicitly terminates FFmpeg.
        process_options = platform_api.subprocess_process_options()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1, **process_options)

        def _stderr_reader() -> None:
            if proc.stderr is None: return
            for line in proc.stderr:
                stderr_lines.append(line.rstrip())
                if len(stderr_lines) > 500: del stderr_lines[:100]

        th = threading.Thread(target=_stderr_reader, daemon=True)
        th.start()
        finished = threading.Event()
        stalled = threading.Event()
        last_progress = [time.monotonic()]          # when the position in the video last moved

        def cancel_process() -> None:
            while not finished.wait(0.1):
                if self.stop_event.is_set() or time.monotonic() - last_progress[0] > self.STALL_SECONDS:
                    if not self.stop_event.is_set(): stalled.set()
                    try:
                        proc.terminate()
                        try: proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                    except OSError: pass
                    return

        monitor = threading.Thread(target=cancel_process, daemon=True)
        monitor.start()

        last_print = 0.0
        out_seconds = 0.0
        try:
            if proc.stdout:
                for raw in proc.stdout:
                    line = raw.strip()
                    before_seconds = out_seconds
                    if line.startswith("out_time_ms="):
                        try: out_seconds = float(line.split("=", 1)[1]) / 1_000_000.0
                        except Exception: pass
                    elif line.startswith("out_time_us="):
                        try: out_seconds = float(line.split("=", 1)[1]) / 1_000_000.0
                        except Exception: pass

                    now = time.monotonic()
                    if out_seconds > before_seconds: last_progress[0] = now
                    if now - last_print >= 2.0 and meta.duration > 0:
                        last_print = now
                        effective_duration = duration_limit or meta.duration
                        pct = min(100.0, 100.0 * out_seconds / max(0.001, effective_duration))
                        elapsed = now - start
                        speed = out_seconds / elapsed if elapsed > 0 else 0
                        eta = ((effective_duration - out_seconds) / speed) if speed > 0 else None
                        label = progress_label or meta.path.name
                        progress_key = progress_label or str(meta.path)
                        CONSOLE_PROGRESS.update_job(progress_key, label, method, backend, pct, speed, elapsed, eta)
            rc = proc.wait()
            th.join(timeout=1)
        except KeyboardInterrupt:
            self.stop_event.set()
            monitor.join()
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            finished.set()
            monitor.join()
            th.join()
            if proc.stdout: proc.stdout.close()
            if proc.stderr: proc.stderr.close()

        if self.stop_event.is_set():
            return ExtractResult(False, method, backend, time.monotonic() - start, 0, "canceled by user", canceled=True)

        elapsed = time.monotonic() - start
        CONSOLE_PROGRESS.finish(progress_label or str(meta.path))
        error = compact_ffmpeg_error("\n".join(stderr_lines[-200:]).strip())
        frames = len(list(temp_frames.glob("frame_*.jpg")))
        if stalled.is_set():
            return ExtractResult(False, method, backend, elapsed, frames, f"FFmpeg made no progress for {self.STALL_SECONDS:g} s and was stopped (is the drive or share still answering?)", stalled=True)

        if rc != 0:
            kind = self._classify_ffmpeg_error(error)
            return ExtractResult(False, method, backend, elapsed, frames, error or f"FFmpeg exited with code {rc}", unsupported=(kind == "unsupported"))
        if frames <= 0: return ExtractResult(False, method, backend, elapsed, 0, "FFmpeg produced no frames")
        return ExtractResult(True, method, backend, elapsed, frames)

    def _pack_frames(self, frames_dir: Path, output_dir: Path) -> tuple[int, int, tuple[int, int]]:
        if not PIL_AVAILABLE: raise RuntimeError("Pillow is required.")
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        if not frame_paths: raise RuntimeError("no extracted frames to pack")
        with Image.open(frame_paths[0]) as im: thumb_w, thumb_h = im.size
        if thumb_w != self.settings.width: raise RuntimeError(f"extracted thumbnail width {thumb_w} != target {self.settings.width}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for old in output_dir.glob("*.jpg"): old.unlink(missing_ok=True)
        capacity = self.settings.capacity
        tiles = int(math.ceil(len(frame_paths) / capacity))
        sprite_size = (thumb_w * self.settings.cols, thumb_h * self.settings.rows)
        for tile_index in range(tiles):
            self._check_canceled()
            canvas = Image.new("RGB", sprite_size, "black")
            batch = frame_paths[tile_index * capacity : (tile_index + 1) * capacity]
            for pos, frame in enumerate(batch):
                row, col = divmod(pos, self.settings.cols)
                with Image.open(frame) as im:
                    rgb = im.convert("RGB")
                    if rgb.size != (thumb_w, thumb_h): raise RuntimeError(f"inconsistent frame size: {frame.name}")
                    canvas.paste(rgb, (col * thumb_w, row * thumb_h))
            canvas.save(output_dir / f"{tile_index}.jpg", "JPEG", quality=self.settings.jpeg_quality)
        return len(frame_paths), tiles, (thumb_w, thumb_h)

    def _attempt_extract_with_retry(self, meta: VideoMeta, temp_frames: Path, method: str, backend: Optional[str], label: str) -> ExtractResult:
        keyframes, ignore_color_tag = self.keyframe_only, False

        def attempt() -> ExtractResult: return self._run_ffmpeg_extract(meta, temp_frames, method, backend, progress_label=label, ignore_color_tag=ignore_color_tag, keyframes=keyframes)
        first = attempt()
        while not (first.success or first.canceled or first.stalled):
            if not ignore_color_tag and COLOR_TAG_ERROR in first.error.lower():
                # Tried first, so a file with such a tag keeps the key-frame speed-up: the tag is the problem, not the key frames.
                self.logger.log("INFO", f"Retrying {method}{('/'+backend) if backend else ''} with the colour space tag ignored: {meta.path}")
                ignore_color_tag = True
            elif keyframes:
                # Not every decoder and file can skip pictures. Nothing about the failure is remembered: it says nothing about the usual way.
                self.logger.log("WARN", f"Key-frame-only extraction failed, extracting every frame instead: {meta.path}: {first.error[-300:].strip()}")
                keyframes = False
            else: break
            first = attempt()
        first.color_tag_ignored, first.keyframe_only = ignore_color_tag, keyframes
        if first.success or first.canceled or first.stalled: return first
        if method == "GPU" and backend:
            kind = self._classify_ffmpeg_error(first.error)
            self.state.record_backend_failure(backend, meta, kind, first.error)
            if kind == "unsupported":
                self.logger.log("WARN", f"GPU {backend} unsupported for {meta.path}: {first.error[-800:]}")
                return first
            self.logger.log("WARN", f"Retrying {method}{('/'+backend) if backend else ''} for {meta.path}")
            self.stop_event.wait(1)
            self._check_canceled()
            second = self._run_ffmpeg_extract(meta, temp_frames, method, backend, progress_label=label, ignore_color_tag=ignore_color_tag, keyframes=keyframes)
            second.color_tag_ignored, second.keyframe_only = ignore_color_tag, keyframes
            return second
        return first

    def _choose_method_and_extract(self, meta: VideoMeta, temp_frames: Path, label: str) -> ExtractResult:
        mode = self.config["acceleration"].get("mode", "auto")
        attempted_gpu = False
        attempts: list[str] = []
        preferred_backends = self._backends_for(meta, mode)
        if mode == "auto" and not preferred_backends and self._preferred_backends():
            self.logger.log("INFO", f"CPU chosen by the benchmark for this kind of video: {path_choice.kind_of(meta)}")
        if mode != "cpu":
            for backend in preferred_backends:
                if self._hdr_gpu_blocked(meta, backend):
                    self.logger.log("INFO", f"HDR GPU path disabled by the HDR check: backend={backend} kind={hdr_kind(meta)}")
                    continue
                if mode != "gpu" and self._gpu_decode_blocked(meta, backend):
                    self.logger.log("INFO", f"GPU path skipped by the GPU decoding check: backend={backend} kind={hardware_decode.sample_key(meta)}")
                    continue
                if self.state.known_unsupported(backend, meta):
                    self.logger.log("INFO", f"Skipping known unsupported GPU signature: {backend}")
                    continue
                attempted_gpu = True
                attempts.append(f"GPU/{backend}")
                with self.gpu_sem:
                    gpu = self._attempt_extract_with_retry(meta, temp_frames, "GPU", backend, label)
                    gpu.attempts = list(attempts)
                    if gpu.success or gpu.canceled or gpu.stalled: return gpu
                    if mode == "gpu": continue
                    self.logger.event("GPU_FAILED_CPU_FALLBACK", path=str(meta.path), backend=backend, error=gpu.error[-2000:])
                    break
            if mode == "gpu": return ExtractResult(False, "GPU", None, 0, 0, "all GPU backends failed", attempts=attempts)
        attempts.append("CPU")
        cpu = self._attempt_extract_with_retry(meta, temp_frames, "CPU", None, label)
        cpu.fallback = attempted_gpu
        cpu.attempts = list(attempts)
        return cpu

    @staticmethod
    def repeated_frames(frames_dir: Path) -> tuple[int, int]:
        """(thumbnails that are exactly the picture before them, thumbnails). Where the key frames are further apart than the interval
        the `fps` filter repeats the last one, and a repeated picture is encoded to exactly the same bytes."""
        previous: Optional[bytes] = None
        repeated = total = 0
        for path in sorted(frames_dir.glob("frame_*.jpg")):
            digest = hashlib.sha1(path.read_bytes()).digest()
            total += 1
            if digest == previous: repeated += 1
            previous = digest
        return repeated, total

    def _redo_with_every_frame_if_sparse(self, meta: VideoMeta, extract: ExtractResult, frames_dir: Path, label: str) -> tuple[ExtractResult, Path]:
        """After a key-frame-only extraction: when more than the allowed share of the thumbnails repeat the one before (the key frames of
        this file are further apart than the interval, so the thumbnails would show pictures up to a whole key frame distance too early),
        the video is extracted again with every frame. The usual way is always right, so its result replaces the key-frame one."""
        if not extract.keyframe_only or self.keyframe_repeat_limit <= 0: return extract, frames_dir
        repeated, total = self.repeated_frames(frames_dir)
        percent = 100.0 * repeated / total if total else 0.0
        if percent <= self.keyframe_repeat_limit: return extract, frames_dir
        self.logger.log("INFO", f"{meta.path}: {repeated:,} of {total:,} thumbnails ({percent:.0f} %) repeat the one before (key frames too far apart); extracting every frame instead")
        retry_dir = frames_dir.parent / "frames_all"
        retry = self._extract_again(meta, retry_dir, extract, label, keyframes=False, normalize_fps=None)
        if not retry.success:
            self.logger.log("WARN", f"Extracting every frame failed, the key-frame result is kept: {meta.path}")
            return extract, frames_dir
        self.logger.event("KEYFRAMES_TOO_SPARSE", path=str(meta.path), repeated=repeated, frames=total, percent=round(percent, 1), limit_percent=self.keyframe_repeat_limit)
        retry.elapsed += extract.elapsed
        retry.attempts, retry.fallback, retry.color_tag_ignored, retry.keyframe_only = extract.attempts, extract.fallback, extract.color_tag_ignored, False
        return retry, retry_dir

    def _extract_again(self, meta: VideoMeta, frames_dir: Path, base: ExtractResult, label: str, *, keyframes: bool, normalize_fps: Optional[float]) -> ExtractResult:
        """One more extraction in the same way (GPU backend or CPU, colour tag) as `base`, with the given changes."""
        if base.method == "GPU":
            with self.gpu_sem:
                return self._run_ffmpeg_extract(meta, frames_dir, "GPU", base.backend, progress_label=label, ignore_color_tag=base.color_tag_ignored, normalize_fps=normalize_fps, keyframes=keyframes)
        return self._run_ffmpeg_extract(meta, frames_dir, base.method, base.backend, progress_label=label, ignore_color_tag=base.color_tag_ignored, normalize_fps=normalize_fps, keyframes=keyframes)

    def _normalize_timestamps_if_suspect(self, meta: VideoMeta, extract: ExtractResult, frames_dir: Path, label: str) -> tuple[ExtractResult, Path]:
        """More extractions for a video that gets too many thumbnails for its length. After a key-frame-only extraction the first one
        decodes every frame. Then comes one with the time stamps made from the frame count and the file's frame rate: the time stamps of
        the file may be damaged (the FFmpeg `fps` filter then repeats pictures), and Jellyfin makes them this way for every file.
        The result that comes closest to the expected number is kept, so a video that this does not help ends up exactly as before,
        and a normal video is never extracted twice. Too few thumbnails are left alone: the video stream is shorter than the length
        the container states, nothing about the extraction changes that, and every try costs a full decode (a minute or more)."""
        def off_by(result: ExtractResult) -> Optional[tuple[int, int]]: return frame_count_deviation(result.frames, meta.duration, self.settings.interval)
        deviation = off_by(extract)
        if deviation is None or extract.frames < deviation[0]: return extract, frames_dir
        expected = deviation[0]
        steps: list[tuple[str, bool, Optional[float]]] = []          # what is different, key frames only, frame rate for the time stamps
        if extract.keyframe_only: steps.append(("every frame decoded", False, None))
        fps = self.tools.reference_frame_rate(meta.path)
        if fps is not None: steps.append((f"time stamps made from the frame count ({fps:.3f} frames/s)", False, fps))
        best, best_dir = extract, frames_dir
        for number, (what, keyframes, rate) in enumerate(steps, 1):
            if off_by(best) is None: break
            retry_dir = frames_dir.parent / f"frames_retry{number}"
            self.logger.log("INFO", f"{meta.path}: {best.frames:,} thumbnails instead of about {expected:,}; trying once more with {what}")
            retry = self._extract_again(meta, retry_dir, extract, label, keyframes=keyframes, normalize_fps=rate)
            if not retry.success or abs(retry.frames - expected) >= abs(best.frames - expected):
                self.logger.log("INFO", f"That did not come closer ({retry.frames:,} thumbnails); the earlier result is kept: {meta.path}")
                continue
            self.logger.event("TIMESTAMPS_NORMALIZED" if rate else "FULL_DECODE_INSTEAD_OF_KEYFRAMES", path=str(meta.path), frames_before=best.frames,
                              frames_after=retry.frames, expected_frames=expected, **({"frame_rate": rate} if rate else {}))
            self.logger.log("INFO", f"{retry.frames:,} thumbnails instead of {best.frames:,} with {what}: {meta.path}")
            retry.elapsed += best.elapsed
            retry.attempts, retry.fallback, retry.color_tag_ignored, retry.keyframe_only = extract.attempts, extract.fallback, extract.color_tag_ignored, keyframes
            best, best_dir = retry, retry_dir
        return best, best_dir
