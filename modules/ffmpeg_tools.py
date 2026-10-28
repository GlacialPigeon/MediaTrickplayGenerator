# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from . import media_classification, platform_api
from .ffmpeg_registry import FFmpegCandidate
from .logger import RunLogger
from .models import TrickplaySettings, VideoMeta


def parse_frame_rate(text: object) -> Optional[float]:
    """FFprobe writes a frame rate as a fraction ("13978/583", "30000/1001", "25/1"). None for "0/0", nonsense or a rate of 0 or less."""
    try:
        num, _, den = str(text).partition("/")
        value = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


class FFmpegTools:
    def __init__(self, candidate: FFmpegCandidate, logger: Optional[RunLogger] = None) -> None:
        self.candidate = candidate
        self.ffmpeg = candidate.ffmpeg
        self.ffprobe = candidate.ffprobe
        self.logger = logger
        self.probe_failures: dict[str, str] = {}            # why the last probe of a path failed (a file without a video stream is not a failure)
        self.gpu_names = self.detect_gpu_names()
        self.hw_profile = self._detect_hardware()

    def detect_gpu_names(self) -> list[str]:
        return platform_api.detect_gpu_names()

    def _detect_hardware(self) -> dict[str, bool]:
        return platform_api.detect_hardware()

    def backends(self) -> list[str]:
        caps = self.candidate.capabilities
        out: list[str] = []
        if caps.get("cuda") and caps.get("scale_cuda") and self.hw_profile["nvidia_gpu"]: out.append("cuda")
        if caps.get("qsv") and caps.get("scale_qsv") and self.hw_profile["intel_gpu"] and self.qsv_available(): out.append("qsv")
        out.extend(platform_api.extra_backends(caps, self.hw_profile))
        return out

    def qsv_available(self) -> bool:
        if not hasattr(self, "_qsv_available"):
            try:
                proc = subprocess.run(
                    [str(self.ffmpeg), "-hide_banner", "-nostdin", "-v", "error",
                     "-init_hw_device", "qsv:hw", "-f", "lavfi", "-i", "color=size=16x16",
                     "-frames:v", "1", "-f", "null", "-"],
                    capture_output=True, timeout=15,
                )
                self._qsv_available = proc.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                self._qsv_available = False
        return self._qsv_available

    def probe(self, path: Path, timeout: int = 60) -> Optional[VideoMeta]:
        self.probe_failures.pop(str(path), None)
        try: st = path.stat()
        except OSError as exc:
            self.probe_failures[str(path)] = f"cannot be read: {exc}"
            return None
        cmd = [str(self.ffprobe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=timeout)
            if proc.returncode != 0:
                lines = [line.strip() for line in proc.stderr.splitlines() if line.strip()]
                self.probe_failures[str(path)] = "ffprobe could not read it" + (f": {lines[-1][:200]}" if lines else "")
                return None
            data = json.loads(proc.stdout)
            streams = data.get("streams", [])
            video = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
            if not video: return None
            duration = 0.0
            for source in (video, data.get("format", {})):
                try:
                    val = float(source.get("duration", 0) or 0)
                    if val > duration: duration = val
                except Exception: pass
            pix_fmt = str(video.get("pix_fmt", "") or "")
            profile = str(video.get("profile", "") or "")
            picture = media_classification.describe_stream(video)
            return VideoMeta(path=path, size=st.st_size, mtime_ns=st.st_mtime_ns, duration=duration,
                             codec=str(video.get("codec_name", "unknown") or "unknown").lower(), profile=profile, pix_fmt=pix_fmt,
                             width=int(video.get("width", 0) or 0), height=int(video.get("height", 0) or 0), hdr=picture["hdr"],
                             bits=picture["bit_depth"], chroma=picture["chroma"], resolution=picture["resolution_class"],
                             hdr_pool=picture["hdr_pool"], hdr_type=picture["hdr_type"], dv_profile=picture["dv_profile"],
                             hdr_note=picture["hdr_note"])
        except subprocess.TimeoutExpired:
            self.probe_failures[str(path)] = f"ffprobe did not answer within {timeout} s"
            return None
        except Exception as exc:
            self.probe_failures[str(path)] = f"{type(exc).__name__}: {exc}"
            return None

    def reference_frame_rate(self, path: Path, timeout: int = 60) -> Optional[float]:
        """The frame rate FFprobe calls r_frame_rate (the one Jellyfin uses too), None when the file does not state a usable one."""
        cmd = [str(self.ffprobe), "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate:stream_disposition=attached_pic",
               "-of", "json", str(path)]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=timeout)
            if proc.returncode != 0: return None
            streams = json.loads(proc.stdout).get("streams", [])
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None
        video = next((s for s in streams if not s.get("disposition", {}).get("attached_pic")), None)
        return parse_frame_rate(video.get("r_frame_rate")) if video else None

    def backend_filter(self, backend: str, settings: TrickplaySettings, meta: VideoMeta) -> tuple[list[str], str]:
        fps = f"fps=1/{settings.interval:g}"
        if backend == "cuda":
            args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-extra_hw_frames", "24"]
            vf = f"{fps},scale_cuda=w={settings.width}:h=-2:format=nv12:interp_algo=lanczos,hwdownload,format=nv12"
            return args, vf
        if backend == "qsv":
            args = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
            vf = f"{fps},scale_qsv=w={settings.width}:h=-2:format=nv12,hwdownload,format=nv12"
            return args, vf
        platform_filter = platform_api.backend_filter(backend, fps, settings.width)
        if platform_filter is not None:
            return platform_filter
        raise ValueError(f"unknown backend: {backend}")

    def cpu_filter(self, settings: TrickplaySettings) -> str:
        return f"fps=1/{settings.interval:g},scale={settings.width}:-2:flags=lanczos"
