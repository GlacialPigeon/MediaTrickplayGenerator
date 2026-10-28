# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""Does a GPU path really decode on the GPU?

The pictures of a GPU path can be identical to the CPU's (decoding is exact) and its speed can match the CPU's, so neither
proves where the video was decoded. This check asks FFmpeg to hand the decoded frames over as *hardware* frames and then
download them: if the decoder had silently fallen back to software there are no hardware frames and FFmpeg fails.
"""
from __future__ import annotations

import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from . import hdr_validation
from .error_summary import classify_ffmpeg_error, diagnostic_error_summary
from .localization import t, tp
from .models import TrickplaySettings, VideoMeta

CODECS = ("h264", "hevc", "av1", "vp9")          # the codecs a GPU is usually expected to decode (the check tests every codec of the library, this only words the reason)
MAX_SAMPLES = 8
SEGMENT_SECONDS = 20.0
# hwdownload hands over frames in exactly the format they have in GPU memory and refuses any other ("Invalid output format
# yuv420p for hwframe download" when the format filter after it allows a choice), so the format is named for the video.
SURFACE_NAMES = {"d3d11va": "d3d11"}              # -hwaccel_output_format differs from the -hwaccel name for this one


@dataclass
class DecodeResult:
    ok: bool
    seconds: float = 0.0
    speed: float = 0.0                 # seconds of video decoded per second (includes starting FFmpeg and the device)
    error: str = ""                    # one short line for the screen
    detail: str = ""                   # the commands and everything FFmpeg said, for the diagnostic log
    decoder_starts: bool = False       # hardware decoding ran at all, even though the proof (hardware frames downloaded) failed
    unsupported: bool = False          # FFmpeg says the card or the driver cannot decode this kind of video at all


def sample_key(meta: VideoMeta) -> str:
    return f"{meta.codec} {meta.bit_depth}-bit"


def pick_samples(videos: list[VideoMeta], limit: Optional[int] = MAX_SAMPLES) -> list[VideoMeta]:
    """One video for every codec and bit depth there is in `videos` (what ffprobe found, not a fixed list): the one with the most
    pixels (the hardest case)."""
    best: dict[str, VideoMeta] = {}
    for meta in videos:
        key = sample_key(meta)
        current = best.get(key)
        if current is None or (meta.width * meta.height, str(meta.path).casefold()) > (current.width * current.height, str(current.path).casefold()):
            best[key] = meta
    picked = [best[key] for key in sorted(best)]
    return picked if limit is None else picked[:limit]


def with_surface_format(hwargs: list[str], backend: str) -> list[str]:
    """The arguments of the Generate path, with the decoded frames forced to stay in GPU memory."""
    if "-hwaccel_output_format" in hwargs: return list(hwargs)
    return [*hwargs, "-hwaccel_output_format", SURFACE_NAMES.get(backend, backend)]


def download_format(meta: VideoMeta) -> str:
    """The pixel format of hardware frames of this video once downloaded: NV12 for 8 bit, P010 for 10 bit, P016 above."""
    bits = meta.bit_depth
    if meta.chroma == "4:4:4": return "yuv444p" if bits <= 8 else "yuv444p16le"
    return "nv12" if bits <= 8 else "p010le" if bits <= 10 else "p016le"


def segment_of(meta: VideoMeta) -> tuple[float, float]:
    length = min(SEGMENT_SECONDS, meta.duration)
    start = max(0.0, min(meta.duration * 0.3, meta.duration - length - 1.0))
    return start, length


def decode_command(ffmpeg: Path, meta: VideoMeta, hwargs: Optional[list[str]], *, download: bool = True) -> list[str]:
    """Decode a short stretch and throw the result away. With `hwargs` the frames must arrive as hardware frames;
    `download` adds the step that fails when they did not (the proof)."""
    start, length = segment_of(meta)
    cmd = [str(ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats", "-ss", f"{start:.3f}"]
    if hwargs is not None: cmd += hwargs
    cmd += ["-i", str(meta.path), "-t", f"{length:.3f}", "-an", "-sn", "-dn"]
    if hwargs is not None and download: cmd += ["-vf", f"hwdownload,format={download_format(meta)}"]
    return cmd + ["-f", "null", "-"]


def _quote(arg: str) -> str:
    return f'"{arg}"' if any(ch in arg for ch in " |&") else arg


def _run(cmd: list[str], timeout: float) -> tuple[Optional[subprocess.CompletedProcess[str]], float, str]:
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, time.monotonic() - started, "Timed out."
    except OSError as exc:
        return None, 0.0, str(exc)
    return proc, max(time.monotonic() - started, 0.001), ""


def _one_line(stderr: str, returncode: int) -> str:
    """The cause in one line: a known explanation, or else the first line FFmpeg printed that is not just the aftermath."""
    return diagnostic_error_summary(stderr.strip() or f"FFmpeg exited with code {returncode}")


def check_decode(tools: Any, settings: TrickplaySettings, meta: VideoMeta, backend: Optional[str], timeout: float = 180.0) -> DecodeResult:
    """Software decoding of the stretch (`backend` None, for comparison) or hardware decoding on a GPU path."""
    hwargs: Optional[list[str]] = None
    if backend is not None:
        hwargs = with_surface_format(tools.backend_filter(backend, settings, meta)[0], backend)
    length = segment_of(meta)[1]
    cmd = decode_command(tools.ffmpeg, meta, hwargs)
    proc, elapsed, problem = _run(cmd, timeout)
    detail = "$ " + " ".join(_quote(a) for a in cmd)
    if proc is None:
        return DecodeResult(False, elapsed, 0.0, problem, detail + "\n" + problem)
    if proc.returncode == 0:
        return DecodeResult(True, elapsed, length / elapsed, "", detail)
    detail += f"\n(exit code {proc.returncode})\n{proc.stderr.strip()}"
    error = _one_line(proc.stderr, proc.returncode)
    if hwargs is None:
        return DecodeResult(False, elapsed, 0.0, error, detail)
    # The proof failed. Does the hardware decoder run at all (same command without the download step)? That tells a decoder that is
    # not used from a download step that FFmpeg could not set up, which is worth knowing when reading the report.
    plain = decode_command(tools.ffmpeg, meta, hwargs, download=False)
    second, _, _ = _run(plain, timeout)
    detail += "\n\nWithout the download step:\n$ " + " ".join(_quote(a) for a in plain)
    if second is not None and second.returncode == 0:
        return DecodeResult(False, elapsed, 0.0, "The decoder ran, but hardware frames could not be confirmed: " + error, detail, decoder_starts=True)
    if second is not None: detail += f"\n(exit code {second.returncode})\n{second.stderr.strip()}"
    unsupported = second is not None and classify_ffmpeg_error(proc.stderr) == "unsupported" and classify_ffmpeg_error(second.stderr) == "unsupported"
    return DecodeResult(False, elapsed, 0.0, error, detail, unsupported=unsupported)


def reason(row: dict[str, Any]) -> str:
    """Why a GPU path did not decode a kind of video on the GPU, in a few words (the full FFmpeg output is in the report)."""
    if row.get("decoder_starts"): return "decoded on the CPU"
    if row.get("unsupported") or str(row.get("kind", "")).split(" ")[0] not in CODECS: return "not supported by the GPU"
    return str(row.get("error") or "failed")


def summarize(rows: list[dict[str, Any]], backends: list[str]) -> list[str]:
    """Per GPU path one line (how many kinds of video it decoded on the GPU); under it the kinds it did not, grouped by what happened."""
    lines: list[str] = []
    width = max((len(b) for b in backends), default=0) + 3
    for backend in backends:
        mine = [r for r in rows if r["backend"] == backend]
        if not mine:
            lines.append(f"{backend:<{width}}" + t("decode.not_tested"))
            continue
        bad = [r for r in mine if not r["ok"]]
        total = len(mine)
        count = tp("decode.all_kinds", total) if not bad and total > 1 else tp("decode.some_kinds", total, decoded=total - len(bad))
        lines.append(f"{backend:<{width}}{count}")
        groups: dict[str, list[str]] = {}
        for row in bad: groups.setdefault(reason(row), []).append(row["kind"])
        for why, kinds in sorted(groups.items(), key=lambda item: (item[0] != "decoded on the CPU", item[0])):
            lead = {"decoded on the CPU": t("decode.lead_cpu"), "not supported by the GPU": t("decode.lead_unsupported")}.get(why, why)
            text = f"{lead}: " + ", ".join(kind.replace(" ", "\u00a0") for kind in kinds)               # a kind is never split over two lines
            lines.extend(line.replace("\u00a0", " ") for line in textwrap.wrap(text, width=96, initial_indent=" " * width, subsequent_indent=" " * (width + 4)))
    return lines


UNSUPPORTED = "UNSUPPORTED"


def verdicts(rows: list[dict[str, Any]]) -> hdr_validation.Results:
    """What Generate may rely on, per GPU path and kind of video: PASS (decoded on the GPU), FAIL (the decoder runs, but the
    frames do not stay in GPU memory: FFmpeg silently decodes on the CPU) or UNSUPPORTED (the card or driver cannot decode it
    at all). A path that failed for any other reason, or that was not tested, decides nothing: Generate keeps trying it."""
    results: hdr_validation.Results = {}
    for row in rows:
        backend, kind = row.get("backend"), row.get("kind")
        if not backend or backend == "cpu" or not kind: continue
        if row.get("ok"): results.setdefault(backend, {})[kind] = hdr_validation.PASS
        elif row.get("decoder_starts"): results.setdefault(backend, {})[kind] = hdr_validation.FAIL
        elif row.get("unsupported"): results.setdefault(backend, {})[kind] = UNSUPPORTED
    return results


def blocked_kinds(results: hdr_validation.Results) -> list[tuple[str, str]]:
    """The (path, kind) pairs Generate should not use: decoded on the CPU after all, or not supported at all."""
    return [(backend, kind) for backend, kinds in results.items() for kind, result in kinds.items() if result != hdr_validation.PASS]


def describe(results: hdr_validation.Results) -> list[str]:
    words = {hdr_validation.PASS: t("decode.on_gpu"), hdr_validation.FAIL: t("decode.on_cpu"), UNSUPPORTED: t("decode.unsupported")}
    return [f"{backend}: " + ", ".join(f"{kind} {words.get(result, result)}" for kind, result in kinds.items()) for backend, kinds in results.items()]


def stored_result(validation: Any, ffmpeg_hash: Optional[str], gpu_names: list[str], backend: str, meta: VideoMeta) -> Optional[str]:
    """What the checks found for this kind of video on this GPU path (PASS, FAIL, UNSUPPORTED) on this FFmpeg build and GPU; None if unknown."""
    if not isinstance(validation, dict) or not hdr_validation.same_system(validation, ffmpeg_hash, gpu_names): return None
    kinds: Optional[Any] = (validation.get("results") or {}).get(backend)
    result = kinds.get(sample_key(meta)) if isinstance(kinds, dict) else None
    return result if isinstance(result, str) else None


def is_blocked(validation: Any, ffmpeg_hash: Optional[str], gpu_names: list[str], backend: str, meta: VideoMeta) -> bool:
    """True when the decoding check found that this GPU path does not decode this kind of video on the GPU
    (on this FFmpeg build and GPU), so using it would only look like GPU work or fail."""
    return stored_result(validation, ffmpeg_hash, gpu_names, backend, meta) in (hdr_validation.FAIL, UNSUPPORTED)


def precheck(tools: Any, settings: TrickplaySettings, videos: list[VideoMeta], validation: Any, backends: Callable[[VideoMeta], list[str]],
             known_unsupported: Callable[[str, VideoMeta], bool], before: Optional[Callable[[VideoMeta, str], None]] = None,
             after: Optional[Callable[[VideoMeta, str, DecodeResult, Optional[str]], None]] = None) -> hdr_validation.Results:
    """Check, before Generate, every kind of video (codec and bit depth) on every GPU path that has no stored result yet for this
    FFmpeg build and GPU, so the table is complete and Generate knows what each path can do. Returns the new verdicts; nothing is saved here."""
    ffmpeg_hash, gpus = tools.candidate.sha256, list(tools.gpu_names)
    found: hdr_validation.Results = {}
    for sample in pick_samples(videos, None):
        kind = sample_key(sample)
        for backend in backends(sample):
            verdict = stored_result(validation, ffmpeg_hash, gpus, backend, sample)
            if verdict is None:
                if known_unsupported(backend, sample): continue            # Generate leaves this path out already
                if before: before(sample, backend)
                result = check_decode(tools, settings, sample, backend)
                row = {"kind": kind, "backend": backend, "ok": result.ok, "decoder_starts": result.decoder_starts, "unsupported": result.unsupported}
                verdict = verdicts([row]).get(backend, {}).get(kind)
                if verdict: found.setdefault(backend, {})[kind] = verdict
                if after: after(sample, backend, result, verdict)
    return found


CELLS = {"ok": "OK", "no": "no", "undecided": "?", "untested": "-"}


def compatibility_lines(kinds: list[str], backends: list[str], cell: Callable[[str, str], str]) -> list[str]:
    """The table of the check before Generate: a row per kind of video, a column per GPU path and the CPU, which decodes everything.
    `cell(kind, backend)` says "ok" (decodes on the GPU), "no" (does not; Generate skips it), "undecided" or "untested"
    (a path before it decoded the kind, so this one was not needed). Under the table, a line explains each sign that is in it."""
    if not kinds: return []
    cpu = "CPU"
    columns = [*backends, cpu]
    codes = {kind: {**{backend: cell(kind, backend) for backend in backends}, cpu: "ok"} for kind in kinds}
    first = max([len(t("diag.col_kind"))] + [len(kind) for kind in kinds]) + 3
    widths = [max(len(column), 2) + 3 for column in columns]
    lines = ["", "  " + t("diag.col_kind").ljust(first) + "".join(column.ljust(width) for column, width in zip(columns, widths)).rstrip()]
    for kind in kinds:
        lines.append("  " + kind.ljust(first) + "".join(CELLS[codes[kind][column]].ljust(width) for column, width in zip(columns, widths)).rstrip())
    used = {code for row in codes.values() for code in row.values()}
    meaning = {"ok": t("check.legend_ok"), "no": t("check.legend_no"), "undecided": t("check.legend_undecided"), "untested": t("check.legend_untested")}
    legend = [text for code, text in meaning.items() if code == "ok" or code in used]
    return lines + [""] + ["  " + line for line in legend]
