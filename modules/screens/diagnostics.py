# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

from collections import Counter
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .. import benchmark_advice, hardware_decode, hdr_validation, library_sample, path_choice, platform_api
from ..activity_log import ACTIVITY
from ..console_progress import CONSOLE_PROGRESS
from ..constants import APP_NAME, APP_VERSION, DIRS, KNOWN_NONVIDEO_EXTENSIONS
from ..diagnostics import (
    DiagnosticJob, DiagnosticRunner, diagnostic_segments, select_hdr_candidates,
)
from ..formatting import display_path, human_bytes, human_duration
from ..paths import remove_outer_quotes
from ..generator import TrickplayGenerator
from ..logger import NullRunLogger
from ..models import VideoMeta
from ..localization import english, lazy, t, tp
from ..localization import number as grouped
from ..prompts import choose, fit, heading, label_width, labelled, pause, prompt_choice, prompt_yes_no
from ..scanner import MediaScanner
from ..selftest import run_self_test
from ..tiles import classify_video, config_settings, hdr_label
from .configuration import ConfigurationScreens


class DiagnosticScreens(ConfigurationScreens):
    """Benchmark menu: CPU vs GPU benchmark, HDR check, GPU decoding check (and the self-test, which Maintenance offers)."""

    def run_diagnostic(self, kind: str, *, before_generate: bool = False, library: Optional[list[VideoMeta]] = None) -> bool:
        """One of the Benchmark tests. `before_generate`: the benchmark Generate offered on its first run - the library is the source,
        nothing is asked but the settings suggested at the end, and the screen is not cleared. `library`: the videos Generate has scanned already
        (no second scan). True when the benchmark ran to the end."""
        if not before_generate: self._screen(self._test_title(kind))
        completed = False
        if not before_generate and not self.ensure_ffmpeg(): return False             # Generate has checked it already
        if not self._require_pillow():
            if not before_generate: pause()
            return False
        source = ("library", None) if before_generate else self._choose_test_source(kind)
        if source is None: return False
        folder = source[1]
        from_library = folder is None
        mode = "sample" if from_library and kind == "benchmark" else ""
        if not mode and kind != "hdr":
            mode = self._choose_test_mode()
            if not mode: return False
        job = DiagnosticJob(kind)
        logger = None
        try:
            sample_info: list[dict[str, str]] = []
            if folder is None: videos = self._library_videos(kind, job, library)
            else:
                job.say("diag.test_folder", folder=folder)
                videos = self._collect_test_videos(folder, job)
            if kind == "hdr":
                hdr_total = sum(1 for meta in videos if meta.hdr)
                videos = select_hdr_candidates(videos)
                for meta in videos: job.log(f"HDR test file: {hdr_label(meta)} | {meta.path}", echo=False)
            elif kind == "benchmark" and from_library:
                picked = library_sample.pick_sample(videos, rng=random.Random())
                videos = [p.meta for p in picked]
                sample_info = [{"video": str(p.meta.path), "kind": library_sample.kind_key(p.meta), "why": p.why} for p in picked]
            if not videos:
                message = job.say("diag.no_hdr_candidate", prefix="\n") if kind == "hdr" else job.say("diag.no_usable_videos", prefix="\n")
                if kind == "hdr": job.say("diag.no_hdr_in_source")
                job.save(message, {"kind": kind, "status": "NOT_RUN", "reason": message})
                if not before_generate: pause()
                return False
            if not mode:                                    # the HDR check asks this only once it has found something to check
                print("\n" + tp("diag.hdr_found", hdr_total, chosen=grouped(len(videos))))
                for meta in videos: print(f"  {hdr_label(meta):<16} {meta.path.name}")
                mode = self._choose_test_mode()
                if not mode: return False
            backends = self._require_tools().backends()
            if before_generate: print("\n" + fit(tp("diag.files_tested", len(videos), paths=", ".join(["CPU", *[f"GPU/{b}" for b in backends]]), report=display_path(job.path)), 0))
            elif from_library and kind == "benchmark": self._show_library_preflight(videos, sample_info, backends, job)
            else: self._show_diagnostic_preflight(videos, backends, mode, job)
            if not before_generate and not prompt_yes_no("\n" + t("diag.start_test"), False):
                job.save("Canceled before execution", {"kind": kind, "status": "CANCELED"})
                return False
            logger = NullRunLogger(kind)
            gen = TrickplayGenerator(self._require_tools(), self.state, self.config, logger)
            runner = DiagnosticRunner(gen, job)
            data = runner.run(videos, mode, backends, kind, local_copy=from_library and kind == "benchmark", sample=sample_info)
            if kind == "hdr" and data["status"] == "COMPLETE":
                self.review_hdr_images(data, job.path, lambda: runner.save(data))
                self._offer_hdr_policy(data)
            self._show_diagnostic_results(kind, data, job)
            completed = data["status"] == "COMPLETE"
            asked = False                                    # a question that was the last thing on the screen needs no extra Enter
            if kind == "benchmark" and completed:
                benchmark_advice.remember_disk_speeds(self.config, data)
                asked = self._offer_benchmark_settings(data)
                tools = self._require_tools()
                path_choice.mark_benchmark_seen(self.config["acceleration"], tools.candidate.sha256, list(tools.gpu_names))
                self.config_mgr.save()
            if not before_generate and not asked: pause()
        except KeyboardInterrupt:
            job.say("diag.canceled_kept", prefix="\n")
            if not before_generate: pause()
        except Exception as exc:
            job.say("diag.error", prefix="\n", kind=type(exc).__name__, error=exc)
            if not before_generate: pause()
        finally:
            if logger is not None: logger.close()
            job.close()
        return completed

    @staticmethod
    def _choose_test_mode() -> str:
        """Quick (a few minutes of each video) or the full files; "" for Back."""
        mode = choose([("1", t("diag.mode_quick")), ("2", t("diag.mode_full")), ("0", t("common.back"))], "1")
        return {"1": "quick", "2": "full"}.get(mode, "")

    def run_decode_check(self) -> None:
        """Does each GPU path really decode on the GPU? One video per codec, compared with software decoding."""
        self._screen(self._test_title("decode"))
        if not self.ensure_ffmpeg(): return
        tools = self._require_tools()
        backends = tools.backends()
        if not backends:
            print("\n" + t("diag.decode_no_path"))
            pause()
            return
        source = self._choose_test_source("decode")
        if source is None: return
        job = DiagnosticJob("decode")
        try:
            if source[1] is None: found = self._library_videos("decode", job)
            else:
                job.say("diag.test_folder", folder=source[1])
                found = self._collect_test_videos(source[1], job)
            samples = hardware_decode.pick_samples(found, None)
            if not samples:
                message = job.say("diag.no_usable_videos")
                job.save(message, {"kind": "decode", "status": "NOT_RUN", "reason": message})
                pause()
                return
            heading(t("heading.preflight"))
            labels = [t("pre.files"), t("pre.paths"), t("pre.test"), t("pre.report")]
            width = label_width(labels)
            labelled([(labels[0], t("diag.decode_files", count=grouped(len(samples)))), (labels[1], ", ".join(["CPU", *[f"GPU/{b}" for b in backends]])),
                      (labels[2], t("diag.decode_test", seconds=f"{hardware_decode.SEGMENT_SECONDS:g}"))], width=width)
            key_width = max(len(hardware_decode.sample_key(meta)) for meta in samples)
            res_width = max(len(meta.resolution_class) for meta in samples)
            kinds = [f"{hardware_decode.sample_key(meta):<{key_width}} | {meta.resolution_class:<{res_width}}" for meta in samples]
            tag_width, kind_width = len(f"[{len(samples)}/{len(samples)}]"), max(len(k) for k in kinds)
            room = CONSOLE_PROGRESS._terminal_width() - 1 - (2 + tag_width + 1 + kind_width + 3)            # what is left of the line for the file name
            for number, (meta, kind) in enumerate(zip(samples, kinds), 1):
                name = meta.path.name if room < 20 or len(meta.path.name) <= room else meta.path.name[:room - 3] + "..."
                print(f"  {f'[{number}/{len(samples)}]':<{tag_width}} {kind:<{kind_width}} | {name}")
            labelled([(labels[3], display_path(job.path))], width=width)
            if not prompt_yes_no("\n" + t("diag.start_test"), False):
                job.save("Canceled before execution", {"kind": "decode", "status": "CANCELED"})
                return
            settings = config_settings(self.config)
            rows: list[dict[str, Any]] = []
            status = "COMPLETE"
            live = sys.stdout.isatty() and CONSOLE_PROGRESS.enabled
            started = time.monotonic()
            heading(t("diag.decode_heading"))
            cell_fail, cell_not_gpu = t("diag.cell_fail"), t("diag.cell_not_gpu")
            cell = max(9, len(cell_fail) + 1, len(cell_not_gpu) + 1)                       # a translated cell is never glued to the next one
            print(t("diag.decode_legend", not_gpu=cell_not_gpu))
            print(f"  {'#':<8}{t('diag.col_kind'):<30}  " + "".join(f"{name:<{cell}}" for name in ["CPU", *backends]).rstrip())
            try:
                for number, meta in enumerate(samples, 1):
                    kind = hardware_decode.sample_key(meta)
                    text = kinds[number - 1]
                    job.log(f"\n{kind} | {meta.resolution_class} | {meta.path.name}", echo=False)
                    cells: list[str] = []
                    for backend in (None, *backends):
                        name = "CPU" if backend is None else f"GPU/{backend}"
                        if live:
                            sys.stdout.write("\r\x1b[2K" + f"  > {number}/{len(samples)} {text} | {t('diag.status_testing', name=name)} | {human_duration(time.monotonic() - started)}"[:CONSOLE_PROGRESS._terminal_width() - 1])
                            sys.stdout.flush()
                        result = hardware_decode.check_decode(tools, settings, meta, backend)
                        rows.append({"video": str(meta.path), "kind": kind, "backend": backend or "cpu", "ok": result.ok,
                                     "seconds": round(result.seconds, 3), "speed": round(result.speed, 2), "error": result.error,
                                     "decoder_starts": result.decoder_starts, "unsupported": result.unsupported})
                        if not result.ok: job.log(f"FFmpeg output for {name} | {meta.path.name}:\n{result.detail}", echo=False)
                        job.log(f"    {name:<14} {('decodes' if backend is None else 'GPU decoding confirmed') if result.ok else ('FAILED' if backend is None else 'NOT confirmed'):<24} "
                                + (f"{result.speed:6.1f}x real time" if result.ok else result.error), echo=False)
                        cells.append(f"{result.speed:.1f}x" if result.ok else (cell_fail if backend is None else cell_not_gpu))
                    if live: sys.stdout.write("\r\x1b[2K")
                    print(f"  {f'{number}/{len(samples)}':<8}{text if len(text) <= 30 else text[:27] + '...':<30}  " + "".join(f"{c:<{cell}}" for c in cells).rstrip())
                print("\n" + t("diag.all_done", time=human_duration(time.monotonic() - started)))
            except KeyboardInterrupt:
                if live: sys.stdout.write("\r\x1b[2K")
                status = "CANCELED"
                job.say("diag.canceled_kept")
            gpu_rows = [r for r in rows if r["backend"] != "cpu"]
            lines = hardware_decode.summarize(gpu_rows, backends)
            with english(): lines_en = hardware_decode.summarize(gpu_rows, backends)                   # the report is English whatever the screen shows
            text = "GPU decoding check | " + status + "\n\n" + "\n".join(lines_en) + (
                "\n\nSpeed is seconds of video per second including starting FFmpeg and the GPU; it is only a rough comparison.\n"
                "A path that is confirmed here but not faster than the CPU in Generate is usually limited by the work done outside the decoder.")
            job.save(text, {"kind": "decode", "status": status, "ffmpeg_hash": tools.candidate.sha256, "gpu_names": list(tools.gpu_names),
                            "backends": backends, "rows": rows, "summary": lines_en})
            heading(t("diag.result_heading"))
            for line in lines: print("  " + line)
            print("\n" + t("diag.report", path=display_path(job.result_txt)))
            if not self._offer_decode_policy(rows): pause()
        except KeyboardInterrupt:
            job.say("diag.canceled")
        finally:
            job.close()

    def _offer_decode_policy(self, rows: list[dict[str, Any]]) -> bool:
        """Turn the decoding check into the rule Generate follows: a GPU path that decodes a kind of video on the CPU after all
        is not used for that kind (the pictures are identical, only the label "GPU" would be wrong). True when a question was asked."""
        results = hardware_decode.verdicts(rows)
        if not results: return False
        with english(): ACTIVITY.log("DECODE", "result: " + "; ".join(hardware_decode.describe(results)))
        tools = self._require_tools()
        gpus = list(tools.gpu_names)
        bad = hardware_decode.blocked_kinds(results)
        saved = self.config.get("acceleration", {}).get("gpu_decoding")
        known = bool(saved) and hdr_validation.same_system(saved, tools.candidate.sha256, gpus)
        fresh = [(backend, kind) for backend, kind in bad if not known or (saved.get("results") or {}).get(backend, {}).get(kind) != results[backend][kind]]
        if bad:
            heading(t("diag.generating_heading"))
            shown = fresh or bad
            print("  " + (t("diag.cpu_instead") if fresh else t("diag.cpu_instead_saved")))
            width = max(len(backend) for backend, _ in shown) + 3
            for backend in dict.fromkeys(backend for backend, _ in shown):
                print(f"    {backend:<{width}}" + ", ".join(kind for b, kind in shown if b == backend))
            if fresh and not prompt_yes_no("\n" + t("diag.use_cpu_question"), True):
                print(t("common.nothing_changed"))
                ACTIVITY.log("DECODE", "Generate keeps trying these GPU paths (declined)")
                return True
        acceleration = self.config.setdefault("acceleration", {})
        acceleration["gpu_decoding"] = hdr_validation.merge(acceleration.get("gpu_decoding"), results, tools.candidate.sha256, gpus)
        self.config_mgr.save()
        if fresh: print(t("common.saved"))
        ACTIVITY.log("DECODE", "Generate skips: " + (", ".join(f"{kind} on {backend}" for backend, kind in bad) if bad else "nothing (all tested paths decode on the GPU)"))
        return bool(fresh and bad)

    def _offer_benchmark_settings(self, data: dict[str, Any]) -> bool:
        """Suggest settings from what the benchmark measured; nothing changes without a yes. True when a question was asked."""
        advice = benchmark_advice.build_advice(data, self.config)
        with english(): advice_en = benchmark_advice.build_advice(data, self.config)                     # the activity log is English
        heading(t("diag.suggested_heading"))
        if not advice:
            print(t("diag.nothing_to_change") if len(data.get("keyframe_rows", [])) >= benchmark_advice.MIN_KEYFRAME_FILES
                  else t("diag.no_suggestion"))
            return False
        for line in benchmark_advice.advice_lines(advice): print("  " + line)
        print()
        if not prompt_yes_no(t("diag.apply_question"), False):
            print(t("common.nothing_changed"))
            ACTIVITY.log("BENCHMARK", "suggested settings declined: " + "; ".join(f"{a.key} {a.current} -> {a.proposed}" for a in advice_en))
            return True
        benchmark_advice.apply_advice(self.config, advice)
        self.config_mgr.save()
        print(t("common.saved"))
        ACTIVITY.log("BENCHMARK", "settings changed: " + "; ".join(f"{a.key} {a.current} -> {a.proposed}" for a in advice_en))
        return True

    def _choose_test_source(self, kind: str) -> Optional[tuple[str, Optional[Path]]]:
        """Where the test videos come from: ("library", None) for the media roots, ("folder", path) for a test folder of the person's own.
        None when the person goes back. Without media roots only a test folder is possible."""
        if self.config.get("media_roots"):
            choice = choose([("1", t("diag.source_library")), ("2", t("diag.source_folder")), ("0", t("common.back"))], "1")
            if choice == "0": return None
            if choice == "1": return "library", None
        folder = self._ask_test_folder(kind)
        return None if folder is None else ("folder", folder)

    @staticmethod
    def _test_title(kind: str) -> str:
        return {"benchmark": lazy("diag.title_benchmark"), "decode": lazy("diag.title_decode")}.get(kind, lazy("diag.title_hdr"))()

    def _library_videos(self, kind: str, job: DiagnosticJob, scanned: Optional[list[VideoMeta]] = None) -> list[VideoMeta]:
        """Every readable video of the media roots (what the probe cache already knows is not read again); `scanned`: already read by the caller."""
        reused = scanned is not None
        if scanned is None:
            print("\n" + t("diag.reading_library"))
            scanned = self.scan_library()
        videos = [m for m in scanned if math.isfinite(m.duration) and m.duration > 0 and m.width > 0 and m.height > 0]
        job.say("diag.library_count", echo=not reused, count=len(videos))
        return videos

    def _show_library_preflight(self, videos: list[VideoMeta], sample_info: list[dict[str, str]], backends: list[str], job: DiagnosticJob) -> None:
        """What the benchmark is about to do, in a few lines; the list of the files themselves is in the report."""
        paths = 1 + len(backends)
        tests = len(videos) * (paths + 1)
        kinds = sum(not info["why"].startswith("random") for info in sample_info)
        codecs = ", ".join(f"{codec} x{count}" for codec, count in sorted(Counter(m.codec for m in videos).items(), key=lambda item: (-item[1], item[0])))
        hdr = sorted({hdr_label(m) for m in videos if m.hdr})
        low, high = library_sample.estimate_seconds(videos, paths)
        biggest = max((library_sample.segment_bytes(m, library_sample.segment_of(m)[1]) for m in videos), default=0)
        for number, (meta, info) in enumerate(zip(videos, sample_info), 1):                 # for the report, not for the screen
            start, length = library_sample.segment_of(meta)
            job.log(f"  [{number}/{len(videos)}] {info['why']} | {meta.path} | {human_duration(start)} - {human_duration(start + length)}", echo=False)
        heading(t("heading.preflight"))
        rows = [(t("pre.source"), t("diag.pre_source_library")),
                (t("pre.files"), tp("diag.pre_files", len(videos), kinds=kinds, random=len(videos) - kinds, minutes=f"{library_sample.SEGMENT_SECONDS / 60:g}")),
                (t("pre.codecs"), codecs)]
        if hdr: rows.append((t("pre.hdr"), ", ".join(hdr)))
        rows += [(t("pre.tests"), t("diag.pre_tests", count=grouped(tests), paths=", ".join(["CPU", *[f"GPU/{b}" for b in backends]]))),
                 (t("pre.copies"), t("diag.pre_copies", size=human_bytes(biggest), folder=display_path(DIRS['temp']))),
                 (t("pre.time"), t("diag.pre_time", low=max(1, low // 60), high=max(2, high // 60))),
                 (t("pre.report"), t("diag.pre_report", path=display_path(job.path)))]
        labelled(rows)

    def _ask_test_folder(self, kind: str) -> Optional[Path]:
        """Ask for the folder to test (remembered between runs). None if the person goes back."""
        options = self.config.setdefault("diagnostics", {})
        key = "hdr_folder" if kind == "hdr" else "benchmark_folder"
        saved = options.get(key) or (options.get("benchmark_folder", "") if kind == "hdr" else "")
        print(t("diag.folder_enter"))
        while True:
            raw = remove_outer_quotes(input(t("diag.folder_prompt", saved=saved) + ": "))
            if raw == "0": return None
            raw = raw or saved
            if not raw:
                print(t("diag.folder_required"))
                continue
            try: folder = Path(raw).expanduser().resolve()
            except (OSError, ValueError):
                print(t("diag.folder_invalid"))
                continue
            if folder.is_dir(): break
            print(t("diag.folder_missing"))
        options[key] = str(folder)
        self.config_mgr.save()
        return folder

    def _collect_test_videos(self, folder: Path, job: DiagnosticJob) -> list[VideoMeta]:
        """Probe every file below `folder` and keep the usable videos, sorted by path."""
        scanner = MediaScanner(self._require_tools(), self.state, self.config)
        videos: list[VideoMeta] = []
        for path in scanner._walk_files(folder):
            if path.suffix.casefold() in KNOWN_NONVIDEO_EXTENSIONS: continue
            meta = self._require_tools().probe(path)
            if meta and math.isfinite(meta.duration) and meta.duration > 0 and meta.width > 0 and meta.height > 0:
                videos.append(meta)
                job.say("diag.found", name=path.name, kind=classify_video(meta))
            else:
                job.say("diag.not_usable", path=path)
        videos.sort(key=lambda m: str(m.path).casefold())
        return videos

    def _show_diagnostic_preflight(self, videos: list[VideoMeta], backends: list[str], mode: str, job: DiagnosticJob) -> None:
        total_segments = sum(len(diagnostic_segments(meta.duration, mode)) for meta in videos)
        heading(t("heading.preflight"))
        labels = [t("pre.files"), t("pre.paths"), t("pre.settings"), t("pre.report")]
        width = label_width(labels)
        labelled([(labels[0], t("diag.pre_files_segments", files=len(videos), segments=total_segments, tests=total_segments * (1 + len(backends)))),
                  (labels[1], ", ".join(["CPU", *[f"GPU/{b}" for b in backends]]))], width=width)
        for index, meta in enumerate(videos, 1):
            windows = diagnostic_segments(meta.duration, mode)
            kind_tag = f"{hdr_label(meta)} | " if meta.hdr else ""
            print(f"  [{index}/{len(videos)}] {kind_tag}{meta.path.name} | " + t("diag.pre_file_segments", segments=len(windows), tests=len(windows) * (1 + len(backends))))
        settings = config_settings(self.config)
        labelled([(labels[2], f"{settings.width} px, " + t("diag.pre_settings", interval=f"{settings.interval:g}", columns=settings.cols, rows=settings.rows, jpeg=settings.jpeg_quality))], width=width)
        print("  " + t("diag.pre_runs_sequentially"))
        labelled([(labels[3], display_path(job.path))], width=width)

    def _show_diagnostic_results(self, kind: str, data: dict[str, Any], job: DiagnosticJob) -> None:
        if kind == "benchmark":
            print("\n".join(benchmark_advice.screen_lines(data, self.config, Path(display_path(job.result_txt)))))
            return
        print("\n" + t("diag.report", path=display_path(job.path)))

    def review_hdr_images(self, data: dict[str, Any], folder: Path, save: Callable[[], None]) -> None:
        def label_of(row: dict[str, Any]) -> str:
            kind = row.get("hdr_label")                      # reports made before one test per HDR kind have none
            return f"{kind} | {row['backend']}" if kind else str(row["backend"])
        labels = list(dict.fromkeys(label_of(r) for r in data["rows"] if r["status"] == "PASS"))
        for label in labels: data["visual"].setdefault(label, "PENDING")
        save()
        if not labels:
            print(t("hdr_review.nothing"))
            return
        heading(t("hdr_review.heading"))
        print(t("hdr_review.explain"))
        print(t("hdr_review.folder", path=display_path(folder)))
        if prompt_yes_no("\n" + t("hdr_review.open_question"), True): platform_api.open_path(folder)
        assessments = {"1": "ACCEPTABLE", "2": "TOO_BRIGHT", "3": "TOO_DARK", "4": "WRONG_COLORS"}
        print(t("hdr_review.ctrl_c"))
        groups: dict[str, list[str]] = {}
        for label in labels: groups.setdefault(label.split(" | ")[0] if " | " in label else label, []).append(label)
        number = 0
        for kind, members in groups.items():
            if len(members) > 1:                                                    # one question when everything of a kind of HDR is fine
                names = ", ".join(m.split(" | ")[-1].replace("cpu", "CPU") for m in members)
                if prompt_yes_no("\n" + t("hdr_review.all_correct", kind=kind, names=names), True):
                    for label in members: data["visual"][label] = "ACCEPTABLE"
                    number += len(members)
                    save()
                    continue
            print("  " + t("hdr_review.open_comparisons", path=display_path(folder / 'comparisons')))
            for label in members:
                number += 1
                default = next((k for k, v in assessments.items() if v == data["visual"].get(label)), None)
                answer = prompt_choice("  " + t("hdr_review.rate", number=number, total=len(labels), label=label.replace('| cpu', '| CPU')), assessments, default)
                data["visual"][label] = assessments[answer]
                save()

    def _offer_hdr_policy(self, data: dict[str, Any]) -> None:
        """Turn the ratings of an HDR report into the rule Generate follows: a GPU path rated badly for a kind of HDR is not
        used for that kind. Only for the FFmpeg build and GPU the report was made with."""
        results = hdr_validation.review_results(data)
        if not results: return
        with english(): ACTIVITY.log("HDR", "ratings: " + "; ".join(hdr_validation.describe(results)))
        tools = self._require_tools()
        gpus = list(tools.gpu_names)
        if not hdr_validation.same_system(data, tools.candidate.sha256, gpus):
            print("\n" + t("hdr_review.other_system"))
            ACTIVITY.log("HDR", "ratings not applied: the report was made with another FFmpeg build or GPU")
            return
        heading(t("hdr_review.results_heading"))
        for line in hdr_validation.describe(results): print(f"  {line}")
        bad = hdr_validation.failing(results)
        if bad:
            print("  " + t("hdr_review.cpu_for", pairs=", ".join(t("hdr_review.kind_on", kind=kind, backend=backend) for backend, kind in bad)))
            if not prompt_yes_no(t("hdr_review.use_cpu_question"), False):
                print(t("common.nothing_changed"))
                ACTIVITY.log("HDR", "GPU use for HDR left as it was (declined)")
                return
        acceleration = self.config.setdefault("acceleration", {})
        acceleration["hdr_validation"] = hdr_validation.merge(acceleration.get("hdr_validation"), results, tools.candidate.sha256, gpus)
        self.config_mgr.save()
        print(t("common.saved") if bad else t("hdr_review.saved_stays"))
        ACTIVITY.log("HDR", "Generate will use the CPU for: " + (", ".join(f"{kind} on {backend}" for backend, kind in bad) if bad else "nothing (all rated paths stay on the GPU)"))

    def self_test(self) -> None:
        self._screen(t("maintenance.self_test"))
        self._run_self_test()
        pause()

    def _run_self_test(self) -> dict[str, Any]:
        """Run and save the self-test (no questions). Returns its results: {"checks": ..., "passed": n, "failed": n}."""
        job = DiagnosticJob("self_test")
        try:
            found = self.ensure_ffmpeg(interactive=False)
            results = run_self_test(self.config, self.state, self.ffmpeg_candidate if found else None,
                                    self.tools if found else None, job)
            text = (f"{APP_NAME} {APP_VERSION} self-test\nPASS: {results['passed']}\nFAIL: {results['failed']}\n" +
                    "\n".join(f"{v['status']} {k}: {v['detail']}" for k, v in results["checks"].items()))
            job.save(text, results)
            total = results["passed"] + results["failed"]
            print("\n" + (tp("selftest.all_passed", results["passed"]) if not results["failed"] else tp("selftest.failed", total, failed=results["failed"])))
            print(t("diag.report", path=display_path(job.path)))
            return results
        finally:
            job.close()

    def benchmark_menu(self) -> None:
        self._menu(lazy("menu.benchmark"), [
            (lazy("diag.title_benchmark"), lambda: self.run_diagnostic("benchmark"), False),
            (lazy("diag.title_hdr"), lambda: self.run_diagnostic("hdr"), False),
            (lazy("diag.title_decode"), self.run_decode_check, False),
        ])
