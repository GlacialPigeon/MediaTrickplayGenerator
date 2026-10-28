# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from .. import benchmark_advice, config_summary, hardware_decode, hdr_validation, library_sample, path_choice, platform_api, storage_workers, time_estimate
from ..activity_log import ACTIVITY
from ..formatting import display_path, human_bytes, human_duration
from ..localization import lazy, number, t, tp
from ..generator import TrickplayGenerator, disk_label
from ..logger import RunLogger
from ..models import PlanItem, VideoMeta
from ..prompts import heading, labelled, pause, prompt_yes_no
from .diagnostics import DiagnosticScreens


# What the plan says about a video, in words (the codes stay in the logs and in the unattended output of a scheduler).
PLAN_REASONS = {
    "MISSING": lazy("plan.missing"),
    "INVALID_FINAL_REBUILD": lazy("plan.invalid"),
    "SETTINGS_CHANGED": lazy("plan.settings_changed"),
    "SOURCE_CHANGED": lazy("plan.source_changed"),
    "REBUILD_REQUESTED": lazy("plan.rebuild_requested"),
    "RESTORE_PREVIOUS": lazy("plan.restore_previous"),
    "SKIPPED_VALID": lazy("plan.valid"),
    "SKIPPED_VALID_FOREIGN": lazy("plan.valid"),
    "SKIPPED_AMBIGUOUS_TARGET": lazy("plan.ambiguous"),
}


class GenerateScreens(DiagnosticScreens):
    """Generate: planning, preflight (disk estimate, unclear folders) and the run itself."""

    def _disk_estimate(self, plan: list[PlanItem], generator: TrickplayGenerator) -> dict[str, Any]:
        need = [p for p in plan if p.action == "GENERATE"]
        observed = self.state.estimate_bytes_per_second(generator.settings.hash)
        confidence = "HIGH" if observed else "LOW"
        if observed is None:
            base_bps = 0.50 * 1024 * 1024 / 60.0
            width_scale = (generator.settings.width / 640.0) ** 2
            interval_scale = 10.0 / generator.settings.interval
            quality_scale = max(0.6, min(1.8, generator.settings.jpeg_quality / 90.0))
            observed = base_bps * width_scale * interval_scale * quality_scale

        groups: dict[str, dict[str, Any]] = {}
        for item in need:
            est = max(1024.0, item.meta.duration * observed)
            g = groups.setdefault(item.storage_group, {"estimated": 0.0, "paths": []})
            g["estimated"] += est
            g["paths"].append(item.meta.path)

        for group, g in groups.items():
            g["with_margin"] = g["estimated"] * 1.20
            sample = g["paths"][0]
            try:
                usage = shutil.disk_usage(sample.parent)
                g["free"] = usage.free
                g["pass"] = usage.free >= g["with_margin"]
            except Exception:
                g["free"] = None
                g["pass"] = None
        return {"confidence": confidence, "bytes_per_second": observed, "groups": groups}

    def _preflight(self, videos: list[VideoMeta], plan: list[PlanItem], generator: TrickplayGenerator) -> tuple[bool, dict[str, int]]:
        self._apply_storage_groups(plan)
        for item in plan: generator.resolve_other_width_policy(item)
        approved = generator.confirm_ambiguous_deletes(plan)
        all_space_ok = self._print_preflight(videos, plan, generator)
        if not all_space_ok:
            print("\n" + t("preflight.no_space_warning"))
            if not prompt_yes_no(t("preflight.continue_anyway"), False): return False, approved
        return prompt_yes_no("\n" + t("preflight.start_question"), False), approved

    def _preflight_unattended(self, videos: list[VideoMeta], plan: list[PlanItem], generator: TrickplayGenerator) -> tuple[bool, dict[str, int]]:
        """Preflight without questions: unclear folders are skipped/kept and a run that would not fit on the disk does not start."""
        self._apply_storage_groups(plan)
        approved = generator.confirm_ambiguous_deletes(plan, interactive=False)
        if not self._print_preflight(videos, plan, generator):
            print("\n" + t("preflight.not_started_space"))
            return False, approved
        return True, approved

    def _acceleration_line(self) -> str:
        """The acceleration mode and, in "auto", what the benchmark found for this FFmpeg and GPU."""
        acceleration = self.config["acceleration"]
        mode = acceleration.get("mode", "auto")
        if mode != "auto" or self.tools is None: return mode
        if not self.tools.backends(): return t("preflight.accel_no_gpu")
        stored = acceleration.get("path_choice")
        if isinstance(stored, dict) and stored.get("results") and hdr_validation.same_system(stored, self.tools.candidate.sha256, list(self.tools.gpu_names)):
            return t("preflight.accel_by_kind", summary=path_choice.summary(stored["results"]))
        return t("preflight.accel_no_benchmark")

    def _gpu_jobs_text(self) -> str:
        acceleration = self.config["acceleration"]
        if acceleration.get("mode", "auto") == "cpu": return t("preflight.gpu_jobs_cpu")
        if self.tools is None or not self.tools.backends(): return t("preflight.gpu_jobs_none")
        jobs = int(acceleration.get("max_gpu_jobs", 2))
        return t("preflight.gpu_jobs_at_once", count=jobs)

    def _time_text(self, plan: list[PlanItem], jobs: int) -> str:
        """About how long the run takes: from the benchmark's speed per kind of video, else from typical speeds (a wider range)."""
        todo = [p.meta for p in plan if p.action == "GENERATE"]
        if not todo: return t("preflight.nothing_to_generate")
        tools = self.tools
        stored = self.config["acceleration"].get("path_choice")
        rate_of: Callable[[VideoMeta], Optional[float]]
        if tools is None: rate_of = lambda meta: None
        else:
            rate_of = lambda meta: path_choice.rate_of(stored, tools.candidate.sha256, list(tools.gpu_names), meta)
        fast, slow, measured = time_estimate.estimate(todo, rate_of, jobs, keyframes=bool(self.config["trickplay"].get("keyframe_only", False)),
                                                      disk_floor=self._disk_floor(plan))
        return time_estimate.describe(fast, slow, measured)

    def _disk_floor(self, plan: list[PlanItem]) -> float:
        """The time the slowest hard disk needs just to read its videos (the disks work side by side)."""
        setting = self.config.get("workers", {}).get("per_storage_group", storage_workers.AUTO)
        layout = storage_workers.layout([p for p in plan if p.action != "SKIP"], setting)
        speeds = self.config.get("disk_speed") or {}
        sizes: dict[str, float] = {}
        for item in plan:
            if item.action == "GENERATE": sizes[item.storage_group] = sizes.get(item.storage_group, 0.0) + float(item.meta.size)
        return max((time_estimate.disk_seconds(kind, sizes.get(group, 0.0), speeds.get(kind)) for group, (kind, _) in layout.items()), default=0.0)

    def _workers_text(self, plan: list[PlanItem]) -> tuple[str, int]:
        """How many videos are processed at once on each disk ("X: HDD: 1 at once; D: SSD: 2 at once") and how many in all."""
        setting = self.config.get("workers", {}).get("per_storage_group", storage_workers.AUTO)
        layout = storage_workers.layout([p for p in plan if p.action != "SKIP"], setting)
        first: dict[str, Path] = {}
        for item in plan:
            if item.action != "SKIP": first.setdefault(item.storage_group, item.meta.path)
        parts = [t("preflight.workers_disk_short", disk=disk_label(first[group]), kind=kind.upper(), count=count) for group, (kind, count) in layout.items()]
        return "; ".join(parts) if parts else t("preflight.nothing_to_generate"), max(1, sum(count for _, count in layout.values()))

    def _group_label(self, group: str, paths: list[Path]) -> str:
        """The media folders a storage group stands for, in words a person knows (the group's own name is an internal one)."""
        roots = [os.path.normpath(str(r)) for r in self.config.get("media_roots", [])]
        found: list[str] = []
        for path in paths:
            norm = os.path.normcase(os.path.normpath(str(path)))
            root = next((r for r in roots if norm.startswith(os.path.normcase(r) + os.sep) or norm == os.path.normcase(r)), None)
            if root is not None and root not in found: found.append(root)
            if len(found) > 3: break
        if not found: return group
        return ", ".join(found[:3]) + (", ..." if len(found) > 3 else "")

    def _print_preflight(self, videos: list[VideoMeta], plan: list[PlanItem], generator: TrickplayGenerator) -> bool:
        """Print the preflight report; True if every storage group has enough free space."""
        estimate = self._disk_estimate(plan, generator)
        heading(t("heading.preflight"))
        skipped = sum(1 for p in plan if p.action == "SKIP")
        generate = sum(1 for p in plan if p.action == "GENERATE")
        restore = sum(1 for p in plan if p.action == "RESTORE")
        counts = t("preflight.counts", found=number(len(videos)), generate=number(generate), valid=number(skipped)) + (" | " + t("preflight.counts_restored", count=number(restore)) if restore else "")
        s = generator.settings
        trickplay = self.config["trickplay"]
        workers_text, in_flight = self._workers_text(plan)
        rows = [(t("preflight.videos"), counts),
                (t("preflight.target"), t("preflight.target_value", width=s.width, interval=f"{s.interval:g}", columns=s.cols, rows=s.rows, jpeg=s.jpeg_quality, qscale=s.qscale)),
                (t("screen.key_frames"), config_summary.key_frames(trickplay)),
                (t("preflight.acceleration"), self._acceleration_line()),
                (t("preflight.gpu_jobs"), self._gpu_jobs_text()),
                (t("preflight.workers"), workers_text),
                (t("pre.time"), self._time_text(plan, in_flight))]
        all_space_ok = True
        for group, g in estimate["groups"].items():
            free_text = t("preflight.free", size=human_bytes(g["free"])) if g["free"] is not None else t("preflight.free_unknown")
            status = t("preflight.status_ok") if g["pass"] is True else (t("preflight.status_no_space") if g["pass"] is False else t("preflight.status_not_checked"))
            if g["pass"] is False: all_space_ok = False
            rows.append((t("preflight.disk"), t("preflight.disk_value", group=self._group_label(group, g['paths']), needs=human_bytes(g['estimated']), free=free_text, status=status)))
        labelled(rows, minimum=15)
        return all_space_ok

    def _offer_benchmark_first(self, plan: list[PlanItem], logger: RunLogger, library: Optional[list[VideoMeta]] = None) -> bool:
        """On the first Generate for this FFmpeg build and GPU there is no benchmark result yet: the fastest path per kind of video and
        whether key frames pay off are not known, so everything would run the slow way. Offers the benchmark (a few minutes, once; yes
        is the default), then True so that the caller plans again with what it saved. Only a finished benchmark stops the question; a "no"
        is asked again at the next Generate."""
        todo = [item.meta for item in plan if item.action == "GENERATE"]
        if len(todo) < benchmark_advice.MIN_KEYFRAME_FILES: return False             # a few videos are not worth a benchmark
        tools = self._require_tools()
        acceleration = self.config["acceleration"]
        if path_choice.benchmark_known(acceleration, tools.candidate.sha256, list(tools.gpu_names)): return False
        paths = 1 + len(tools.backends())
        sample = [p.meta for p in library_sample.pick_sample(todo, rng=random.Random(0))]
        low, high = library_sample.estimate_seconds(sample, paths)
        heading(t("first.heading"))
        print(t("first.none_yet") + "\n")
        print(tp("first.explain", len(sample)) + "\n")
        print(t("first.takes", low=max(1, low // 60), high=max(2, high // 60)))
        print(t("first.repeat"))
        if not prompt_yes_no("\n" + t("first.question"), True):
            logger.log("INFO", "Benchmark before Generate declined (asked again next time, until a benchmark has run)")
            return False
        logger.log("INFO", "Benchmark before Generate started")
        ran = self.run_diagnostic("benchmark", before_generate=True, library=library)
        logger.log("INFO", f"Benchmark before Generate {'finished' if ran else 'did not finish'}")
        return ran

    def _check_new_kinds_of_video(self, plan: list[PlanItem], generator: TrickplayGenerator, logger: RunLogger) -> bool:
        """Before generating: find out, without asking, whether the GPU really decodes every kind of video (codec and bit depth)
        that has no result yet for this FFmpeg build and GPU, and remember it, also what the card cannot do at all, so that
        Generate does not try those paths again and again. Nothing is printed when every kind is known already.
        False if the person interrupted the check."""
        if self.config["acceleration"].get("mode", "auto") != "auto": return True       # "cpu" uses no GPU; "gpu" is the person's decision
        tools = self._require_tools()
        backends = generator._preferred_backends()
        videos = [item.meta for item in plan if item.action == "GENERATE"]
        if not backends or not videos: return True
        acceleration = self.config["acceleration"]
        shown = False
        outcomes: list[tuple[str, str, str]] = []                      # (backend, kind, what came of it)
        live = sys.stdout.isatty()

        def before(sample: VideoMeta, backend: str) -> None:
            nonlocal shown
            if not shown:
                heading(t("check.heading"))
                print(t("check.kinds_not_checked"))
                shown = True
            if live: sys.stdout.write("\r\x1b[2K  " + t("check.checking", kind=hardware_decode.sample_key(sample), backend=backend))
            sys.stdout.flush()

        def after(sample: VideoMeta, backend: str, result: hardware_decode.DecodeResult, verdict: Optional[str]) -> None:
            kind = hardware_decode.sample_key(sample)
            text = {hdr_validation.PASS: "decodes on the GPU",
                    hdr_validation.FAIL: "decoded on the CPU after all - Generate will skip it",
                    hardware_decode.UNSUPPORTED: "not supported by the GPU - Generate will skip it"}.get(verdict or "", "not decided (" + (result.error or "no reason") + ") - Generate will try it")
            outcomes.append((backend, kind, verdict or "undecided"))
            logger.log("INFO", f"GPU decoding check: {kind} on {backend}: {text}; {sample.path.name}")
            ACTIVITY.log("DECODE", f"before Generate: {kind} on {backend}: {text}")
            if verdict is None and result.detail: logger.log("INFO", f"GPU decoding check output:\n{result.detail}")

        try:
            found = hardware_decode.precheck(tools, generator.settings, videos, acceleration.get("gpu_decoding"),
                                             lambda meta: generator._backends_for(meta, "auto"), self.state.known_unsupported, before, after)
        except KeyboardInterrupt:
            if live and shown: sys.stdout.write("\r\x1b[2K")
            print("\n" + t("check.canceled"))
            return False
        if live and shown: sys.stdout.write("\r\x1b[2K")
        if found:
            acceleration["gpu_decoding"] = hdr_validation.merge(acceleration.get("gpu_decoding"), found, tools.candidate.sha256, list(tools.gpu_names))
            self.config_mgr.save()
        if outcomes:
            samples = {hardware_decode.sample_key(sample): sample for sample in hardware_decode.pick_samples(videos, None)}
            kinds = list(dict.fromkeys(kind for _, kind, _ in outcomes))

            def cell(kind: str, backend: str) -> str:
                this_time = next((verdict for b, k, verdict in outcomes if (b, k) == (backend, kind)), None)
                verdict = this_time or hardware_decode.stored_result(acceleration.get("gpu_decoding"), tools.candidate.sha256, list(tools.gpu_names), backend, samples[kind])
                if verdict == hdr_validation.PASS: return "ok"
                if verdict in (hdr_validation.FAIL, hardware_decode.UNSUPPORTED): return "no"
                if verdict == "undecided": return "undecided"
                return "no" if self.state.known_unsupported(backend, samples[kind]) else "untested"

            for line in hardware_decode.compatibility_lines(kinds, backends, cell): print(line)
        return True

    def generate(self, *, dry_run: bool = False, single: Optional[Path] = None) -> None:
        self._screen(t("screen.dry_run") if dry_run else t("screen.generate"))
        if not single and not self.config.get("media_roots"):
            print(t("msg.no_roots"))
            pause()
            return

        if not self.ensure_ffmpeg(): return
        if not self._require_pillow():
            pause()
            return

        logger = RunLogger("dry-run" if dry_run else "generate")
        self.state.begin_run(logger.run_id, logger.mode)
        try:
            if single:
                self.unreadable_videos = []
                meta = self.tools.probe(single) if self.tools else None
                if not meta:
                    print(t("generate.not_readable", path=single))
                    pause()
                    return
                self.state.upsert_video(meta)
                videos = [meta]
            else:
                videos = self.scan_library(logger)
                if not videos:
                    print(t("msg.no_videos"))
                    pause()
                    return

            if self._plan_and_run(videos, logger, dry_run=dry_run, offer_benchmark=single is None) or dry_run:
                pause()
        finally: logger.close()

    @staticmethod
    def _print_dry_run(plan: list[PlanItem], limit: int = 200) -> None:
        """What a real run would do: the numbers first, then every video that would be touched (not the ones that are fine)."""
        valid = [p for p in plan if p.action == "SKIP" and p.reason.startswith("SKIPPED_VALID")]
        todo = [p for p in plan if not (p.action == "SKIP" and p.reason.startswith("SKIPPED_VALID"))]
        counts = {"GENERATE": lazy("dry.to_generate"), "RESTORE": lazy("dry.to_restore"), "SKIP": lazy("dry.left_alone")}
        parts = [t("dry.part", count=number(sum(p.action == a for p in todo)), what=text()) for a, text in counts.items() if any(p.action == a for p in todo)]
        parts.append(t("dry.already_valid", count=number(len(valid))))
        heading(t("dry.heading", parts=", ".join(parts)))
        for item in todo[:limit]:
            reason = PLAN_REASONS.get(item.reason)
            print(f"  {reason() if reason else item.reason.lower()} | {item.meta.path}")
        if len(todo) > limit: print("  " + t("dry.more", count=number(len(todo) - limit)))
        if not todo: print("  " + t("dry.nothing"))

    def _plan_and_run(self, videos: list[VideoMeta], logger: RunLogger, *, dry_run: bool = False, interactive: bool = True, offer_benchmark: bool = False) -> bool:
        """Plan, confirm (disk estimate, ambiguous deletes) and run the generation for `videos`. True if it ran to a summary.
        Without a person to ask (`interactive=False`) nothing is confirmed: unclear folders are left alone and a run that
        would not fit on the disk is not started. The summary stays in `last_summary`."""
        self.last_summary = None
        generator = TrickplayGenerator(self._require_tools(), self.state, self.config, logger)
        plan = generator.plan(videos)
        self._apply_storage_groups(plan)

        if not dry_run and not self._check_new_kinds_of_video(plan, generator, logger):            # first: the benchmark and the preflight come after what each GPU path can do
            logger.event("CANCELED", reason="gpu_decoding_check_interrupted")
            return False

        if interactive and offer_benchmark and not dry_run and self._offer_benchmark_first(plan, logger, videos):
            generator = TrickplayGenerator(self._require_tools(), self.state, self.config, logger)          # the settings it saved
            plan = generator.plan(videos)
            self._apply_storage_groups(plan)

        if dry_run:
            if not interactive: generator.confirm_ambiguous_deletes(plan, interactive=False)       # list what the real run would do
            self._print_dry_run(plan)
            return False

        proceed, approved = self._preflight(videos, plan, generator) if interactive else self._preflight_unattended(videos, plan, generator)
        if not proceed:
            logger.event("CANCELED", reason="preflight_not_confirmed")
            return False

        with platform_api.prevent_sleep(bool(self.config.get("runtime", {}).get("prevent_sleep", True))):
            summary = generator.run_plan(plan, approved)
        summary["unreadable_videos"] = len(self.unreadable_videos)
        self.last_summary = summary
        logger.write_summary(summary)
        self.state.end_run(logger.run_id, summary)

        self._print_final_summary(summary, logger.folder)
        return True

    @staticmethod
    def _print_final_summary(summary: dict[str, Any], log_path: Optional[Path] = None) -> None:
        """Scanned, completed, failed and the time always; everything else only when it is not zero."""
        labels = {"found": t("final.found"), "completed": t("final.completed"), "failed": t("final.failed"), "skipped_valid": t("final.skipped_valid"),
                  "restored": t("final.restored"), "skipped": t("final.skipped_other"), "unclear": t("final.unclear"), "canceled": t("final.not_started"),
                  "source_changed_during_processing": t("final.source_changed"), "source_missing": t("final.source_missing"), "unreadable": t("final.unreadable"),
                  "suspect": t("final.suspect"), "elapsed": t("final.elapsed"), "log": t("final.log")}
        width = max(26, max(len(label) for label in labels.values()) + 2)               # a translated label is never glued to its number

        def row(label: str, value: Any, note: str = "") -> None: print(f"  {label + ':':<{width}}{number(value):>7}" + (f"  {note}" if note else ""))
        heading(t("final.heading"))
        row(labels["found"], summary["scanned"])
        row(labels["completed"], summary["completed"])
        row(labels["failed"], summary["failed"])
        for key in ("skipped_valid", "restored", "skipped"):
            if summary.get(key): row(labels[key], summary[key])
        if summary.get("skipped_unclear_folder"): row(labels["unclear"], summary["skipped_unclear_folder"], t("final.unclear_note"))
        if summary.get("canceled"): row(labels["canceled"], summary["canceled"], t("final.not_started_note"))
        for key in ("source_changed_during_processing", "source_missing"):
            if summary.get(key): row(labels[key], summary[key])
        if summary.get("unreadable_videos"): row(labels["unreadable"], summary["unreadable_videos"], t("final.unreadable_note"))
        if summary.get("suspect_videos"):
            row(labels["suspect"], summary["suspect_videos"], t("final.suspect_note"))
            for detail in summary["suspect_details"][:10]: print(f"    {detail['path']}\n      - {detail['note']}")
            if len(summary["suspect_details"]) > 10: print("    " + t("final.suspect_more", count=number(len(summary["suspect_details"]) - 10)))
        print(f"  {labels['elapsed'] + ':':<{width}}{human_duration(summary['elapsed_seconds']):>7}")
        if log_path is not None: print(f"  {labels['log'] + ':':<{width}}{display_path(log_path)}")
