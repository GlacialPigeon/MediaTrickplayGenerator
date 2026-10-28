# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import os
import queue
import re
import shutil
import sqlite3
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from . import platform_api, storage_workers
from .console_progress import CONSOLE_PROGRESS
from .constants import DIRS, KNOWN_VIDEO_EXTENSIONS
from .datetime_helpers import local_stamp
from .extraction import FrameExtraction
from .ffmpeg_tools import FFmpegTools
from .fingerprint import source_edge_fingerprint
from .formatting import human_bytes
from .logger import RunLogger
from .models import ExtractResult, PlanItem, ValidationResult, VideoMeta
from .paths import normalize_path
from .localization import t
from .prompts import choose
from .state_db import StateDB
from .storage import storage_kind
from .verify import INVALID, MISSING, Restorable, VerifyRow, find_leftovers, find_restorable, restore_previous
from .tiles import (
    building_dir_for, config_settings, describe_frame_deviation, frame_count_deviation, list_other_widths,
    parse_trickplay_width_dir, quick_validate_tile_directory, target_dir_for, trickplay_root_for, validate_tile_directory,
)


def disk_label(path: Path) -> str:
    """The disk a video is on, short: `X:` on Windows, the share for a network path, the first folder elsewhere."""
    if path.drive: return path.drive if len(path.drive) <= 10 else "\\\\" + path.drive.strip("\\").split("\\")[0]
    return "/" + path.parts[1] if len(path.parts) > 2 else "/"


def library_disks(roots: list[Any]) -> list[str]:
    """The disks the media folders are on, short and with their kind: ["D: SSD", "X: HDD"]."""
    found: dict[str, str] = {}
    for root in roots:
        path = Path(str(root))
        found.setdefault(disk_label(path), storage_workers.kind_label(storage_kind(path)))
    return [f"{disk} {kind}" for disk, kind in found.items()]


class TrickplayGenerator(FrameExtraction):
    def __init__(self, tools: FFmpegTools, state: StateDB, config: dict[str, Any], logger: RunLogger) -> None:
        self.tools = tools
        self.state = state
        self.config = config
        self.logger = logger
        self.settings = config_settings(config)
        self.gpu_sem = threading.Semaphore(max(1, int(config["acceleration"].get("max_gpu_jobs", 2))))
        self.verify_all = bool(config["verification"].get("verify_all_tiles", True))
        self.strong_fingerprint = bool(config["source_detection"].get("strong_fingerprint", False))
        self.fingerprint_chunk_mib = int(config["source_detection"].get("fingerprint_chunk_mib", 1))
        self.keyframe_only = bool(config["trickplay"].get("keyframe_only", False))
        self.keyframe_repeat_limit = float(config["trickplay"].get("keyframe_repeat_limit_percent", 10.0))
        self.stop_event = threading.Event()
        self.drain_event = threading.Event()

    def request_stop(self) -> None:
        if self.drain_event.is_set():
            self.stop_event.set()
            CONSOLE_PROGRESS.set_stop(immediate=True)
        else:
            self.drain_event.set()
            CONSOLE_PROGRESS.set_stop(immediate=False)

    def _check_target_conflicts(self, videos: list[VideoMeta]) -> None:
        targets: dict[str, Path] = {}
        siblings_by_parent: dict[str, dict[str, list[Path]]] = {}
        for meta in videos:
            target = normalize_path(trickplay_root_for(meta.path))
            if target in targets:
                raise ValueError(f"Conflicting trickplay target: {targets[target]} and {meta.path}")
            targets[target] = meta.path
            # Also protect single-video runs from a sibling with the same stem.
            parent_key = normalize_path(meta.path.parent)
            if parent_key not in siblings_by_parent:
                sibling_targets: dict[str, list[Path]] = {}
                for sibling in meta.path.parent.iterdir():
                    if sibling.suffix.casefold() in KNOWN_VIDEO_EXTENSIONS and sibling.is_file():
                        sibling_targets.setdefault(normalize_path(trickplay_root_for(sibling)), []).append(sibling)
                siblings_by_parent[parent_key] = sibling_targets
            for sibling in siblings_by_parent[parent_key].get(target, []):
                if normalize_path(sibling) != normalize_path(meta.path):
                    raise ValueError(f"Conflicting trickplay target: {meta.path} and {sibling}")

    def _output_state_matches(self, meta: VideoMeta, previous: Optional[sqlite3.Row], fingerprint: Optional[str]) -> bool:
        if previous is None: return True
        if int(previous["source_size"]) != meta.size or int(previous["source_mtime_ns"]) != meta.mtime_ns: return False
        if previous["source_duration"] and meta.duration:
            if abs(float(previous["source_duration"]) - meta.duration) > max(1.0, self.settings.interval): return False
        prev_fp = previous["source_fingerprint"]
        if prev_fp and self.strong_fingerprint:
            if not fingerprint: return False
            if prev_fp != fingerprint: return False
        return True

    def _source_fingerprint(self, path: Path) -> Optional[str]:
        if not self.strong_fingerprint: return None
        try: return source_edge_fingerprint(path, self.fingerprint_chunk_mib)
        except OSError: return None

    def plan(self, videos: list[VideoMeta]) -> list[PlanItem]:
        self._check_target_conflicts(videos)
        plan: list[PlanItem] = []
        same_policy = self.config["trickplay"].get("same_width_policy", "skip_valid")
        for meta in videos:
            target = target_dir_for(meta.path, self.settings)
            previous = self.state.current_target_output(meta.path, self.settings)
            fingerprint = self._source_fingerprint(meta.path) if previous and previous["source_fingerprint"] else None
            source_matches = self._output_state_matches(meta, previous, fingerprint)
            exact_tiles = int(previous["tiles"]) if previous and source_matches else None
            validation = ValidationResult(False, "missing directory")
            if target.exists():
                # An output this program wrote and recorded (same settings, same source) is trusted when its file names and sizes
                # still match the record: reading every picture of every video at each run is slow, most of all over a network.
                # Anything else, and any difference, goes through the full check. Verify always reads every picture.
                quick = None
                if previous and source_matches and previous["settings_hash"] == self.settings.hash:
                    quick = quick_validate_tile_directory(target, tiles=int(previous["tiles"]), output_bytes=int(previous["output_bytes"] or 0))
                validation = quick or validate_tile_directory(target, self.settings, meta.duration, expected_exact_tiles=exact_tiles, verify_all=self.verify_all)
            other = list_other_widths(meta.path, self.settings)

            if previous and previous["settings_hash"] != self.settings.hash:
                action, reason, source_changed = "GENERATE", "SETTINGS_CHANGED", False
            elif previous and not source_matches:
                action, reason, source_changed = "GENERATE", "SOURCE_CHANGED", True
            elif validation.valid and same_policy == "skip_valid":
                action, reason, source_changed = "SKIP", "SKIPPED_VALID" if previous else "SKIPPED_VALID_FOREIGN", False
            elif validation.valid and same_policy == "rebuild_all":
                action, reason, source_changed = "GENERATE", "REBUILD_REQUESTED", False
            else:
                action, reason, source_changed = "GENERATE", "INVALID_FINAL_REBUILD" if target.exists() else "MISSING", False

            restorable = None
            if action == "GENERATE" and reason in {"MISSING", "INVALID_FINAL_REBUILD"}:
                restorable = self._restorable_previous(meta, reason)
                if restorable is not None: action, reason = "RESTORE", "RESTORE_PREVIOUS"

            plan.append(PlanItem(meta=meta, action=action, reason=reason, validation=validation, source_changed=source_changed, other_widths=other, restorable=restorable))
        return plan

    def _restorable_previous(self, meta: VideoMeta, reason: str) -> Optional[Restorable]:
        """A still valid `.__old__` copy (an interrupted replacement) that can take the place of the missing or damaged output.
        Restoring it is a rename, so it is always cheaper than building the output again."""
        leftovers = find_leftovers(meta.path)
        if not leftovers: return None
        target = target_dir_for(meta.path, self.settings)
        if target.exists() and not self._is_safe_trickplay_dir_for_delete(target)[0]: return None   # unexpected files: the usual question/skip applies
        row = VerifyRow(meta.path, MISSING if reason == "MISSING" else INVALID, leftovers=leftovers)
        return find_restorable(meta, self.settings, row, verify_all=self.verify_all)

    def _restore_previous_output(self, item: PlanItem) -> bool:
        """Put the previous output back. False (and the video is generated as usual) when that is not possible after all."""
        restorable = item.restorable
        if restorable is None: return False
        problem = restore_previous(restorable)
        if problem is not None:
            self.logger.event("RESTORE_FAILED", path=str(item.meta.path), previous=str(restorable.path), reason=problem)
            self.logger.log("WARN", f"Could not restore the previous output, generating it instead: {item.meta.path} ({problem})")
            return False
        self.logger.event("RESTORED", path=str(item.meta.path), previous=str(restorable.path), target=str(restorable.target), storage_group=item.storage_group)
        self.logger.log("INFO", f"RESTORED the previous output instead of generating it: {item.meta.path}")
        return True

    def resolve_other_width_policy(self, item: PlanItem) -> str:
        """"replace" (remove the other widths once the target is complete), "keep" or "none" (there are no other widths)."""
        if item.other_policy is not None: return item.other_policy
        if not item.other_widths: item.other_policy = "none"
        else: item.other_policy = "replace" if self.config["trickplay"].get("other_width_policy", "keep_add") == "replace" else "keep"
        return item.other_policy

    def _is_safe_trickplay_dir_for_delete(self, path: Path) -> tuple[bool, list[str]]:
        unknown: list[str] = []
        if not path.is_dir(): return False, ["not a directory"]
        parsed = parse_trickplay_width_dir(path)
        if not parsed: return False, ["unrecognized directory name"]
        try:
            for child in path.iterdir():
                if child.is_file() and re.fullmatch(r"\d+\.jpg", child.name, re.IGNORECASE): continue
                unknown.append(child.name)
        except OSError as exc: return False, [str(exc)]
        return len(unknown) == 0, unknown

    def confirm_ambiguous_deletes(self, plan: list[PlanItem], *, interactive: bool = True) -> dict[str, int]:
        """Ask before replacing/deleting folders that hold unexpected files. Without a person to ask (`interactive=False`)
        nothing is approved: such a target is skipped and such old widths are kept."""
        approved: dict[str, int] = {}
        for item in plan:
            target = target_dir_for(item.meta.path, self.settings)
            if item.action == "GENERATE" and target.exists():
                safe, unknown = self._is_safe_trickplay_dir_for_delete(target)
                if not safe and not interactive:
                    item.action = "SKIP"
                    item.reason = "SKIPPED_AMBIGUOUS_TARGET"
                    print(t("unexpected.skipped", path=target))
                elif not safe:
                    print("\n" + t("unexpected.in_existing"))
                    print(f"  {target}")
                    for name in unknown[:20]: print(f"    - {name}")
                    print("\n" + t("unexpected.wont_replace"))
                    while True:
                        c = choose([("1", t("unexpected.keep_skip")), ("2", t("unexpected.replace")), ("3", t("unexpected.open"))], "1")
                        if c == "3":
                            platform_api.open_path(target)
                            continue
                        if c == "2":
                            try: approved[normalize_path(target)] = target.stat().st_mtime_ns
                            except OSError:
                                item.action = "SKIP"
                                item.reason = "SKIPPED_AMBIGUOUS_TARGET"
                        else:
                            item.action = "SKIP"
                            item.reason = "SKIPPED_AMBIGUOUS_TARGET"
                        break
            if self.resolve_other_width_policy(item) != "replace": continue
            for old in item.other_widths:
                safe, unknown = self._is_safe_trickplay_dir_for_delete(old)
                if safe: continue
                if not interactive:
                    print(t("unexpected.kept", path=old))
                    continue
                print("\n" + t("unexpected.in_old"))
                print(f"  {old}")
                for name in unknown[:20]: print(f"    - {name}")
                print("\n" + t("unexpected.wont_delete"))
                while True:
                    c = choose([("1", t("unexpected.keep")), ("2", t("unexpected.delete")), ("3", t("unexpected.open"))], "1")
                    if c == "3":
                        platform_api.open_path(old)
                        continue
                    if c == "2":
                        try: approved[normalize_path(old)] = old.stat().st_mtime_ns
                        except OSError: pass
                    break
        return approved

    def _clean_stale_build(self, item: PlanItem) -> None:
        building = building_dir_for(item.meta.path, self.settings)
        if building.exists():
            self.logger.event("RECOVERED_AFTER_INTERRUPTION", path=str(item.meta.path), building_dir=str(building), action="remove_stale_build_and_rebuild")
            try: shutil.rmtree(building)
            except OSError as exc: raise RuntimeError(f"cannot remove stale building directory: {exc}")

    def _ambiguous_approval_still_valid(self, path: Path, approved_ambiguous: dict[str, int]) -> bool:
        key = normalize_path(path)
        expected_mtime = approved_ambiguous.get(key)
        if expected_mtime is None: return False
        try: return path.stat().st_mtime_ns == expected_mtime
        except OSError: return False

    # Pauses between attempts when Windows says a file or folder is in use (about 5 s in all). Short locks from
    # an antivirus scan, the indexer or a media server scanning the folder are gone by then.
    LOCK_RETRY_DELAYS: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0)

    def _replace_with_retry(self, src: Path, dst: Path, cancelable: bool = True) -> None:
        """os.replace that waits out a short-lived lock on Windows before giving up.

        Only lock errors are retried; anything else (and the last lock error) is raised unchanged.
        """
        delays = self.LOCK_RETRY_DELAYS
        for attempt in range(len(delays) + 1):
            try:
                os.replace(src, dst)
                return
            except OSError as exc:
                if attempt >= len(delays) or not platform_api.is_transient_lock_error(exc): raise
                self.logger.log("WARN", f"{src} is in use by another program ({exc}); retry {attempt + 1}/{len(delays)} in {delays[attempt]:g} s")
                if cancelable:
                    self.stop_event.wait(delays[attempt])
                    self._check_canceled()
                else:
                    time.sleep(delays[attempt])

    def _restore_backup(self, backup: Optional[Path], target: Path) -> None:
        """Put the previous valid output back after a failed replacement.

        If even that fails the previous output is still intact in the backup folder, so say exactly where it is.
        """
        if not (backup and backup.exists() and not target.exists()): return
        try: self._replace_with_retry(backup, target, cancelable=False)
        except Exception as exc:
            self.logger.say("ERROR", "generate.rollback_failed", backup=backup, target=target, error=exc)
            self.logger.event("ROLLBACK_FAILED", backup=str(backup), target=str(target), error=str(exc))

    def _finalize_target(self, item: PlanItem, building: Path, source_fp: Optional[str], extract: ExtractResult, frames: int, tiles: int, approved_ambiguous: dict[str, int]) -> tuple[bool, str, int]:
        self._check_canceled()
        target = target_dir_for(item.meta.path, self.settings)
        root = target.parent
        root.mkdir(parents=True, exist_ok=True)
        backup: Optional[Path] = None
        if target.exists():
            safe, unknown = self._is_safe_trickplay_dir_for_delete(target)
            if not safe and not self._ambiguous_approval_still_valid(target, approved_ambiguous):
                return False, f"target changed or contains unapproved files; refusing replacement: {unknown}", 0
            backup = root / f"{target.name}.__old__.{local_stamp()}.{uuid.uuid4().hex[:6]}"
            self._replace_with_retry(target, backup)
        try: self._replace_with_retry(building, target)
        except Exception:
            self._restore_backup(backup, target)
            raise
        validation = validate_tile_directory(target, self.settings, item.meta.duration, expected_exact_tiles=tiles, verify_all=True)
        if not validation.valid:
            failed = root / f"{target.name}.__failed_finalize__.{local_stamp()}"
            try: os.replace(target, failed)
            except Exception as exc:
                self.logger.say("WARN", "generate.move_aside_failed", target=target, failed=failed, error=exc)
            self._restore_backup(backup, target)
            return False, f"final validation failed: {validation.reason}", 0

        method = "GPU" if extract.method == "GPU" else "CPU"
        self.state.record_output(item.meta, self.settings, frames, tiles, validation.output_bytes, method=method, backend=extract.backend, source_fingerprint=source_fp)
        status = "GPU_DONE" if extract.method == "GPU" else ("CPU_FALLBACK_DONE" if extract.fallback else "CPU_DONE")
        self.logger.event(status, path=str(item.meta.path), storage_group=item.storage_group, method=extract.method, backend=extract.backend,
                          duration=item.meta.duration, frames=frames, tiles=tiles, elapsed=extract.elapsed, output_bytes=validation.output_bytes, keyframe_only=extract.keyframe_only, settings=asdict(self.settings))
        if backup and backup.exists():
            try: shutil.rmtree(backup)
            except Exception as exc: self.logger.log("WARN", f"Could not remove old same-width backup {backup}: {exc}")
        return True, "OK", validation.output_bytes

    def _report_suspect_frame_count(self, item: PlanItem, frames: int) -> Optional[dict[str, Any]]:
        """The output is kept as it is (it shows what the file really holds), but a number of thumbnails that does not fit the
        length of the video is reported: in Jellyfin every thumbnail after the first missing or repeated one is at the wrong time."""
        meta = item.meta
        deviation = frame_count_deviation(frames, meta.duration, self.settings.interval)
        if deviation is None: return None
        expected, found = deviation
        text = describe_frame_deviation(expected, found, self.settings.interval)
        self.logger.event("SUSPECT_FRAME_COUNT", path=str(meta.path), storage_group=item.storage_group, duration=meta.duration, frames=found,
                          expected_frames=expected, interval=self.settings.interval)
        self.logger.log("WARN", f"SUSPECT {meta.path}: {text}", echo=True)
        return {"path": str(meta.path), "frames": found, "expected_frames": expected, "duration": meta.duration, "note": text}

    def _delete_old_widths_last(self, item: PlanItem, approved_ambiguous: dict[str, int]) -> None:
        for old in item.other_widths:
            if not old.exists(): continue
            safe, unknown = self._is_safe_trickplay_dir_for_delete(old)
            if not safe and not self._ambiguous_approval_still_valid(old, approved_ambiguous):
                self.logger.log("WARN", f"Preserving ambiguous old trickplay directory: {old}")
                continue
            try:
                shutil.rmtree(old)
                self.logger.event("OLD_WIDTH_REMOVED", path=str(item.meta.path), removed=str(old))
            except Exception as exc: self.logger.log("WARN", f"Could not remove old trickplay width {old}: {exc}")

    def process_item(self, item: PlanItem, approved_ambiguous: dict[str, int], index_text: str = "") -> dict[str, Any]:
        self._check_canceled()
        meta = item.meta
        if not meta.path.exists():
            self.logger.event("SOURCE_MISSING", path=str(meta.path), storage_group=item.storage_group)
            return {"status": "SOURCE_MISSING", "path": str(meta.path)}

        other_policy = self.resolve_other_width_policy(item)
        if item.action == "SKIP" and building_dir_for(meta.path, self.settings).exists(): self._clean_stale_build(item)
        if item.action == "SKIP":
            if item.reason in {"SKIPPED_VALID", "SKIPPED_VALID_FOREIGN"}:
                self.logger.event(item.reason, path=str(meta.path), storage_group=item.storage_group)
                if other_policy == "replace": self._delete_old_widths_last(item, approved_ambiguous)
                return {"status": "SKIPPED_VALID", "path": str(meta.path)}
            self.logger.event(item.reason, path=str(meta.path), storage_group=item.storage_group)
            return {"status": "SKIPPED", "path": str(meta.path), "unclear_folder": item.reason == "SKIPPED_AMBIGUOUS_TARGET"}

        if item.action == "RESTORE" and self._restore_previous_output(item):
            if building_dir_for(meta.path, self.settings).exists(): self._clean_stale_build(item)
            if other_policy == "replace": self._delete_old_widths_last(item, approved_ambiguous)
            return {"status": "RESTORED", "path": str(meta.path)}
        if item.action == "RESTORE": item.reason = "INVALID_FINAL_REBUILD" if target_dir_for(meta.path, self.settings).exists() else "MISSING"

        if item.source_changed: self.logger.event("SOURCE_CHANGED", path=str(meta.path), storage_group=item.storage_group)
        elif item.reason == "INVALID_FINAL_REBUILD":
            self.logger.event("INVALID_FINAL_REBUILD", path=str(meta.path), reason=item.validation.reason if item.validation else "invalid")

        before = meta.path.stat()
        if (before.st_size, before.st_mtime_ns) != meta.signature:
            return {"status": "SOURCE_CHANGED_DURING_PROCESSING", "path": str(meta.path)}
        source_fp = self._source_fingerprint(meta.path)
        self._clean_stale_build(item)
        building = building_dir_for(item.meta.path, self.settings)
        building.parent.mkdir(parents=True, exist_ok=True)
        building.mkdir(parents=True, exist_ok=True)
        job_temp = DIRS["temp"] / f"run_{self.logger.run_id}" / f"job_{uuid.uuid4().hex[:10]}"
        frames_temp = job_temp / "frames"

        try:
            extract = self._choose_method_and_extract(meta, frames_temp, index_text or meta.path.name)
            if not extract.success:
                status = "CANCELED" if extract.canceled else "FAILED"
                self.logger.event(status, path=str(meta.path), error=extract.error[-4000:], method=extract.method, backend=extract.backend)
                if status == "FAILED": self.logger.failed(meta.path, extract.error)
                return {"status": status, "path": str(meta.path), "error": extract.error, "attempts": extract.attempts}

            extract, frames_dir = self._redo_with_every_frame_if_sparse(meta, extract, frames_temp, index_text or meta.path.name)
            extract, frames_dir = self._normalize_timestamps_if_suspect(meta, extract, frames_dir, index_text or meta.path.name)
            pack_start = time.monotonic()
            frames, tiles, thumb = self._pack_frames(frames_dir, building)
            pack_elapsed = time.monotonic() - pack_start
            build_validation = validate_tile_directory(building, self.settings, meta.duration, expected_exact_tiles=tiles, verify_all=True)
            if not build_validation.valid:
                error = f"building validation failed: {build_validation.reason}"
                self.logger.event("FAILED", path=str(meta.path), error=error)
                self.logger.failed(meta.path, error)
                return {"status": "FAILED", "path": str(meta.path), "error": error, "attempts": extract.attempts}

            try: after = meta.path.stat()
            except OSError:
                self.logger.event("SOURCE_MISSING", path=str(meta.path), action="not_finalized")
                return {"status": "SOURCE_MISSING", "path": str(meta.path)}

            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                self.logger.event("SOURCE_CHANGED_DURING_PROCESSING", path=str(meta.path), action="not_finalized_rebuild_next_run")
                return {"status": "SOURCE_CHANGED_DURING_PROCESSING", "path": str(meta.path), "attempts": extract.attempts}

            if self.strong_fingerprint:
                after_fp = self._source_fingerprint(meta.path)
                if source_fp and after_fp and after_fp != source_fp:
                    self.logger.event("SOURCE_CHANGED_DURING_PROCESSING", path=str(meta.path), action="fingerprint_changed_not_finalized")
                    return {"status": "SOURCE_CHANGED_DURING_PROCESSING", "path": str(meta.path), "attempts": extract.attempts}

            ok, msg, out_bytes = self._finalize_target(item, building, source_fp, extract, frames, tiles, approved_ambiguous)
            if not ok:
                self.logger.event("FAILED", path=str(meta.path), error=msg)
                self.logger.failed(meta.path, msg)
                return {"status": "FAILED", "path": str(meta.path), "error": msg, "attempts": extract.attempts}

            if other_policy == "replace": self._delete_old_widths_last(item, approved_ambiguous)
            self.logger.log("INFO", f"DONE {meta.path} | {extract.method}{('/'+extract.backend) if extract.backend else ''} | extract={extract.elapsed:.2f}s pack={pack_elapsed:.2f}s frames={frames} tiles={tiles} size={human_bytes(out_bytes)}")
            result: dict[str, Any] = {"status": "DONE", "path": str(meta.path), "method": extract.method, "backend": extract.backend, "frames": frames, "tiles": tiles, "elapsed": extract.elapsed + pack_elapsed, "output_bytes": out_bytes}
            suspect = self._report_suspect_frame_count(item, frames)
            if suspect is not None: result["suspect"] = suspect
            return result

        except (KeyboardInterrupt, InterruptedError):
            self.logger.event("CANCELED", path=str(meta.path), action="force_interrupt")
            return {"status": "CANCELED", "path": str(meta.path)}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.logger.event("FAILED", path=str(meta.path), error=error, traceback=traceback.format_exc()[-6000:])
            self.logger.failed(meta.path, error)
            culprit = [str(name) for name in (getattr(exc, "filename", None), getattr(exc, "filename2", None)) if name]
            return {"status": "FAILED", "path": str(meta.path), "error": error, "culprit": culprit}
        finally:
            shutil.rmtree(job_temp, ignore_errors=True)
            try: job_temp.parent.rmdir()  # the shared per-run folder: only goes when no other job is still using it
            except OSError: pass
            if building.exists():
                try: shutil.rmtree(building)
                except Exception as exc: self.logger.log("WARN", f"Could not remove the incomplete build folder {building}: {exc}")

    def run_plan(self, plan: list[PlanItem], approved_ambiguous: dict[str, int]) -> dict[str, Any]:
        self._workers: list[threading.Thread] = []
        try:
            return self._run_plan(plan, approved_ambiguous)
        finally:
            CONSOLE_PROGRESS.end_run()               # the dashboard goes and the cursor comes back, also after an error or Ctrl+C
            # Even interrupts outside the result loop must not outlive the logger/database.
            if any(th.is_alive() for th in self._workers):
                self.stop_event.set()
            for th in self._workers:
                if th.ident is None: continue
                while th.is_alive():
                    try: th.join(timeout=0.2)
                    except KeyboardInterrupt: self.stop_event.set()

    def _run_plan(self, plan: list[PlanItem], approved_ambiguous: dict[str, int]) -> dict[str, Any]:
        self._check_target_conflicts([item.meta for item in plan])
        start = time.monotonic()
        summary: dict[str, Any] = {"scanned": len(plan), "completed": 0, "gpu": 0, "cpu": 0, "skipped_valid": 0, "restored": 0, "skipped": 0, "skipped_unclear_folder": 0, "failed": 0, "canceled": 0, "source_changed_during_processing": 0, "source_missing": 0, "output_bytes": 0, "suspect_videos": 0, "suspect_details": [], "per_storage_group": {}}
        groups: dict[str, list[PlanItem]] = {}
        for item in plan: groups.setdefault(item.storage_group, []).append(item)
        total = len(plan)
        work = [item for item in plan if item.action != "SKIP"]
        seconds_of = {str(item.meta.path): max(0.0, float(item.meta.duration or 0.0)) for item in work}
        CONSOLE_PROGRESS.begin_run(len(work), sum(seconds_of.values()))
        counter = {"started": 0}
        counter_lock = threading.Lock()
        result_q: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        threads = self._workers
        setting = self.config.get("workers", {}).get("per_storage_group", storage_workers.AUTO)
        layout = storage_workers.layout(plan, setting)
        self.logger.log("INFO", "Videos at once per disk: " + ", ".join(f"{group} ({storage_workers.kind_label(kind)}) {count}" for group, (kind, count) in layout.items())
                        + (" (automatic)" if storage_workers.is_auto(setting) else " (set by hand)"))

        for group, items in groups.items():
            item_q: queue.Queue[PlanItem] = queue.Queue()
            for item in items: item_q.put(item)

            def worker(group_name: str = group, q: queue.Queue[PlanItem] = item_q) -> None:
                while True:
                    try: item = q.get_nowait()
                    except queue.Empty: return
                    progress_label: Optional[str] = None
                    try:
                        if self.drain_event.is_set() or self.stop_event.is_set():
                            result_q.put((group_name, {"status": "CANCELED", "path": str(item.meta.path)}))
                            continue
                        if not item.meta.path.exists():
                            self.logger.event("SOURCE_MISSING", storage_group=group_name, path=str(item.meta.path))
                            result_q.put((group_name, {"status": "SOURCE_MISSING", "path": str(item.meta.path)}))
                            continue
                        with counter_lock:
                            counter["started"] += 1
                            current = counter["started"]
                        index_text = f"[{current}/{total}]"
                        progress_label = f"{index_text} {item.meta.path.name}"
                        CONSOLE_PROGRESS.register_job(progress_label, disk_label(item.meta.path), item.meta.path.name, seconds_of.get(str(item.meta.path), 0.0))
                        res = self.process_item(item, approved_ambiguous, progress_label)
                        res["index_text"] = index_text
                        result_q.put((group_name, res))
                    except InterruptedError:
                        result_q.put((group_name, {"status": "CANCELED", "path": str(item.meta.path)}))
                    except Exception as exc:
                        result_q.put((group_name, {"status": "FAILED", "path": str(item.meta.path), "error": str(exc)}))
                        try: self.logger.failed(item.meta.path, str(exc))
                        except Exception: pass  # The FAILED result must survive a logging failure.
                    finally:
                        if progress_label:
                            CONSOLE_PROGRESS.finish(progress_label, final=True)
                        q.task_done()

            for worker_no in range(layout[group][1]):
                th = threading.Thread(target=worker, name=f"storage-{group}-{worker_no+1}", daemon=True)
                threads.append(th)

        for th in threads: th.start()
        alive = True
        while alive:
            try:
                group, result = result_q.get(timeout=0.2)
                status = result.get("status")
                g = summary["per_storage_group"].setdefault(group, {"done": 0, "failed": 0, "skipped": 0, "unavailable": 0})
                if status == "DONE":
                    CONSOLE_PROGRESS.record_done(seconds_of.get(result.get("path", ""), 0.0))
                    summary["completed"] += 1
                    summary["output_bytes"] += int(result.get("output_bytes", 0))
                    if result.get("method") == "GPU": summary["gpu"] += 1
                    else: summary["cpu"] += 1
                    g["done"] += 1
                    if result.get("suspect"):
                        summary["suspect_videos"] += 1
                        summary["suspect_details"].append(result["suspect"])
                elif status == "SKIPPED_VALID":
                    summary["skipped_valid"] += 1
                    g["skipped"] += 1
                elif status == "RESTORED":
                    summary["restored"] += 1
                    g["done"] += 1
                elif status == "SKIPPED":
                    summary["skipped"] += 1
                    g["skipped"] += 1
                    if result.get("unclear_folder"): summary["skipped_unclear_folder"] += 1
                elif status == "FAILED":
                    CONSOLE_PROGRESS.record_failed(result.get("index_text", ""), result.get("path", ""), result.get("attempts", []), result.get("error", "Processing failed."), result.get("culprit", []), seconds_of.get(result.get("path", ""), 0.0))
                    summary["failed"] += 1
                    g["failed"] += 1
                elif status == "CANCELED": summary["canceled"] += 1
                elif status == "SOURCE_CHANGED_DURING_PROCESSING":
                    CONSOLE_PROGRESS.record_failed(result.get("index_text", ""), result.get("path", ""), result.get("attempts", []), "Source changed during generation.", seconds=seconds_of.get(result.get("path", ""), 0.0))
                    summary["source_changed_during_processing"] += 1
                elif status == "SOURCE_MISSING":
                    CONSOLE_PROGRESS.record_failed(result.get("index_text", ""), result.get("path", ""), result.get("attempts", []), "Source file disappeared.", seconds=seconds_of.get(result.get("path", ""), 0.0))
                    summary["source_missing"] += 1
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                self.request_stop()
            CONSOLE_PROGRESS.tick()
            alive = any(th.is_alive() for th in threads) or not result_q.empty()
        for th in threads: th.join(timeout=1)
        CONSOLE_PROGRESS.end_run()
        summary["elapsed_seconds"] = time.monotonic() - start
        return summary
