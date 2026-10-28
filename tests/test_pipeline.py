# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""End-to-end tests with real FFmpeg: scan -> plan -> generate -> verify, and the data-safety guarantees."""
from __future__ import annotations

import contextlib
import io
import os
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import IsolatedApp, build_tools, config_with, make_video, needs_ffmpeg, new_logger, open_state

from modules import constants, platform_api
from modules.diagnostics import DiagnosticJob, DiagnosticRunner, select_hdr_candidates
from modules.generator import TrickplayGenerator
from modules.platform_windows import WindowsPlatform
from modules.models import ExtractResult
from modules.scanner import MediaScanner, sanitize_media_roots
from modules.selftest import run_self_test
from modules.tiles import building_dir_for, target_dir_for, trickplay_root_for, validate_tile_directory
from modules.verify import INVALID, MISSING, SETTINGS_CHANGED, SOURCE_CHANGED, VALID, verify_video


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


@needs_ffmpeg
class PipelineBase(IsolatedApp):
    def setUp(self) -> None:
        super().setUp()
        self.tools = build_tools()
        self.state = open_state()
        self.addCleanup(self.state.close)
        self.config = config_with()
        self.config["trickplay"].update(width=64, interval_seconds=5.0, tile_columns=3, tile_rows=3, jpeg_quality=80)
        self.config["acceleration"]["mode"] = "cpu"
        self.config["media_roots"] = [str(self.media)]

    def scan(self):
        logger = new_logger("scan")
        self.addCleanup(logger.close)
        with quiet():
            return MediaScanner(self.tools, self.state, self.config, logger).scan([self.media], show_progress=False)

    def generator(self) -> TrickplayGenerator:
        logger = new_logger("test")
        self.addCleanup(logger.close)
        return TrickplayGenerator(self.tools, self.state, self.config, logger)

    def run_all(self, gen: TrickplayGenerator | None = None, videos=None):
        gen = gen or self.generator()
        videos = videos if videos is not None else self.scan()
        with quiet():
            plan = gen.plan(videos)
            for item in plan:
                item.storage_group = platform_api.storage_group(item.meta.path)
            summary = gen.run_plan(plan, {})
        return gen, plan, summary


class ScanAndGenerate(PipelineBase):
    def test_new_videos_generate_valid_output_and_non_videos_are_ignored(self) -> None:
        make_video(self.media / "a.mp4", 25)
        make_video(self.media / "sub" / "b.mkv", 12)
        (self.media / "notes.txt").write_text("hello")
        (self.media / "broken.mkv").write_bytes(b"not a video" * 50)
        videos = self.scan()
        self.assertEqual(sorted(v.path.name for v in videos), ["a.mp4", "b.mkv"])

        gen, plan, summary = self.run_all(videos=videos)
        self.assertEqual((summary["completed"], summary["cpu"], summary["failed"]), (2, 2, 0))
        for video in videos:
            target = target_dir_for(video.path, gen.settings)
            self.assertEqual(target.name, "64 - 3x3")
            self.assertTrue(validate_tile_directory(target, gen.settings, video.duration, verify_all=True).valid)
            self.assertEqual(verify_video(video, gen.settings, self.state).status, VALID)
            self.assertFalse(building_dir_for(video.path, gen.settings).exists())
        self.assertTrue(list(target_dir_for(videos[0].path, gen.settings).glob("*.jpg")))
        self.assertFalse((self.media / "broken.trickplay").exists())

    def test_temp_folder_is_empty_after_a_run(self) -> None:
        make_video(self.media / "a.mp4", 25)
        make_video(self.media / "b.mp4", 12)
        self.run_all()
        self.assertEqual([p.name for p in constants.DIRS["temp"].iterdir()], [], "no job_ or run_ folders may be left behind")

    def test_second_run_skips_valid_and_changes_nothing(self) -> None:
        make_video(self.media / "a.mp4", 25)
        gen, _, first = self.run_all()
        target = target_dir_for(self.media / "a.mp4", gen.settings)
        before = {p.name: p.stat().st_mtime_ns for p in target.iterdir()}
        _, plan, second = self.run_all()
        self.assertEqual([i.action for i in plan], ["SKIP"])
        self.assertEqual((second["skipped_valid"], second["completed"]), (1, 0))
        self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in target.iterdir()})

    def videos_at_once(self, kind: str, setting: object) -> int:
        """The most videos that were being processed at the same time on one disk of this kind under this setting."""
        import threading
        for name in "abcd": make_video(self.media / f"{name}.mp4", 5)
        self.config["workers"]["per_storage_group"] = setting
        gen = self.generator()
        state = {"now": 0, "most": 0}
        lock = threading.Lock()

        def slow(item, approved, label=None):
            with lock:
                state["now"] += 1
                state["most"] = max(state["most"], state["now"])
            time.sleep(0.25)
            with lock: state["now"] -= 1
            return {"status": "DONE", "path": str(item.meta.path), "method": "CPU"}
        gen.process_item = slow                                  # type: ignore[method-assign]
        with mock.patch("modules.storage.storage_kind", return_value=kind):
            self.run_all(gen)
        return state["most"]

    def test_a_hard_disk_gets_one_video_at_once_and_an_ssd_two(self) -> None:
        self.assertEqual(self.videos_at_once("hdd", "auto"), 1)

    def test_an_ssd_gets_two_videos_at_once(self) -> None:
        self.assertEqual(self.videos_at_once("ssd", "auto"), 2)

    def test_a_number_set_by_hand_is_used_on_every_kind_of_disk(self) -> None:
        self.assertEqual(self.videos_at_once("hdd", 3), 3)

    def test_scanning_uses_8_probe_workers_per_storage_group(self) -> None:
        import concurrent.futures
        make_video(self.media / "a.mp4", 12)
        made: list[int] = []
        real = concurrent.futures.ThreadPoolExecutor

        def recording(*args, **kwargs):
            made.append(kwargs.get("max_workers"))
            return real(*args, **kwargs)
        with mock.patch("modules.scanner.concurrent.futures.ThreadPoolExecutor", recording):
            self.scan()
        self.assertEqual(made, [8])

    def test_excluded_folders_are_not_scanned_at_any_depth_and_capitals_do_not_matter(self) -> None:
        make_video(self.media / "keep.mp4", 12)
        make_video(self.media / "Extras" / "skip.mp4", 12)
        make_video(self.media / "Shows" / "Show" / "extras" / "deep.mp4", 12)
        make_video(self.media / "Shows" / "Show" / "episode.mp4", 12)
        self.config["excluded_folders"] = ["EXTRAS"]
        self.assertEqual(sorted(v.path.name for v in self.scan()), ["episode.mp4", "keep.mp4"])
        self.config["excluded_folders"] = []
        self.assertEqual(len(self.scan()), 4)

    def test_the_system_folders_of_a_drive_are_skipped_without_a_warning(self) -> None:
        make_video(self.media / "keep.mp4", 12)
        make_video(self.media / "$RECYCLE.BIN" / "S-1-5-21" / "gone.mp4", 12)
        make_video(self.media / "System Volume Information" / "x.mp4", 12)
        make_video(self.media / "Shows" / "$Recycle.Bin" / "deep.mp4", 12)
        self.config["excluded_folders"] = []
        lines: list[str] = []
        with mock.patch("modules.logger.RunLogger.log", lambda self_, level, message: lines.append(f"{level} {message}")):
            self.assertEqual([v.path.name for v in self.scan()], ["keep.mp4"])
        self.assertFalse([line for line in lines if "Could not scan" in line])

    def test_scanner_reuses_probe_cache(self) -> None:
        make_video(self.media / "a.mp4", 12)
        first = self.scan()
        with mock.patch.object(type(self.tools), "probe", side_effect=AssertionError("probe must not run again")):
            again = self.scan()
        self.assertEqual([v.path for v in first], [v.path for v in again])

    def test_damaged_output_is_rebuilt_and_source_change_regenerates(self) -> None:
        video = make_video(self.media / "a.mp4", 25)
        gen, *_ = self.run_all()
        target = target_dir_for(video, gen.settings)
        victim = sorted(target.glob("*.jpg"))[-1]
        victim.write_bytes(b"")
        meta = self.scan()[0]
        self.assertEqual(verify_video(meta, gen.settings, self.state).status, INVALID)
        _, plan, summary = self.run_all(videos=[meta])
        self.assertEqual(plan[0].reason, "INVALID_FINAL_REBUILD")
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(verify_video(meta, gen.settings, self.state).status, VALID)
        self.assertFalse(list(trickplay_root_for(video).glob("*.__old__.*")), "replacement backup must be cleaned after success")

        make_video(video, 30)  # source replaced by different content
        os.utime(video, None)
        changed = self.scan()[0]
        self.assertEqual(verify_video(changed, gen.settings, self.state).status, SOURCE_CHANGED)
        _, plan, summary = self.run_all(videos=[changed])
        self.assertEqual((plan[0].reason, summary["completed"]), ("SOURCE_CHANGED", 1))

    # --- Generate puts a valid previous output back instead of building it again ---------------------------------------
    def interrupted(self, damaged: bool = False):
        video = make_video(self.media / "a.mp4", 25)
        gen, *_ = self.run_all()
        target = target_dir_for(video, gen.settings)
        old = target.parent / f"{target.name}.__old__.2026-10-26_220000.abc123"
        if damaged:
            import shutil
            shutil.copytree(target, old)
            sorted(target.glob("*.jpg"))[-1].write_bytes(b"")
        else:
            target.rename(old)
        return video, gen, target, old

    def test_generate_restores_the_previous_output_instead_of_building_it_again(self) -> None:
        video, gen, target, old = self.interrupted()
        names = sorted(p.name for p in old.iterdir())
        gen, plan, summary = self.run_all(videos=self.scan())
        self.assertEqual((plan[0].action, plan[0].reason), ("RESTORE", "RESTORE_PREVIOUS"))
        self.assertEqual((summary["restored"], summary["completed"], summary["failed"]), (1, 0, 0))
        self.assertFalse(old.exists())
        self.assertEqual(sorted(p.name for p in target.iterdir()), names)
        self.assertEqual(verify_video(self.scan()[0], gen.settings, self.state).status, VALID)
        self.assertIn("RESTORED", gen.logger.events_path.read_text(encoding="utf-8"))

    def test_generate_restores_over_a_damaged_output_and_keeps_the_damaged_copy_aside(self) -> None:
        video, gen, target, old = self.interrupted(damaged=True)
        _, plan, summary = self.run_all(videos=self.scan())
        self.assertEqual((plan[0].action, summary["restored"], summary["completed"]), ("RESTORE", 1, 0))
        self.assertEqual(verify_video(self.scan()[0], gen.settings, self.state).status, VALID)
        self.assertEqual(len([p for p in target.parent.iterdir() if ".__failed_finalize__." in p.name]), 1)

    def test_generate_builds_normally_when_the_old_copy_is_not_valid(self) -> None:
        video, gen, target, old = self.interrupted()
        sorted(old.glob("*.jpg"))[-1].write_bytes(b"")
        _, plan, summary = self.run_all(videos=self.scan())
        self.assertEqual((plan[0].action, plan[0].reason), ("GENERATE", "MISSING"))
        self.assertEqual((summary["completed"], summary["restored"]), (1, 0))

    def test_generate_builds_when_the_restore_fails_after_planning(self) -> None:
        video, gen, target, old = self.interrupted()
        meta = self.scan()[0]
        with mock.patch("modules.generator.restore_previous", return_value="disk says no"):
            _, plan, summary = self.run_all(videos=[meta])
        self.assertEqual(plan[0].action, "RESTORE")
        self.assertEqual((summary["completed"], summary["restored"], summary["failed"]), (1, 0, 0))
        self.assertEqual(verify_video(meta, gen.settings, self.state).status, VALID)

    def test_a_target_with_unexpected_files_is_never_restored_over(self) -> None:
        video, gen, target, old = self.interrupted(damaged=True)
        (target / "notes.txt").write_text("mine")
        _, plan, _ = self.run_all(videos=self.scan())
        self.assertEqual(plan[0].action, "GENERATE")
        self.assertTrue((target / "notes.txt").exists())

    def test_settings_change_regenerates_and_other_width_policies(self) -> None:
        video = make_video(self.media / "a.mp4", 25)
        self.run_all()
        old_dir = trickplay_root_for(video) / "64 - 3x3"
        self.config["trickplay"]["width"] = 96
        self.config["trickplay"]["other_width_policy"] = "keep_add"
        gen, plan, summary = self.run_all()
        self.assertEqual(summary["completed"], 1)
        self.assertTrue(old_dir.exists() and (trickplay_root_for(video) / "96 - 3x3").exists())

        self.config["trickplay"]["width"] = 128
        self.config["trickplay"]["other_width_policy"] = "replace"
        _, _, summary = self.run_all()
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(sorted(p.name for p in trickplay_root_for(video).iterdir()), ["128 - 3x3"])

    def test_ambiguous_old_width_directory_is_preserved(self) -> None:
        video = make_video(self.media / "a.mp4", 25)
        self.run_all()
        old = trickplay_root_for(video) / "64 - 3x3"
        (old / "user_notes.txt").write_text("mine")
        self.config["trickplay"]["width"] = 96
        self.config["trickplay"]["other_width_policy"] = "replace"
        self.run_all()
        self.assertTrue((old / "user_notes.txt").exists(), "unknown files must never be deleted without approval")

    def test_stale_building_directory_is_cleaned(self) -> None:
        video = make_video(self.media / "a.mp4", 25)
        meta = self.scan()[0]
        gen = self.generator()
        stale = building_dir_for(video, gen.settings)
        stale.mkdir(parents=True)
        (stale / "0.jpg").write_bytes(b"partial")
        _, _, summary = self.run_all(gen, [meta])
        self.assertEqual(summary["completed"], 1)
        self.assertFalse(stale.exists())

    def test_target_conflicts_are_refused(self) -> None:
        make_video(self.media / "movie.mp4", 12)
        make_video(self.media / "movie.mkv", 12)
        videos = self.scan()
        with self.assertRaises(ValueError):
            self.generator().plan(videos)
        self.assertFalse((self.media / "movie.trickplay").exists())


class DataSafety(PipelineBase):
    def test_failed_processing_never_destroys_existing_valid_output(self) -> None:
        video = make_video(self.media / "a.mp4", 25)
        gen, *_ = self.run_all()
        target = target_dir_for(video, gen.settings)
        snapshot = {p.name: p.read_bytes() for p in target.iterdir()}

        self.config["trickplay"]["same_width_policy"] = "rebuild_all"
        gen2 = self.generator()
        failing = ExtractResult(False, "CPU", None, 0.1, 0, "simulated ffmpeg failure")
        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", return_value=failing):
            _, _, summary = self.run_all(gen2)
        self.assertEqual((summary["failed"], summary["completed"]), (1, 0))
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in target.iterdir()})
        self.assertFalse(building_dir_for(video, gen2.settings).exists())

    def test_failed_rollback_is_reported_with_the_location_of_the_preserved_output(self) -> None:
        gen = self.generator()
        target = self.root / "t" / "64 - 3x3"
        backup = self.root / "t" / "64 - 3x3.__old__.x"
        backup.mkdir(parents=True)
        (backup / "0.jpg").write_bytes(b"previous valid data")
        with mock.patch("os.replace", side_effect=OSError("disk gone")), quiet():
            gen._restore_backup(backup, target)
        self.assertTrue((backup / "0.jpg").exists(), "the previous output must never be lost")
        text = gen.logger.path.read_text(encoding="utf-8")
        self.assertIn("[ERROR]", text)
        self.assertIn(str(backup), text)
        self.assertIn("ROLLBACK_FAILED", gen.logger.events_path.read_text(encoding="utf-8"))

    def test_successful_rollback_restores_the_previous_output(self) -> None:
        gen = self.generator()
        target = self.root / "t" / "64 - 3x3"
        backup = self.root / "t" / "64 - 3x3.__old__.x"
        backup.mkdir(parents=True)
        (backup / "0.jpg").write_bytes(b"previous valid data")
        gen._restore_backup(backup, target)
        self.assertEqual((target / "0.jpg").read_bytes(), b"previous valid data")
        gen._restore_backup(None, target)   # nothing to restore: must not raise
        gen._restore_backup(backup, target)  # backup gone and target present: must not raise

    @staticmethod
    def locked(name: str = "in use") -> PermissionError:
        exc = PermissionError(13, "The process cannot access the file because it is being used by another process", name)
        exc.winerror = 32  # type: ignore[attr-defined]
        return exc

    def fast_retries(self, gen: TrickplayGenerator) -> None:
        gen.LOCK_RETRY_DELAYS = (0.0, 0.0, 0.0)

    def setUp(self) -> None:
        super().setUp()
        # Locked folders are a Windows thing: run these tests with the Windows rule for what counts as a lock.
        patcher = mock.patch.object(platform_api, "is_transient_lock_error", WindowsPlatform().is_transient_lock_error)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_lock_errors_are_recognised_and_other_errors_are_not(self) -> None:
        windows = WindowsPlatform()
        self.assertTrue(windows.is_transient_lock_error(self.locked()))
        self.assertTrue(windows.is_transient_lock_error(PermissionError("access denied")))
        self.assertFalse(windows.is_transient_lock_error(OSError("disk gone")))
        self.assertFalse(windows.is_transient_lock_error(FileNotFoundError("gone")))
        self.assertFalse(windows.is_transient_lock_error(ValueError("x")))
        from modules.platform_linux import LinuxPlatform
        self.assertFalse(LinuxPlatform().is_transient_lock_error(self.locked()))

    def test_replace_waits_out_a_short_lock(self) -> None:
        gen = self.generator()
        self.fast_retries(gen)
        src, dst = self.root / "a", self.root / "b"
        src.mkdir()
        real = os.replace
        calls: list[int] = []

        def flaky(a, b):
            calls.append(1)
            if len(calls) <= 2: raise self.locked()
            return real(a, b)

        with mock.patch("os.replace", flaky):
            gen._replace_with_retry(src, dst)
        self.assertEqual(len(calls), 3)
        self.assertTrue(dst.exists() and not src.exists())
        self.assertIn("retry 1/3", gen.logger.path.read_text(encoding="utf-8"))

    def test_replace_gives_up_after_the_last_retry_and_never_retries_other_errors(self) -> None:
        gen = self.generator()
        self.fast_retries(gen)
        src, dst = self.root / "a", self.root / "b"
        with mock.patch("os.replace", side_effect=self.locked()) as always_locked:
            with self.assertRaises(PermissionError): gen._replace_with_retry(src, dst)
        self.assertEqual(always_locked.call_count, 4)
        with mock.patch("os.replace", side_effect=OSError("disk gone")) as other:
            with self.assertRaises(OSError): gen._replace_with_retry(src, dst)
        self.assertEqual(other.call_count, 1)

    def test_replace_wait_can_be_canceled(self) -> None:
        gen = self.generator()
        gen.LOCK_RETRY_DELAYS = (30.0,)
        gen.stop_event.set()
        with mock.patch("os.replace", side_effect=self.locked()) as locked, self.assertRaises(InterruptedError):
            started = time.monotonic()
            try: gen._replace_with_retry(self.root / "a", self.root / "b")
            finally: self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(locked.call_count, 1)

    @needs_ffmpeg
    def test_generate_replaces_output_once_a_short_lock_is_gone(self) -> None:
        video = make_video(self.media / "a.mp4", 25)
        gen, *_ = self.run_all()
        target = target_dir_for(video, gen.settings)
        self.config["trickplay"]["same_width_policy"] = "rebuild_all"
        gen2 = self.generator()
        self.fast_retries(gen2)
        real = os.replace
        calls: list[int] = []

        def flaky(a, b):
            if Path(a) == target and len(calls) < 2:
                calls.append(1)
                raise self.locked()
            return real(a, b)

        with mock.patch("os.replace", flaky):
            _, _, summary = self.run_all(gen2)
        self.assertEqual((summary["completed"], summary["failed"]), (1, 0))
        self.assertEqual(len(calls), 2)
        self.assertEqual(verify_video(self.scan()[0], gen2.settings, self.state).status, VALID)
        self.assertEqual([p.name for p in target.parent.iterdir()], [target.name])

    @needs_ffmpeg
    def test_generate_with_a_lasting_lock_fails_cleanly_and_keeps_the_old_output(self) -> None:
        video = make_video(self.media / "a.mp4", 25)
        gen, *_ = self.run_all()
        target = target_dir_for(video, gen.settings)
        snapshot = {p.name: p.read_bytes() for p in target.iterdir()}
        self.config["trickplay"]["same_width_policy"] = "rebuild_all"
        gen2 = self.generator()
        self.fast_retries(gen2)
        real = os.replace

        def locked_target(a, b):
            if Path(a) == target: raise self.locked(str(a))
            return real(a, b)

        results: list[dict] = []
        real_process = TrickplayGenerator.process_item

        def capture(self_, *args, **kwargs):
            result = real_process(self_, *args, **kwargs)
            results.append(result)
            return result

        with mock.patch("os.replace", locked_target), mock.patch.object(TrickplayGenerator, "process_item", capture):
            _, _, summary = self.run_all(gen2)
        self.assertEqual((summary["failed"], summary["completed"]), (1, 0))
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in target.iterdir()})
        self.assertEqual([p.name for p in target.parent.iterdir()], [target.name], "no building or backup leftovers")
        self.assertEqual(results[0]["culprit"], [str(target)])
        self.assertIn("being used by another process", results[0]["error"])

    def test_failed_line_names_the_video_and_the_file_that_caused_it_not_unknown(self) -> None:
        from modules.console_progress import ConsoleProgress
        progress = ConsoleProgress()
        progress._enabled = False
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            progress.record_failed("[12/35]", "D:/v/a.mkv", [], "PermissionError: [WinError 32] being used by another process", ["D:/v/a.trickplay/64 - 3x3"])
            progress.record_failed("[13/35]", "D:/v/b.mkv", ["GPU/cuda", "CPU"], "ffmpeg failed")
        text = out.getvalue()
        self.assertNotIn("unknown", text)
        self.assertIn("FAILED [12/35] D:/v/a.mkv", text)
        self.assertIn("Concerns: D:/v/a.trickplay/64 - 3x3", text)
        self.assertIn("in use by another program", text)
        self.assertIn("Tried: GPU/cuda -> CPU", text)
        self.assertEqual(text.count("Tried:"), 1)

    def test_gpu_only_mode_without_backends_fails_without_touching_output(self) -> None:
        video = make_video(self.media / "a.mp4", 25)
        self.run_all()
        target = target_dir_for(video, self.generator().settings)
        snapshot = sorted(p.name for p in target.iterdir())
        self.config["acceleration"]["mode"] = "gpu"
        self.config["acceleration"]["preferred_backend"] = "none-such"
        self.config["trickplay"]["same_width_policy"] = "rebuild_all"
        with mock.patch.object(type(self.tools), "backends", return_value=[]):
            _, _, summary = self.run_all()
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(snapshot, sorted(p.name for p in target.iterdir()))
        self.assertEqual(verify_video(self.scan()[0], self.generator().settings, self.state).status, VALID)

    def test_source_modified_during_processing_is_not_finalized(self) -> None:
        video = make_video(self.media / "a.mp4", 25)
        meta = self.scan()[0]
        gen = self.generator()
        real = TrickplayGenerator._run_ffmpeg_extract

        def touching(self_, *args, **kwargs):
            result = real(self_, *args, **kwargs)
            future = time.time() + 5
            os.utime(video, (future, future))
            return result

        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", touching):
            _, _, summary = self.run_all(gen, [meta])
        self.assertEqual(summary["source_changed_during_processing"], 1)
        self.assertFalse(target_dir_for(video, gen.settings).exists())
        self.assertFalse(building_dir_for(video, gen.settings).exists())

    def test_missing_source_is_reported_not_crashed(self) -> None:
        video = make_video(self.media / "a.mp4", 12)
        meta = self.scan()[0]
        video.unlink()
        _, _, summary = self.run_all(videos=[meta])
        self.assertEqual((summary["source_missing"], summary["completed"]), (1, 0))

    def test_verify_reports_missing_for_never_processed_video(self) -> None:
        make_video(self.media / "a.mp4", 12)
        meta = self.scan()[0]
        self.assertEqual(verify_video(meta, self.generator().settings, self.state).status, MISSING)


class VerifyThenRegenerate(PipelineBase):
    """Verify's offer to fix what it found, through the real generator."""

    def setUp(self) -> None:
        super().setUp()
        from modules.ui import Application
        self.app = Application()
        self.addCleanup(self.app.state.close)
        self.app.config = self.config
        self.app.tools = self.tools
        self.app.state.open()

    def regenerate(self, videos, answers):
        from modules.verify import verify_videos
        settings = self.generator().settings
        _, problems = verify_videos(videos, settings, self.app.state)
        with mock.patch("builtins.input", side_effect=list(answers)), quiet(), \
                mock.patch.object(type(self.app), "_preflight", side_effect=lambda v, plan, g: (True, {})):
            self.app._offer_regeneration(problems, videos)
        return settings

    def generate_all(self, videos) -> None:
        logger = new_logger("generate")
        self.addCleanup(logger.close)
        with mock.patch.object(type(self.app), "_preflight", side_effect=lambda v, plan, g: (True, {})), quiet():
            self.assertTrue(self.app._plan_and_run(videos, logger))

    def test_missing_and_damaged_outputs_are_regenerated_after_confirmation(self) -> None:
        from modules.verify import verify_video
        make_video(self.media / "a.mp4", 25)
        make_video(self.media / "b.mp4", 25)
        make_video(self.media / "c.mp4", 25)
        videos = self.scan()
        self.generate_all(videos)
        settings = self.generator().settings
        import shutil
        shutil.rmtree(target_dir_for(videos[0].path, settings))                       # a: missing
        sprite = sorted(target_dir_for(videos[1].path, settings).glob("*.jpg"))[0]
        sprite.write_bytes(b"")                                                       # b: damaged
        c_before = {p.name: p.stat().st_mtime_ns for p in target_dir_for(videos[2].path, settings).iterdir()}

        self.regenerate(videos, ["n"])                                                # declined: nothing changes
        self.assertEqual(verify_video(videos[0], settings, self.app.state).status, MISSING)
        self.assertEqual(verify_video(videos[1], settings, self.app.state).status, INVALID)

        self.regenerate(videos, ["y"])                                                # confirmed
        for video in videos:
            self.assertEqual(verify_video(video, settings, self.app.state).status, VALID, video.path.name)
        self.assertEqual(c_before, {p.name: p.stat().st_mtime_ns for p in target_dir_for(videos[2].path, settings).iterdir()},
                         "the untouched video must not be rebuilt")

    def test_outdated_settings_are_only_regenerated_when_asked_separately(self) -> None:
        from modules.verify import verify_video
        make_video(self.media / "a.mp4", 25)
        videos = self.scan()
        self.generate_all(videos)
        self.config["trickplay"]["interval_seconds"] = 10.0                           # same width/grid, different interval
        settings = self.generator().settings
        self.assertEqual(verify_video(videos[0], settings, self.app.state).status, SETTINGS_CHANGED)
        self.regenerate(videos, ["n"])                                                # only the optional question is asked
        self.assertEqual(verify_video(videos[0], settings, self.app.state).status, SETTINGS_CHANGED)
        self.regenerate(videos, ["y"])
        self.assertEqual(verify_video(videos[0], settings, self.app.state).status, VALID)

    def test_nothing_to_fix_asks_nothing(self) -> None:
        make_video(self.media / "a.mp4", 25)
        videos = self.scan()
        self.generate_all(videos)
        self.regenerate(videos, [])                                                   # any prompt would raise StopIteration


class QuickCheckOfOwnOutput(PipelineBase):
    """Generate trusts an output it recorded when names and sizes still match; Verify and any doubt read every picture."""

    def make(self):
        video = make_video(self.media / "a.mp4", 25)
        gen, *_ = self.run_all()
        return video, gen, target_dir_for(video, gen.settings)

    def plan_one(self, gen):
        with quiet():
            return gen.plan(self.scan())[0]

    def test_a_recorded_output_is_skipped_without_reading_a_single_picture(self) -> None:
        _, gen, _ = self.make()
        with mock.patch("PIL.Image.open", side_effect=AssertionError("no picture may be read")):
            item = self.plan_one(self.generator())
        self.assertEqual((item.action, item.reason), ("SKIP", "SKIPPED_VALID"))

    def test_a_tile_that_changed_size_is_found_and_rebuilt(self) -> None:
        _, gen, target = self.make()
        tile = sorted(target.glob("*.jpg"))[0]
        tile.write_bytes(b"not a picture")
        item = self.plan_one(self.generator())
        self.assertEqual((item.action, item.reason), ("GENERATE", "INVALID_FINAL_REBUILD"))

    def test_a_missing_or_extra_file_is_found(self) -> None:
        _, gen, target = self.make()
        (target / "notes.txt").write_text("x")
        self.assertEqual(self.plan_one(self.generator()).reason, "INVALID_FINAL_REBUILD")
        (target / "notes.txt").unlink()
        sorted(target.glob("*.jpg"))[-1].unlink()
        self.assertEqual(self.plan_one(self.generator()).reason, "INVALID_FINAL_REBUILD")

    def test_damage_that_keeps_the_size_is_left_to_verify(self) -> None:
        _, gen, target = self.make()
        tile = sorted(target.glob("*.jpg"))[0]
        tile.write_bytes(b"\0" * tile.stat().st_size)
        self.assertEqual(self.plan_one(self.generator()).action, "SKIP", "Generate does not read the pictures again")
        row = verify_video(self.scan()[0], gen.settings, self.state)
        self.assertEqual(row.status, INVALID, "Verify reads every picture")

    def test_an_output_without_a_record_is_checked_in_full(self) -> None:
        _, gen, target = self.make()
        self.state.conn.execute("DELETE FROM outputs")
        self.state.conn.commit()
        sorted(target.glob("*.jpg"))[0].write_bytes(b"\0" * 100)
        self.assertEqual(self.plan_one(self.generator()).reason, "INVALID_FINAL_REBUILD")


class StalledFfmpeg(PipelineBase):
    """FFmpeg that stops making progress (a drive that no longer answers) is stopped, once, and the video is reported as failed."""

    @staticmethod
    def silent_process():
        import subprocess
        import sys
        real = subprocess.Popen

        def fake(cmd, *args, **kwargs):
            return real([sys.executable, "-c", "import time; time.sleep(60)"], *args, **kwargs)
        return mock.patch("modules.extraction.subprocess.Popen", fake)

    def test_a_silent_ffmpeg_is_killed_and_the_video_fails_without_touching_the_output(self) -> None:
        video = make_video(self.media / "a.mp4", 25)
        gen = self.generator()
        gen.STALL_SECONDS = 0.5
        videos = self.scan()                      # (scanning runs ffprobe through the same Popen: not under the fake)
        started = time.monotonic()
        with self.silent_process():
            _, _, summary = self.run_all(gen, videos)
        self.assertLess(time.monotonic() - started, 30)
        self.assertEqual((summary["failed"], summary["completed"]), (1, 0))
        self.assertFalse(target_dir_for(video, gen.settings).exists())
        self.assertFalse(building_dir_for(video, gen.settings).exists())
        self.assertIn("no progress", gen.logger.events_path.read_text(encoding="utf-8"))

    def test_the_result_says_stalled(self) -> None:
        make_video(self.media / "a.mp4", 25)
        gen = self.generator()
        gen.STALL_SECONDS = 0.5
        meta = self.scan()[0]
        with self.silent_process():
            result = gen._run_ffmpeg_extract(meta, self.root / "frames", "CPU", None)
        self.assertTrue(result.stalled and not result.success and not result.canceled)

    def test_a_stall_is_neither_retried_nor_blamed_on_the_gpu_nor_tried_on_the_cpu(self) -> None:
        make_video(self.media / "a.mp4", 25)
        gen = self.generator()
        meta = self.scan()[0]
        stalled = ExtractResult(False, "GPU", "cuda", 1.0, 0, "FFmpeg made no progress", stalled=True)
        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", return_value=stalled) as run, \
                mock.patch.object(TrickplayGenerator, "_preferred_backends", return_value=["cuda"]), \
                mock.patch.object(type(self.state), "record_backend_failure") as record:
            result = gen._choose_method_and_extract(meta, self.root / "frames", "x")
        self.assertTrue(result.stalled)
        self.assertEqual(run.call_count, 1, "no second GPU try and no CPU fallback")
        record.assert_not_called()


class InterruptedScan(IsolatedApp):
    """Ctrl+C during the probing of a big library must not leave the whole queue of probes running behind the menu."""

    def test_the_probes_that_have_not_started_are_dropped(self) -> None:
        import threading
        import time
        from unittest import mock

        class SlowTools:
            def __init__(self) -> None:
                self.probed = 0
                self.lock = threading.Lock()

            def probe(self, path: Path) -> None:
                with self.lock: self.probed += 1
                time.sleep(0.05)

        for i in range(80): (self.media / f"{i:02d}.mkv").write_bytes(b"x")
        state = open_state()
        self.addCleanup(state.close)
        tools = SlowTools()
        logger = new_logger("interrupted")
        self.addCleanup(logger.close)
        scanner = MediaScanner(tools, state, config_with(), logger)       # type: ignore[arg-type]
        with mock.patch("modules.scanner.concurrent.futures.as_completed", side_effect=KeyboardInterrupt), quiet(), self.assertRaises(KeyboardInterrupt):
            scanner.scan([self.media], show_progress=False)
        self.assertLess(tools.probed, 40, "only what was already running finished")
        self.assertEqual([t.name for t in threading.enumerate() if t.name.startswith("ffprobe")], [], "no probe thread is left behind")


class UnreadableVideos(PipelineBase):
    """A file with a video extension that ffprobe cannot read must not vanish without a word."""

    def broken(self, name: str = "broken.mkv") -> Path:
        path = self.media / name
        path.write_bytes(b"this is not a video" * 100)
        return path

    def scanner(self) -> MediaScanner:
        logger = new_logger("unreadable")
        self.addCleanup(logger.close)
        return MediaScanner(self.tools, self.state, self.config, logger)

    def test_a_video_that_cannot_be_read_is_reported_with_the_reason(self) -> None:
        make_video(self.media / "good.mp4", 12)
        broken = self.broken()
        scanner = self.scanner()
        with quiet():
            videos = scanner.scan([self.media], show_progress=False)
        self.assertEqual([v.path.name for v in videos], ["good.mp4"])
        [(path, reason)] = scanner.unreadable_videos
        self.assertEqual(path, broken)
        self.assertIn("ffprobe", reason)

    def test_files_that_are_not_videos_are_not_reported(self) -> None:
        make_video(self.media / "good.mp4", 12)
        (self.media / "notes.weirdext").write_bytes(b"x")          # unknown extension: could be a video, is not
        (self.media / "movie.nfo").write_text("x")                  # known as no video
        scanner = self.scanner()
        with quiet():
            scanner.scan([self.media], show_progress=False)
        self.assertEqual(scanner.unreadable_videos, [])

    def test_a_probe_that_times_out_is_reported(self) -> None:
        import subprocess
        make_video(self.media / "good.mp4", 12)
        scanner = self.scanner()
        with mock.patch("modules.ffmpeg_tools.subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 60)), quiet():
            scanner.scan([self.media], show_progress=False)
        [(_, reason)] = scanner.unreadable_videos
        self.assertIn("did not answer", reason)

    def test_the_list_is_cleared_for_every_scan(self) -> None:
        broken = self.broken()
        scanner = self.scanner()
        with quiet():
            scanner.scan([self.media], show_progress=False)
            self.assertEqual(len(scanner.unreadable_videos), 1)
            broken.unlink()
            scanner.scan([self.media], show_progress=False)
        self.assertEqual(scanner.unreadable_videos, [])


class OrphanFolders(PipelineBase):
    """`.trickplay` folders whose video is gone are found while scanning, and removed only when that cannot hurt anything."""

    def scanner(self) -> MediaScanner:
        logger = new_logger("orphans")
        self.addCleanup(logger.close)
        return MediaScanner(self.tools, self.state, self.config, logger)

    def sprites(self, folder: Path, name: str = "64 - 3x3", count: int = 2) -> Path:
        out = folder / name
        out.mkdir(parents=True)
        for i in range(count): (out / f"{i}.jpg").write_bytes(b"x")
        return folder

    def found(self) -> list[str]:
        scanner = self.scanner()
        with quiet():
            scanner.scan([self.media], show_progress=False)
        return [p.name for p in scanner.orphan_trickplay_dirs()]

    def test_only_folders_without_a_video_next_to_them_are_found(self) -> None:
        make_video(self.media / "kept.mp4", 12)
        self.sprites(self.media / "kept.trickplay")
        self.sprites(self.media / "gone.trickplay")
        sub = self.media / "Shows" / "Show"
        sub.mkdir(parents=True)
        self.sprites(sub / "S01E01.trickplay")
        (sub / "S01E02.mkv").write_bytes(b"x")                       # not a playable video, but a video-like file: never an orphan
        self.sprites(sub / "S01E02.trickplay")
        self.assertEqual(sorted(self.found()), ["S01E01.trickplay", "gone.trickplay"])

    def test_the_name_match_ignores_case_and_any_video_like_extension(self) -> None:
        (self.media / "Movie.MKV").write_bytes(b"x")
        self.sprites(self.media / "movie.trickplay")
        (self.media / "Other.weirdext").write_bytes(b"x")          # unknown extension: the scanner would probe it, so it may be a video
        self.sprites(self.media / "Other.trickplay")
        (self.media / "Third.srt").write_bytes(b"x")                # a subtitle is not a video
        self.sprites(self.media / "Third.trickplay")
        self.assertEqual(self.found(), ["Third.trickplay"])

    def test_a_folder_that_could_not_be_listed_never_produces_orphans(self) -> None:
        self.sprites(self.media / "gone.trickplay")
        scanner = self.scanner()
        with quiet():
            scanner.scan([self.media], show_progress=False)
        self.assertEqual(len(scanner.orphan_trickplay_dirs()), 1)
        scanner._unreadable_dirs.add(str(self.media))
        self.assertEqual(scanner.orphan_trickplay_dirs(), [])

    def test_excluded_folders_are_not_looked_at(self) -> None:
        self.config["excluded_folders"] = ["extras"]
        self.sprites(self.media / "extras" / "gone.trickplay")
        self.assertEqual(self.found(), [])

    def test_removal_rechecks_everything(self) -> None:
        from modules.verify import assess_orphan, remove_orphan
        folder = self.sprites(self.media / "gone.trickplay")
        (folder / "64 - 3x3.__old__.2026-10-26_1.abc123").mkdir()
        item = assess_orphan(folder)
        self.assertTrue(item.safe, item.reason)
        (self.media / "gone.mkv").write_bytes(b"x")                 # the video came back after the check
        self.assertIn("video", remove_orphan(item) or "")
        self.assertTrue(folder.exists())
        (self.media / "gone.mkv").unlink()
        (folder / "notes.txt").write_text("mine")                   # something of the user's appeared
        self.assertIn("not trickplay output", remove_orphan(item) or "")
        self.assertTrue((folder / "notes.txt").exists())
        (folder / "notes.txt").unlink()
        self.assertIsNone(remove_orphan(item))
        self.assertFalse(folder.exists())

    def test_folders_with_foreign_content_or_wrong_names_are_not_safe(self) -> None:
        from modules.verify import assess_orphan, remove_orphan
        odd = self.sprites(self.media / "a.trickplay", name="thumbs")             # not a width folder
        self.assertFalse(assess_orphan(odd).safe)
        other = self.sprites(self.media / "b.trickplay")
        (other / "64 - 3x3" / "readme.txt").write_text("x")                        # something besides sprites inside
        self.assertFalse(assess_orphan(other).safe)
        self.assertIsNotNone(remove_orphan(assess_orphan(other)))
        self.assertTrue(other.exists())
        empty = self.media / "c.trickplay"
        empty.mkdir()
        self.assertTrue(assess_orphan(empty).safe)

    def test_a_link_is_never_followed_or_removed(self) -> None:
        from tests.helpers import can_symlink
        if not can_symlink(): self.skipTest("symbolic links are not available")
        real = self.sprites(self.root / "elsewhere")
        link = self.media / "gone.trickplay"
        link.symlink_to(real, target_is_directory=True)
        self.assertEqual(self.found(), [])
        self.assertTrue((real / "64 - 3x3" / "0.jpg").exists())


class Roots(IsolatedApp):
    def test_sanitize_media_roots(self) -> None:
        a, nested, other = self.root / "a", self.root / "a" / "inner", self.root / "b"
        for p in (a, nested, other):
            p.mkdir(parents=True, exist_ok=True)
        inside_app = constants.APP_ROOT / "videos"
        inside_app.mkdir(parents=True, exist_ok=True)
        with quiet():
            roots = sanitize_media_roots([str(nested), str(a), str(a), f'"{other}"', "", str(inside_app)])
        self.assertEqual(sorted(roots), sorted([a, other]))


    def test_a_tilde_root_is_the_home_folder_and_an_unreadable_shortcut_is_skipped_with_a_message(self) -> None:
        home = self.root / "home"
        (home / "Movies").mkdir(parents=True)
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"HOME": str(home), "USERPROFILE": str(home)}), contextlib.redirect_stdout(out):
            self.assertEqual(sanitize_media_roots(["~/Movies"]), [home / "Movies"])
            with mock.patch.object(platform_api, "resolve_media_path", side_effect=ValueError("no")):
                self.assertEqual(sanitize_media_roots(["C:\\x\\Movies.lnk"]), [])
        self.assertIn("C:\\x\\Movies.lnk", out.getvalue())

    def test_a_root_that_holds_the_program_is_kept_and_the_program_folder_is_skipped_by_the_scan(self) -> None:
        app = self.root / "drive" / "program"
        for name in ("drive/media", "drive/program/temp"): (self.root / name).mkdir(parents=True, exist_ok=True)
        (self.root / "drive" / "media" / "a.mkv").write_bytes(b"x")
        (app / "temp" / "b.mkv").write_bytes(b"x")
        with mock.patch.object(constants, "APP_ROOT", app), quiet():
            roots = sanitize_media_roots([str(self.root / "drive")])
            scanner = MediaScanner(None, None, {})                        # type: ignore[arg-type]
            files = sorted(p.name for p in scanner._walk_files(roots[0]))
        self.assertEqual(roots, [self.root / "drive"])
        self.assertEqual(files, ["a.mkv"])


@needs_ffmpeg
class DiagnosticsAndSelfTest(PipelineBase):
    def test_cpu_only_benchmark_runs_and_saves_results(self) -> None:
        make_video(self.media / "a.mp4", 40)
        gen = self.generator()
        videos = self.scan()
        job = DiagnosticJob("benchmark")
        self.addCleanup(job.close)
        with quiet():
            data = DiagnosticRunner(gen, job).run(videos, "quick", [], "benchmark")
        self.assertEqual(data["status"], "COMPLETE")
        self.assertTrue(data["rows"] and all(r["status"] == "PASS" for r in data["rows"]), data["rows"])
        self.assertTrue(job.result_json.exists())

    def test_gpu_backends_are_warmed_up_before_and_outside_the_measurements(self) -> None:
        make_video(self.media / "a.mp4", 40)
        gen = self.generator()
        videos = self.scan()
        real = TrickplayGenerator._run_ffmpeg_extract
        calls: list[tuple[str, str | None, str, float | None]] = []

        def fake_gpu(self_, meta, frames_dir, method, backend, **kwargs):
            calls.append((method, backend, kwargs.get("progress_label", ""), kwargs.get("duration_limit")))
            return real(self_, meta, frames_dir, "CPU", None, **kwargs)  # stands in for a GPU path

        job = DiagnosticJob("benchmark")
        self.addCleanup(job.close)
        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", fake_gpu), quiet():
            data = DiagnosticRunner(gen, job).run(videos, "quick", ["cuda"], "benchmark")
        self.assertEqual(calls[0][:2], ("GPU", "cuda"))
        self.assertTrue(calls[0][2].startswith("Warm-up"))
        self.assertLessEqual(calls[0][3], DiagnosticRunner.WARMUP_SECONDS)
        self.assertEqual(data["warmup"]["cuda"]["status"], "PASS")
        self.assertEqual(sorted(r["backend"] for r in data["rows"]), ["cpu", "cuda"], "warm-up must not appear in the measurements")
        measured = [c for c in calls if c[2].startswith("Test ")]
        self.assertEqual(len(measured), 2, "the CPU and cuda measurements; the warm-up, the key frame run and the parallel jobs are extra")
        self.assertEqual(data["status"], "COMPLETE")

    def test_hdr_check_tests_every_kind_of_hdr_and_labels_it(self) -> None:
        import dataclasses
        make_video(self.media / "a.mp4", 40)
        make_video(self.media / "b.mp4", 40)
        gen = self.generator()
        a, b = self.scan()
        videos = [dataclasses.replace(a, hdr=True, hdr_type="HDR10 / PQ"),
                  dataclasses.replace(b, hdr=True, hdr_type="Dolby Vision", dv_profile="8")]
        real = TrickplayGenerator._run_ffmpeg_extract
        job = DiagnosticJob("hdr")
        self.addCleanup(job.close)
        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract",
                               lambda self_, meta, frames_dir, method, backend, **kw: real(self_, meta, frames_dir, "CPU", None, **kw)), quiet():
            data = DiagnosticRunner(gen, job).run(select_hdr_candidates(videos), "quick", ["cuda"], "hdr")
        self.assertEqual(data["status"], "COMPLETE")
        self.assertEqual(sorted({r["hdr_label"] for r in data["rows"]}), ["Dolby Vision P8", "HDR10"])
        for label in ("HDR10", "Dolby Vision P8"):
            gpu = [r for r in data["rows"] if r["hdr_label"] == label and r["backend"] == "cuda"]
            self.assertTrue(gpu and all((job.path / r["comparison"]).exists() for r in gpu), label)
            cpu = [r for r in data["rows"] if r["hdr_label"] == label and r["backend"] == "cpu"]
            self.assertTrue(cpu and all((job.path / r["comparison"]).name.endswith("CPU only.jpg") and (job.path / r["comparison"]).exists()
                                        for r in cpu), f"{label}: the CPU result has its own picture to rate")
        pictures = {o["hdr_label"]: o["picture"] for o in data["overview"]}
        self.assertEqual(sorted(pictures), ["Dolby Vision P8", "HDR10"])
        for label, name in pictures.items():
            self.assertTrue(name.endswith("all paths.jpg") and (job.path / name).exists(), label)
        self.assertTrue(all(r["comparison"].startswith("comparisons/") for r in data["rows"] if r.get("comparison")))

    def test_failed_warm_up_does_not_stop_the_benchmark(self) -> None:
        make_video(self.media / "a.mp4", 40)
        gen = self.generator()
        videos = self.scan()
        real = TrickplayGenerator._run_ffmpeg_extract
        seen = {"warmup": 0}

        def flaky(self_, meta, frames_dir, method, backend, **kwargs):
            if kwargs.get("progress_label", "").startswith("Warm-up"):
                seen["warmup"] += 1
                raise RuntimeError("driver not ready")
            return real(self_, meta, frames_dir, "CPU", None, **kwargs)

        job = DiagnosticJob("benchmark")
        self.addCleanup(job.close)
        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", flaky), quiet():
            data = DiagnosticRunner(gen, job).run(videos, "quick", ["cuda"], "benchmark")
        self.assertEqual(seen["warmup"], 1)
        self.assertEqual(data["warmup"]["cuda"]["status"], "FAILED")
        self.assertEqual(data["status"], "COMPLETE")
        self.assertTrue(all(r["status"] == "PASS" for r in data["rows"]))

    def test_self_test_passes_with_working_ffmpeg(self) -> None:
        job = DiagnosticJob("selftest")
        self.addCleanup(job.close)
        results = run_self_test(self.config, self.state, self.tools.candidate, self.tools, job, progress=lambda _m: None)
        failed = {k: v for k, v in results["checks"].items() if v["status"] != "PASS"}
        self.assertEqual(failed, {})
        self.assertIn("trickplay CPU pipeline", results["checks"])
        self.assertFalse(list(constants.DIRS["temp"].glob("selftest_*")), "self-test must clean its temp files")

    def test_self_test_leaves_no_header_only_log_file(self) -> None:
        before = set(constants.DIRS["logs"].glob("*.log"))
        job = DiagnosticJob("selftest")
        self.addCleanup(job.close)
        run_self_test(self.config, self.state, self.tools.candidate, self.tools, job, progress=lambda _m: None)
        self.assertEqual(set(constants.DIRS["logs"].glob("*.log")), before)

    def test_null_logger_writes_nothing_but_still_echoes(self) -> None:
        from modules.logger import NullRunLogger
        logger = NullRunLogger("benchmark")
        before = sorted(p.name for p in constants.DIRS["logs"].iterdir())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            logger.log("INFO", "hello", echo=True)
        logger.event("X", a=1)
        logger.failed(self.media / "x.mp4", "err")
        logger.write_summary({"a": 1})
        logger.close()
        self.assertEqual(sorted(p.name for p in constants.DIRS["logs"].iterdir()), before)
        self.assertIn("hello", buf.getvalue())

    def test_self_test_reports_missing_ffmpeg_as_failure(self) -> None:
        job = DiagnosticJob("selftest")
        self.addCleanup(job.close)
        results = run_self_test(self.config, self.state, None, None, job, progress=lambda _m: None)
        self.assertEqual(results["checks"]["ffmpeg"]["status"], "FAIL")
        self.assertGreaterEqual(results["failed"], 1)


class SuspectFrameCount(PipelineBase):
    """A video whose length (as the container states it) does not fit the thumbnails it gave is reported, and its output is kept."""

    def make_suspect(self):
        video = make_video(self.media / "short.mp4", 25)           # 25 s at 5 s: 5 thumbnails
        videos = self.scan()
        videos[0].duration = 100.0                                  # the container claims 100 s, the video stream holds 25 s
        gen, plan, summary = self.run_all(videos=videos)
        return video, gen, summary

    def test_a_normal_video_is_not_suspect(self) -> None:
        make_video(self.media / "a.mp4", 25)
        gen, _, summary = self.run_all()
        self.assertEqual((summary["completed"], summary["suspect_videos"], summary["suspect_details"]), (1, 0, []))
        self.assertNotIn("SUSPECT_FRAME_COUNT", gen.logger.events_path.read_text(encoding="utf-8"))

    def test_too_few_thumbnails_are_reported_but_the_output_is_kept(self) -> None:
        video, gen, summary = self.make_suspect()
        self.assertEqual((summary["completed"], summary["failed"], summary["suspect_videos"]), (1, 0, 1))
        detail = summary["suspect_details"][0]
        self.assertEqual((detail["path"], detail["frames"], detail["expected_frames"]), (str(video), 5, 20))
        self.assertIn("short", detail["note"])
        self.assertIn("SUSPECT_FRAME_COUNT", gen.logger.events_path.read_text(encoding="utf-8"))
        self.assertTrue(validate_tile_directory(target_dir_for(video, gen.settings), gen.settings, 25.0, expected_exact_tiles=1).valid)

    def test_the_next_run_does_not_rebuild_it_and_verify_still_lists_it(self) -> None:
        video, gen, _ = self.make_suspect()
        videos = self.scan()
        videos[0].duration = 100.0
        with quiet():
            item = self.generator().plan(videos)[0]
        self.assertEqual((item.action, item.reason), ("SKIP", "SKIPPED_VALID"), "a known deviation is not a reason to build it again")
        row = verify_video(videos[0], gen.settings, self.state)
        self.assertEqual(row.status, VALID, "the output itself is fine")
        self.assertIn("short", row.suspect)


class NormalizedTimestamps(PipelineBase):
    """A video whose thumbnails do not fit its length gets a second extraction with the time stamps made from the frame count."""

    def run_claiming(self, duration: float, frame_rate):
        make_video(self.media / "v.mp4", 25)                        # 250 pictures at 10 per second; 5 s interval
        videos = self.scan()
        videos[0].duration = duration
        gen = self.generator()
        calls: list[dict] = []
        real = gen._run_ffmpeg_extract

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real(*args, **kwargs)
        with mock.patch.object(gen, "_run_ffmpeg_extract", spy), mock.patch.object(self.tools, "reference_frame_rate", return_value=frame_rate):
            _, _, summary = self.run_all(gen=gen, videos=videos)
        return gen, summary, calls

    def test_a_normal_video_is_extracted_once(self) -> None:
        gen, summary, calls = self.run_claiming(25.0, 10.0)
        self.assertEqual((summary["completed"], summary["suspect_videos"], len(calls)), (1, 0, 1))
        self.assertNotIn("TIMESTAMPS_NORMALIZED", gen.logger.events_path.read_text(encoding="utf-8"))

    def test_the_second_extraction_is_kept_when_it_comes_closer(self) -> None:
        # The container "claims" 5 s while the time stamps spread the 250 pictures over 25 s (5 thumbnails instead of 1);
        # at 50 frames/s the 250 pictures last 5 s, which is what the claim needs.
        gen, summary, calls = self.run_claiming(5.0, 50.0)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0].get("normalize_fps"))
        self.assertEqual(calls[1].get("normalize_fps"), 50.0)
        self.assertEqual((summary["completed"], summary["failed"], summary["suspect_videos"]), (1, 0, 0))
        events = gen.logger.events_path.read_text(encoding="utf-8")
        self.assertIn("TIMESTAMPS_NORMALIZED", events)
        self.assertIn('"frames_before": 5', events)
        self.assertIn('"frames_after": 1', events)
        self.assertEqual(gen.state.current_target_output(self.media / "v.mp4", gen.settings)["frames"], 1)

    def test_the_first_result_is_kept_when_the_second_does_not_help(self) -> None:
        gen, summary, calls = self.run_claiming(5.0, 10.0)         # the true frame rate: same pictures, same count
        self.assertEqual(len(calls), 2)
        self.assertEqual((summary["completed"], summary["suspect_videos"]), (1, 1))
        self.assertEqual(summary["suspect_details"][0]["frames"], 5)
        self.assertNotIn("TIMESTAMPS_NORMALIZED", gen.logger.events_path.read_text(encoding="utf-8"))

    def test_without_a_usable_frame_rate_there_is_no_second_extraction(self) -> None:
        _, summary, calls = self.run_claiming(5.0, None)
        self.assertEqual((len(calls), summary["suspect_videos"]), (1, 1))

    def test_too_few_thumbnails_are_not_extracted_again(self) -> None:
        # The container claims 100 s, the stream ends after 25 s: no way of extracting makes the missing 15 thumbnails appear.
        gen, summary, calls = self.run_claiming(100.0, 2.5)
        self.assertEqual((len(calls), summary["completed"], summary["suspect_videos"]), (1, 1, 1))
        self.assertNotIn("TIMESTAMPS_NORMALIZED", gen.logger.events_path.read_text(encoding="utf-8"))

    def test_the_real_ffprobe_gives_the_frame_rate_of_the_file(self) -> None:
        video = make_video(self.media / "v.mp4", 5)
        self.assertEqual(self.tools.reference_frame_rate(video), 10.0)
        self.assertIsNone(self.tools.reference_frame_rate(self.media / "missing.mp4"))


class KeyFrameOnlyExtraction(PipelineBase):
    """The option "key frames only": off by default; when on, the decoder skips what is not a key frame and a failure falls back to the usual way."""

    def generate(self, *, keyframes: bool, claimed: float | None = None, frame_rate=None, gop: int = 20, limit: float | None = None):
        make_video(self.media / "v.mp4", 25, gop=gop)              # a key frame every gop / 10 s; 5 s interval
        if limit is not None: self.config["trickplay"]["keyframe_repeat_limit_percent"] = limit
        self.config["trickplay"]["keyframe_only"] = keyframes
        videos = self.scan()
        if claimed is not None: videos[0].duration = claimed
        gen = self.generator()
        commands: list[list[str]] = []
        import subprocess
        real_popen = subprocess.Popen

        def spy(cmd, *args, **kwargs):
            commands.append([str(part) for part in cmd])
            return real_popen(cmd, *args, **kwargs)
        patches = [mock.patch("modules.extraction.subprocess.Popen", spy)]
        if frame_rate is not None: patches.append(mock.patch.object(self.tools, "reference_frame_rate", return_value=frame_rate))
        with contextlib.ExitStack() as stack:
            for patch in patches: stack.enter_context(patch)
            _, _, summary = self.run_all(gen=gen, videos=videos)
        return gen, summary, commands

    def test_off_by_default(self) -> None:
        self.assertIs(constants.DEFAULT_CONFIG["trickplay"]["keyframe_only"], False)
        gen, summary, commands = self.generate(keyframes=False)
        self.assertEqual((summary["completed"], len(commands)), (1, 1))
        self.assertNotIn("-skip_frame", commands[0])
        self.assertIn('"keyframe_only": false', gen.logger.events_path.read_text(encoding="utf-8"))

    def test_the_decoder_skips_what_is_not_a_key_frame_and_the_count_still_fits(self) -> None:
        gen, summary, commands = self.generate(keyframes=True)
        self.assertEqual((summary["completed"], summary["failed"], summary["suspect_videos"], len(commands)), (1, 0, 0, 1))
        cmd = commands[0]
        self.assertEqual(cmd[cmd.index("-skip_frame") + 1], "nokey")
        self.assertLess(cmd.index("-skip_frame"), cmd.index("-i"), "an input option goes before the input")
        self.assertIn('"keyframe_only": true', gen.logger.events_path.read_text(encoding="utf-8"))
        frames = gen.state.current_target_output(self.media / "v.mp4", gen.settings)["frames"]
        self.assertLessEqual(abs(frames - 5), 1, "the picture is the nearest key frame but there is still one every interval")

    def test_a_failure_falls_back_to_decoding_every_frame(self) -> None:
        make_video(self.media / "v.mp4", 25, gop=20)
        self.config["trickplay"]["keyframe_only"] = True
        gen = self.generator()
        real = gen._run_ffmpeg_extract
        calls: list[bool] = []

        def fail_on_keyframes(*args, **kwargs):
            calls.append(bool(kwargs.get("keyframes")))
            if kwargs.get("keyframes"): return ExtractResult(False, "CPU", None, 0.1, 0, "this decoder cannot skip pictures")
            return real(*args, **kwargs)
        with mock.patch.object(gen, "_run_ffmpeg_extract", fail_on_keyframes):
            _, _, summary = self.run_all(gen=gen)
        self.assertEqual((summary["completed"], summary["failed"]), (1, 0))
        self.assertEqual(calls, [True, False])
        self.assertIn('"keyframe_only": false', gen.logger.events_path.read_text(encoding="utf-8"), "the output says how it was made")

    def test_a_video_that_does_not_fit_gets_every_frame_and_then_the_frame_count(self) -> None:
        gen, summary, commands = self.generate(keyframes=True, claimed=5.0, frame_rate=50.0)
        self.assertEqual([("-skip_frame" in cmd) for cmd in commands], [True, False, False])
        self.assertEqual((summary["completed"], summary["suspect_videos"]), (1, 0))
        self.assertIn("TIMESTAMPS_NORMALIZED", gen.logger.events_path.read_text(encoding="utf-8"))

    def test_a_video_with_too_few_thumbnails_is_not_decoded_again(self) -> None:
        gen, summary, commands = self.generate(keyframes=True, claimed=100.0, frame_rate=2.5)
        self.assertEqual(([("-skip_frame" in cmd) for cmd in commands], summary["suspect_videos"]), ([True], 1))

    def test_key_frames_further_apart_than_the_interval_bring_back_every_frame(self) -> None:
        # a key frame every 10 s but a thumbnail every 5 s: every second thumbnail would repeat the one before
        gen, summary, commands = self.generate(keyframes=True, gop=100)
        self.assertEqual([("-skip_frame" in cmd) for cmd in commands], [True, False])
        events = gen.logger.events_path.read_text(encoding="utf-8")
        self.assertIn("KEYFRAMES_TOO_SPARSE", events)
        self.assertIn('"keyframe_only": false', events)
        self.assertEqual((summary["completed"], summary["suspect_videos"]), (1, 0))

    def test_the_result_of_the_second_extraction_is_what_is_packed(self) -> None:
        gen, _, _ = self.generate(keyframes=True, gop=100)
        output = target_dir_for(self.media / "v.mp4", gen.settings)
        frames = gen.state.current_target_output(self.media / "v.mp4", gen.settings)["frames"]
        self.assertEqual(frames, 5)
        self.assertTrue((output / "0.jpg").is_file())

    def test_a_limit_of_zero_never_brings_back_every_frame(self) -> None:
        gen, _, commands = self.generate(keyframes=True, gop=100, limit=0.0)
        self.assertEqual(len(commands), 1)
        self.assertNotIn("KEYFRAMES_TOO_SPARSE", gen.logger.events_path.read_text(encoding="utf-8"))

    def test_a_limit_above_the_repeats_keeps_the_key_frame_result(self) -> None:
        gen, _, commands = self.generate(keyframes=True, gop=100, limit=50.0)         # 2 of 5 thumbnails repeat = 40 %
        self.assertEqual(len(commands), 1)
        self.assertIn('"keyframe_only": true', gen.logger.events_path.read_text(encoding="utf-8"))

    def test_close_key_frames_are_left_alone(self) -> None:
        _, _, commands = self.generate(keyframes=True, gop=20)
        self.assertEqual(len(commands), 1)

    def test_repeated_frames_are_counted_by_their_bytes(self) -> None:
        folder = self.root / "rep"
        folder.mkdir()
        for number, content in enumerate([b"a", b"a", b"b", b"b", b"b", b"a"], 1): (folder / f"frame_{number:08d}.jpg").write_bytes(content)
        self.assertEqual(self.generator().repeated_frames(folder), (3, 6))
        self.assertEqual(self.generator().repeated_frames(self.root / "none"), (0, 0))

    def test_when_every_frame_fails_the_key_frame_result_is_kept(self) -> None:
        make_video(self.media / "v.mp4", 25, gop=100)
        self.config["trickplay"]["keyframe_only"] = True
        gen = self.generator()
        real = gen._run_ffmpeg_extract

        def fail_without_keyframes(*args, **kwargs):
            if not kwargs.get("keyframes"): return ExtractResult(False, "CPU", None, 0.1, 0, "no")
            return real(*args, **kwargs)
        with mock.patch.object(gen, "_run_ffmpeg_extract", fail_without_keyframes):
            _, _, summary = self.run_all(gen=gen)
        self.assertEqual((summary["completed"], summary["failed"]), (1, 0))
        self.assertNotIn("KEYFRAMES_TOO_SPARSE", gen.logger.events_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
