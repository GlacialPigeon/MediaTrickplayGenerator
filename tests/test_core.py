# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import contextlib
import io
import json
import shutil
import os
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import HAVE_PIL, IsolatedApp, needs_symlinks, new_logger, open_state

from modules import constants
from modules.config import ConfigManager, deep_merge
from modules.diagnostics import diagnostic_segments, select_hdr_candidates
from modules.error_summary import diagnostic_error_summary
from modules.ffmpeg_tools import parse_frame_rate
from modules.fingerprint import sha256_file, source_edge_fingerprint
from modules.formatting import human_bytes, human_duration
from modules.maintenance import all_items, old_items, remove_items
from modules.models import TrickplaySettings, VideoMeta
from modules.prompts import parse_number_selection
from modules.tiles import describe_frame_deviation, expected_foreign_tile_counts, frame_count_deviation, parse_trickplay_width_dir, validate_tile_directory


def settings(**kw) -> TrickplaySettings:
    base = dict(width=64, interval=10.0, cols=3, rows=2, qscale=4, jpeg_quality=80)
    base.update(kw)
    return TrickplaySettings(**base)


def meta(path: Path, **kw) -> VideoMeta:
    base = dict(path=path, size=100, mtime_ns=5, duration=60.0, codec="h264", profile="High", pix_fmt="yuv420p", width=1920, height=1080, hdr=False)
    base.update(kw)
    return VideoMeta(**base)


def write_sprites(directory: Path, st: TrickplaySettings, count: int, height: int | None = None) -> None:
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (st.width * st.cols, height or 36 * st.rows), (i * 20 % 255, 80, 120)).save(directory / f"{i}.jpg", "JPEG")


class Formatting(unittest.TestCase):
    def test_human_values(self) -> None:
        self.assertEqual(human_bytes(512), "512 B")
        self.assertEqual(human_bytes(1536), "1.50 KiB")
        self.assertEqual(human_duration(75), "01:15")
        self.assertEqual(human_duration(3661), "01:01:01")
        self.assertEqual(human_duration(None), "N/A")
        self.assertEqual(human_duration(float("nan")), "N/A")

    def test_number_selection(self) -> None:
        self.assertEqual(parse_number_selection("1,3-5, x, 9", 6), {1, 3, 4, 5})
        self.assertEqual(parse_number_selection("0-2", 5), {1, 2})

    def test_error_summaries(self) -> None:
        self.assertIn("space", diagnostic_error_summary("write failed: No space left on device"))
        self.assertIn("driver", diagnostic_error_summary("Cannot load nvcuda.dll"))
        self.assertTrue(diagnostic_error_summary("x\ny"))
        self.assertEqual(diagnostic_error_summary(""), "Processing failed.")
        noisy = "[rv40 @ 0x1] Invalid decoder state: B-frame without reference data.\n[vf#0:0 @ 0x2] Error reinitializing filters!"
        self.assertEqual(diagnostic_error_summary(noisy), "Invalid decoder state: B-frame without reference data.", "the cause, not 'see the log'")

    def test_repeated_ffmpeg_lines_are_shown_once_with_a_count(self) -> None:
        from modules.error_summary import compact_ffmpeg_error
        text = "\n".join(f"[rv40 @ 0x{i:08x}] Invalid decoder state: B-frame without reference data." for i in range(100)) + "\n[mjpeg @ 0x9] Error opening encoder"
        compact = compact_ffmpeg_error(text).splitlines()
        self.assertEqual(len(compact), 2)
        self.assertIn("[repeated 100 times]", compact[0])
        self.assertEqual(compact_ffmpeg_error("a\nb\na"), "a\nb\na", "only neighbours are merged")
        self.assertEqual(compact_ffmpeg_error(""), "")


class Models(unittest.TestCase):
    def test_settings_validation_and_naming(self) -> None:
        st = settings()
        self.assertEqual(st.folder_name, "64 - 3x2")
        self.assertEqual(st.capacity, 6)
        self.assertEqual(st.hash, settings().hash)
        self.assertNotEqual(st.hash, settings(interval=5.0).hash)
        for bad in (dict(interval=0), dict(interval=float("inf")), dict(width=0), dict(cols=0), dict(qscale=0), dict(jpeg_quality=101)):
            with self.assertRaises(ValueError):
                settings(**bad)

    def test_video_meta_properties(self) -> None:
        self.assertEqual(meta(Path("a"), pix_fmt="yuv420p10le").bit_depth, 10)
        self.assertEqual(meta(Path("a")).resolution_class, "1080p")
        self.assertEqual(meta(Path("a"), width=3840, height=2160).resolution_class, "4K")
        self.assertEqual(meta(Path("a"), width=640, height=480).resolution_class, "SD")

    def test_hdr_candidate_selection(self) -> None:
        a = meta(Path("a"), hdr=True, codec="h264", width=1920, height=1080)
        b = meta(Path("b"), hdr=True, codec="hevc", width=3840, height=2160)
        self.assertEqual(select_hdr_candidates([meta(Path("c")), a, b]), [b], "same kind of HDR: the better file")
        self.assertEqual(select_hdr_candidates([meta(Path("c"))]), [])


class Fingerprints(unittest.TestCase):
    def test_edge_fingerprint_sees_changes_at_both_ends(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "f.bin"
            f.write_bytes(os.urandom(300 * 1024))
            base = source_edge_fingerprint(f, 0)
            data = bytearray(f.read_bytes())
            data[10] ^= 1
            f.write_bytes(bytes(data))
            self.assertNotEqual(base, source_edge_fingerprint(f, 0))
            data[-10] ^= 1
            f.write_bytes(bytes(data))
            self.assertNotEqual(sha256_file(f), "")


class Extensions(unittest.TestCase):
    def test_tables_are_never_probed_as_video(self) -> None:
        for ext in (".csv", ".tsv"):
            self.assertIn(ext, constants.KNOWN_NONVIDEO_EXTENSIONS)
            self.assertNotIn(ext, constants.KNOWN_VIDEO_EXTENSIONS)
        self.assertFalse(constants.KNOWN_VIDEO_EXTENSIONS & constants.KNOWN_NONVIDEO_EXTENSIONS)


class Segments(unittest.TestCase):
    def test_quick_mode_windows(self) -> None:
        self.assertEqual(diagnostic_segments(90, "quick"), [(0.0, 90)])
        self.assertEqual(diagnostic_segments(200, "quick"), [(40.0, 120.0)])
        three = diagnostic_segments(1800, "quick")
        self.assertEqual(len(three), 3)
        for (s1, d1), (s2, _d2) in zip(three, three[1:]):
            self.assertLessEqual(s1 + d1, s2, "windows must not overlap")
        self.assertEqual(diagnostic_segments(1800, "full"), [(0.0, 1800)])
        with self.assertRaises(ValueError): diagnostic_segments(0, "quick")
        with self.assertRaises(ValueError): diagnostic_segments(10, "bogus")


class Config(IsolatedApp):
    def test_deep_merge_keeps_defaults_and_unknown_keys(self) -> None:
        merged = deep_merge({"a": {"b": 1, "c": 2}, "d": 3}, {"a": {"b": 9}, "extra": 1})
        self.assertEqual(merged, {"a": {"b": 9, "c": 2}, "d": 3, "extra": 1})

    def test_save_load_roundtrip_and_backup(self) -> None:
        mgr = ConfigManager()
        mgr.load()
        mgr.config["media_roots"] = ["/x"]
        mgr.save()
        mgr.config["media_roots"] = ["/y"]
        mgr.save()
        fresh = ConfigManager()
        fresh.load()
        self.assertEqual(fresh.config["media_roots"], ["/y"])
        self.assertEqual(json.loads(fresh.backup_path.read_text())["media_roots"], ["/x"])

    def test_corrupt_config_falls_back_to_backup_then_restores(self) -> None:
        mgr = ConfigManager()
        mgr.load()
        mgr.config["media_roots"] = ["/keep"]
        mgr.save()
        mgr.save()
        mgr.path.write_text("{ not json")
        broken = ConfigManager()
        broken.load()
        self.assertTrue(broken.corrupt_error)
        self.assertTrue(broken.loaded_from_backup)
        self.assertEqual(broken.config["media_roots"], ["/keep"])
        self.assertTrue(broken.restore_backup())
        self.assertEqual(json.loads(broken.path.read_text())["media_roots"], ["/keep"])
        self.assertTrue(list(broken.path.parent.glob("config.json.broken.*")))

    def test_unreadable_backup_is_reported_not_silent(self) -> None:
        mgr = ConfigManager()
        mgr.load()
        mgr.save()
        mgr.save()
        mgr.path.write_text("{ broken")
        mgr.backup_path.write_text("{ also broken")
        with contextlib.redirect_stdout(io.StringIO()):
            broken = ConfigManager()
            broken.load()
        self.assertTrue(broken.corrupt_error)
        self.assertFalse(broken.loaded_from_backup)
        self.assertTrue(any("backup" in w for w in broken.warnings), broken.warnings)

    def test_failed_backup_refresh_is_reported_but_save_still_succeeds(self) -> None:
        mgr = ConfigManager()
        mgr.load()
        mgr.config["media_roots"] = ["/a"]
        mgr.save()
        real = __import__("modules.config", fromlist=["atomic_write_json"]).atomic_write_json

        def flaky(path, data):
            if Path(path) == mgr.backup_path: raise OSError("no space")
            return real(path, data)

        mgr.config["media_roots"] = ["/b"]
        with mock.patch("modules.config.atomic_write_json", flaky), contextlib.redirect_stdout(io.StringIO()):
            mgr.save()
        self.assertTrue(any("backup" in w for w in mgr.warnings))
        self.assertEqual(json.loads(mgr.path.read_text())["media_roots"], ["/b"])

    def test_newer_schema_is_rejected_not_silently_downgraded(self) -> None:
        mgr = ConfigManager()
        mgr.path.write_text(json.dumps({"schema_version": constants.CONFIG_SCHEMA + 1}))
        mgr.load()
        self.assertIn("newer", mgr.corrupt_error or "")

    def test_settings_of_removed_features_are_dropped_and_removed_choices_become_safe_ones(self) -> None:
        mgr = ConfigManager()
        mgr.path.write_text(json.dumps({"workers": {"probe_concurrency": {"D:": 4}, "per_storage_group": 2},
                                        "acceleration": {"hdr_gpu_disabled": True, "mode": "cpu"},
                                        "trickplay": {"same_width_policy": "ask", "other_width_policy": "skip_any", "width": 320}}))
        mgr.load()
        self.assertNotIn("probe_concurrency", mgr.config["workers"])
        self.assertNotIn("hdr_gpu_disabled", mgr.config["acceleration"])
        t = mgr.config["trickplay"]
        self.assertEqual((t["same_width_policy"], t["other_width_policy"], t["width"]), ("skip_valid", "keep_add", 320))
        self.assertEqual((mgr.config["workers"]["per_storage_group"], mgr.config["acceleration"]["mode"]), (2, "cpu"))
        mgr.path.write_text(json.dumps({"trickplay": {"same_width_policy": "rebuild_all", "other_width_policy": "replace"}}))
        mgr.load()
        self.assertEqual((mgr.config["trickplay"]["same_width_policy"], mgr.config["trickplay"]["other_width_policy"]), ("rebuild_all", "replace"))
        mgr.path.write_text(json.dumps({"trickplay": {"other_width_policy": "ask"}}))
        mgr.save()   # keeps the previous file as backup; restoring a backup with an old choice must also fix it
        mgr.backup_path.write_text(json.dumps({"trickplay": {"other_width_policy": "ask"}}))
        self.assertTrue(mgr.restore_backup())
        self.assertEqual(mgr.config["trickplay"]["other_width_policy"], "keep_add")

    def test_start_fresh_archives_old_file(self) -> None:
        mgr = ConfigManager()
        mgr.load()
        mgr.config["media_roots"] = ["/old"]
        mgr.save()
        mgr.start_fresh()
        self.assertEqual(mgr.config["media_roots"], [])
        self.assertTrue(list(mgr.path.parent.glob("config.json.broken.*")))


@unittest.skipUnless(HAVE_PIL, "Pillow required")
class TileValidation(IsolatedApp):
    def test_valid_directory(self) -> None:
        st = settings()
        d = self.root / "t"
        write_sprites(d, st, 3)
        r = validate_tile_directory(d, st, 170.0, expected_exact_tiles=3)
        self.assertTrue(r.valid, r.reason)
        self.assertEqual(r.tiles, 3)
        self.assertEqual(r.tile_size, (192, 72))

    def test_each_failure_is_detected(self) -> None:
        st = settings()
        cases = {}
        d = self.root / "missing"
        cases["missing directory"] = d
        d = self.root / "gap"
        write_sprites(d, st, 3)
        (d / "1.jpg").unlink()
        cases["non-sequential"] = d
        d = self.root / "extra"
        write_sprites(d, st, 2)
        (d / "notes.txt").write_text("x")
        cases["unexpected"] = d
        d = self.root / "zero"
        write_sprites(d, st, 2)
        (d / "1.jpg").write_bytes(b"")
        cases["zero-byte"] = d
        d = self.root / "wrongw"
        write_sprites(d, settings(width=32), 2)
        cases["wrong sprite width"] = d
        d = self.root / "junk"
        write_sprites(d, st, 2)
        (d / "0.jpg").write_bytes(b"not a jpeg at all")
        cases["unreadable"] = d
        d = self.root / "count"
        write_sprites(d, st, 2)
        cases["expected 5 tiles"] = d
        d = self.root / "empty"
        d.mkdir()
        cases["no numeric"] = d
        for expected, directory in cases.items():
            exact = 5 if expected == "expected 5 tiles" else None
            r = validate_tile_directory(directory, st, 60.0 if exact is None else 170.0, expected_exact_tiles=exact)
            self.assertFalse(r.valid, expected)
            self.assertIn(expected, r.reason, expected)

    def test_foreign_tile_count_tolerance(self) -> None:
        st = settings()  # 6 thumbnails per sprite, interval 10s
        self.assertEqual(expected_foreign_tile_counts(600, st), {10, 11})
        self.assertEqual(expected_foreign_tile_counts(0, st), set())
        self.assertEqual(parse_trickplay_width_dir(Path("640 - 10x10")), (640, 10, 10))
        self.assertIsNone(parse_trickplay_width_dir(Path("640 - 10x10.__building__")))


class StateDatabase(IsolatedApp):
    def test_output_history_roundtrip(self) -> None:
        state = open_state()
        st, m = settings(), meta(self.root / "a.mkv", size=10, mtime_ns=20)
        self.assertIsNone(state.current_target_output(m.path, st))
        state.record_output(m, st, frames=12, tiles=2, output_bytes=4096, method="CPU", backend=None, source_fingerprint="fp")
        row = state.current_target_output(m.path, st)
        self.assertEqual((row["tiles"], row["source_size"], row["source_fingerprint"]), (2, 10, "fp"))
        self.assertEqual(state.previous_output(m.path, st.hash)["frames"], 12)
        state.record_output(m, st, frames=13, tiles=3, output_bytes=8192, method="GPU", backend="cuda", source_fingerprint=None)
        self.assertEqual(state.current_target_output(m.path, st)["tiles"], 3, "re-recording replaces, never duplicates")
        self.assertAlmostEqual(state.estimate_bytes_per_second(st.hash), 8192 / 60.0)
        state.close()

    def test_last_probed_and_meta(self) -> None:
        state = open_state()
        m = meta(self.root / "v.mkv", size=7, mtime_ns=9)
        self.assertIsNone(state.last_probed(m.path))
        state.upsert_video(m)
        self.assertTrue(state.last_probed(m.path))
        self.assertIsNone(state.last_probed(self.root / "other.mkv"))
        self.assertIsNone(state.get_meta("probe_benchmark.X"))
        state.set_meta("probe_benchmark.X", "2026-10-27T01:00:00+00:00")
        state.set_meta("probe_benchmark.X", "2026-10-27T02:00:00+00:00")
        self.assertEqual(state.get_meta("probe_benchmark.X"), "2026-10-27T02:00:00+00:00")
        state.close()

    def test_probe_cache_requires_matching_signature(self) -> None:
        state = open_state()
        m = meta(self.root / "v.mkv", size=7, mtime_ns=9, hdr=True)
        state.upsert_video(m)
        hit = state.cached_video(m.path, 7, 9)
        self.assertTrue(hit.hdr)
        self.assertIsNone(state.cached_video(m.path, 8, 9))
        self.assertIsNone(state.cached_video(m.path, 7, 10))
        state.close()

    def test_backend_failure_memory(self) -> None:
        state = open_state()
        m = meta(self.root / "v.mkv")
        self.assertFalse(state.known_unsupported("cuda", m))
        state.record_backend_failure("cuda", m, "transient", "boom")
        self.assertFalse(state.known_unsupported("cuda", m))
        state.record_backend_failure("cuda", m, "unsupported", "nope")
        self.assertTrue(state.known_unsupported("cuda", m))
        self.assertFalse(state.known_unsupported("qsv", m))
        state.close()

    def test_reset_and_corruption_recovery_archive_the_old_database(self) -> None:
        state = open_state()
        m = meta(self.root / "v.mkv")
        state.record_output(m, settings(), 1, 1, 1, "CPU", None, None)
        backup = state.reset_with_backup()
        self.assertTrue(backup.exists())
        self.assertIsNone(state.current_target_output(m.path, settings()))
        state.close()
        state.path.write_bytes(b"this is not sqlite" * 100)
        with self.assertRaises(Exception):
            state.open()
            state.current_target_output(m.path, settings())
        state.close()
        archived = state.recover_corrupt()
        self.assertTrue(archived.exists())
        self.assertIsNone(state.current_target_output(m.path, settings()))
        state.close()


@unittest.skipUnless(HAVE_PIL, "Pillow required")
class VerifyLogic(IsolatedApp):
    def test_every_status(self) -> None:
        from modules.tiles import building_dir_for, target_dir_for
        from modules.verify import INVALID, MISSING, SETTINGS_CHANGED, SOURCE_CHANGED, VALID, verify_video, verify_videos
        state = open_state()
        st = settings()

        def video(name: str, **kw) -> VideoMeta:
            p = self.media / name
            p.write_bytes(b"x")
            return meta(p, size=1, mtime_ns=p.stat().st_mtime_ns, duration=170.0, **kw)

        ok = video("ok.mkv")
        write_sprites(target_dir_for(ok.path, st), st, 3)
        state.record_output(ok, st, 17, 3, 1, "CPU", None, None)
        self.assertEqual(verify_video(ok, st, state).status, VALID)

        foreign = video("foreign.mkv")
        write_sprites(target_dir_for(foreign.path, st), st, 3)
        self.assertEqual(verify_video(foreign, st, state).status, VALID, "valid output without history still verifies")

        missing = video("missing.mkv")
        self.assertEqual(verify_video(missing, st, state).status, MISSING)

        changed = video("changed.mkv")
        write_sprites(target_dir_for(changed.path, st), st, 3)
        state.record_output(changed, st, 17, 3, 1, "CPU", None, None)
        newer = meta(changed.path, size=999, mtime_ns=1, duration=170.0)
        self.assertEqual(verify_video(newer, st, state).status, SOURCE_CHANGED)

        bad = video("bad.mkv")
        write_sprites(target_dir_for(bad.path, st), st, 3)
        state.record_output(bad, st, 17, 3, 1, "CPU", None, None)
        (target_dir_for(bad.path, st) / "2.jpg").unlink()
        row = verify_video(bad, st, state)
        self.assertEqual(row.status, INVALID)

        extra = video("extra.mkv")
        write_sprites(target_dir_for(extra.path, st), st, 3)
        (target_dir_for(extra.path, st) / "x.txt").write_text("x")
        row = verify_video(extra, st, state)
        self.assertEqual(row.status, INVALID)
        self.assertIn("unexpected", row.reason)

        regen = video("regen.mkv")
        write_sprites(target_dir_for(regen.path, st), st, 3)
        state.record_output(regen, settings(interval=5.0), 17, 3, 1, "CPU", None, None)
        self.assertEqual(verify_video(regen, st, state).status, SETTINGS_CHANGED)

        stale = video("stale.mkv")
        write_sprites(target_dir_for(stale.path, st), st, 3)
        building_dir_for(stale.path, st).mkdir(parents=True)
        (target_dir_for(stale.path, st).parent / "64 - 3x2.__old__.2026-01-01_000000.abc").mkdir()
        row = verify_video(stale, st, state)
        self.assertTrue(row.stale_building)
        self.assertEqual(len(row.leftovers), 1)

        videos = [ok, foreign, missing, newer, bad, extra, regen, stale]
        summary, problems = verify_videos(videos, st, state)
        self.assertEqual(summary["total"], 8)
        self.assertEqual((summary["valid"], summary["missing"], summary["invalid"], summary["source_changed"], summary["settings_changed"]), (3, 1, 2, 1, 1))
        self.assertEqual((summary["unexpected"], summary["stale_building"], summary["leftover_dirs"]), (1, 1, 1))
        self.assertEqual(len(problems), 6)
        stopped, _ = verify_videos(videos, st, state, should_stop=lambda: True)
        self.assertEqual(stopped["canceled"], 8)
        state.close()


class FFmpegLocation(IsolatedApp):
    """FFmpeg is taken from the `ffmpeg` folder beside the program; on Linux the system's FFmpeg is the fallback."""

    def setUp(self) -> None:
        super().setUp()
        from modules import platform_api
        patcher = mock.patch.object(platform_api.platform(), "system_ffmpeg", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _install(self, folder: Path) -> None:
        from modules import platform_api
        folder.mkdir(parents=True, exist_ok=True)
        for name in platform_api.ffmpeg_executable_names():
            (folder / name).write_bytes(b"fake")

    def discover(self, config: dict | None = None):
        from modules.ffmpeg_registry import FFmpegRegistry
        return FFmpegRegistry().discover(config or {})

    def test_nothing_found_by_default(self) -> None:
        self.assertEqual(self.discover(), [])

    def test_ffmpeg_bin_beside_the_program_is_found(self) -> None:
        self._install(constants.APP_ROOT / "ffmpeg" / "bin")
        found = self.discover()
        self.assertEqual([c.root for c in found], [constants.APP_ROOT / "ffmpeg"])
        self.assertEqual(found[0].source, "portable")

    def test_builds_in_subfolders_are_found_with_their_date(self) -> None:
        self._install(constants.APP_ROOT / "ffmpeg" / "ffmpeg-2026-03-01-build" / "bin")
        self._install(constants.APP_ROOT / "ffmpeg" / "ffmpeg-2026-09-15-build" / "bin")
        found = self.discover()
        self.assertEqual(sorted(c.folder_date for c in found), ["2026-03-01", "2026-09-15"])

    def test_other_locations_are_ignored(self) -> None:
        self._install(constants.APP_ROOT / "somewhere" / "bin")
        self._install(constants.APP_ROOT)
        self._install(self.root / "custom" / "bin")
        found = self.discover({"ffmpeg": {"custom_path": str(self.root / "custom")}})
        self.assertEqual(found, [])

    def test_system_ffmpeg_is_a_candidate_when_the_platform_provides_one(self) -> None:
        from modules import platform_api
        ffmpeg_name, ffprobe_name = platform_api.ffmpeg_executable_names()
        on_path = self.root / "pathdir"
        self._install(on_path)
        with mock.patch.object(platform_api.platform(), "system_ffmpeg", return_value=(on_path / ffmpeg_name, on_path / ffprobe_name)):
            found = self.discover()
        self.assertEqual([(c.source, c.root) for c in found], [("system", on_path)])

    def test_system_ffmpeg_needs_both_programs(self) -> None:
        from modules import platform_api
        ffmpeg_name, ffprobe_name = platform_api.ffmpeg_executable_names()
        on_path = self.root / "pathdir"
        on_path.mkdir()
        (on_path / ffmpeg_name).write_bytes(b"fake")
        with mock.patch.object(platform_api.platform(), "system_ffmpeg", return_value=(on_path / ffmpeg_name, on_path / ffprobe_name)):
            self.assertEqual(self.discover(), [])

    def test_a_build_beside_the_program_wins_over_the_system_one(self) -> None:
        from modules import platform_api
        from modules.ffmpeg_registry import FFmpegRegistry
        ffmpeg_name, ffprobe_name = platform_api.ffmpeg_executable_names()
        on_path = self.root / "pathdir"
        self._install(on_path)
        self._install(constants.APP_ROOT / "ffmpeg" / "bin")
        with mock.patch.object(platform_api.platform(), "system_ffmpeg", return_value=(on_path / ffmpeg_name, on_path / ffprobe_name)):
            found = self.discover()
            for candidate in found:
                candidate.valid = True
                candidate.sha256 = candidate.source
            registry = FFmpegRegistry()
            registry.data["selected_hash"] = "system"      # an earlier choice of the system build does not stick once a local one exists
            with mock.patch.object(registry, "save"):
                chosen = registry.choose(found, interactive=False)
        self.assertEqual([c.source for c in found], ["portable", "system"])
        self.assertEqual(chosen.source, "portable")

    def test_without_a_build_beside_the_program_the_system_one_is_chosen(self) -> None:
        from modules import platform_api
        from modules.ffmpeg_registry import FFmpegRegistry
        ffmpeg_name, ffprobe_name = platform_api.ffmpeg_executable_names()
        on_path = self.root / "pathdir"
        self._install(on_path)
        with mock.patch.object(platform_api.platform(), "system_ffmpeg", return_value=(on_path / ffmpeg_name, on_path / ffprobe_name)):
            found = self.discover()
            found[0].valid = True
            found[0].sha256 = "abc"
            registry = FFmpegRegistry()
            with mock.patch.object(registry, "save"):
                chosen = registry.choose(found, interactive=False)
        self.assertEqual(chosen.source, "system")


@unittest.skipUnless(HAVE_PIL, "Pillow required")
class LeftoverCleanup(IsolatedApp):
    """Leftover .__old__ / .__failed_finalize__ folders are removed only when that cannot lose the last good output."""

    def setUp(self) -> None:
        super().setUp()
        from modules.tiles import target_dir_for
        self.st = settings()
        self.video = self.media / "a.mkv"
        self.video.write_bytes(b"x")
        self.meta = meta(self.video, size=1, mtime_ns=self.video.stat().st_mtime_ns, duration=170.0)
        self.target = target_dir_for(self.video, self.st)
        self.root = self.target.parent
        self.state = open_state()
        self.addCleanup(self.state.close)

    def make_valid_target(self) -> None:
        write_sprites(self.target, self.st, 3)
        self.state.record_output(self.meta, self.st, 17, 3, 1, "CPU", None, None)

    def folder(self, name: str, count: int = 3, st=None) -> Path:
        d = self.root / name
        write_sprites(d, st or self.st, count)
        return d

    def verify(self):
        from modules.verify import verify_video
        return verify_video(self.meta, self.st, self.state)

    def test_old_copy_and_failed_output_are_removable_when_current_output_is_valid(self) -> None:
        from modules.verify import remove_leftover
        self.make_valid_target()
        old = self.folder("64 - 3x2.__old__.2026-10-26_220000.abc123")
        failed = self.folder("64 - 3x2.__failed_finalize__.2026-10-26_220000", 1)
        row = self.verify()
        self.assertEqual(sorted(i.kind for i in row.leftover_items), ["failed", "old"])
        self.assertTrue(all(i.safe for i in row.leftover_items), [i.reason for i in row.leftover_items])
        for item in row.leftover_items:
            self.assertIsNone(remove_leftover(item))
        self.assertFalse(old.exists() or failed.exists())
        self.assertEqual(self.verify().status, "VALID")

    def test_old_copy_is_kept_when_it_may_be_the_last_good_output(self) -> None:
        from modules.verify import remove_leftover
        old = self.folder("64 - 3x2.__old__.2026-10-26_220000.abc123")   # target is missing
        row = self.verify()
        self.assertEqual(row.status, "MISSING")
        (item,) = row.leftover_items
        self.assertFalse(item.safe)
        self.assertIn("last good output", item.reason)
        self.assertIsNotNone(remove_leftover(item))
        self.assertTrue(old.exists())

    # --- restoring a previous output -------------------------------------------------------------------------------------
    def test_missing_output_is_restored_from_the_old_copy(self) -> None:
        from modules.verify import restore_previous
        old = self.folder("64 - 3x2.__old__.2026-10-26_220000.abc123")
        row = self.verify()
        self.assertEqual(row.status, "MISSING")
        self.assertIsNotNone(row.restorable)
        self.assertIsNone(restore_previous(row.restorable))
        self.assertFalse(old.exists())
        self.assertEqual(self.verify().status, "VALID")
        self.assertEqual(len(list(self.target.glob("*.jpg"))), 3)

    def test_damaged_output_is_moved_aside_not_deleted_when_the_old_copy_is_restored(self) -> None:
        from modules.verify import restore_previous
        write_sprites(self.target, self.st, 2)                       # one tile short: damaged
        self.folder("64 - 3x2.__old__.x.1")
        row = self.verify()
        self.assertEqual(row.status, "INVALID")
        self.assertIsNone(restore_previous(row.restorable))
        self.assertEqual(self.verify().status, "VALID")
        (aside,) = [p for p in self.root.iterdir() if ".__failed_finalize__." in p.name]
        self.assertEqual(len(list(aside.glob("*.jpg"))), 2)

    def test_newest_valid_old_copy_wins(self) -> None:
        self.folder("64 - 3x2.__old__.2026-10-26_100000.aaaaaa")
        newest = self.folder("64 - 3x2.__old__.2026-10-26_230000.bbbbbb")
        broken = self.folder("64 - 3x2.__old__.2026-10-27_010000.cccccc", 1)       # newest of all, but incomplete
        self.assertEqual(self.verify().restorable.path, newest)
        self.assertTrue(broken.exists())

    def test_nothing_to_restore_when_the_old_copy_is_not_a_valid_output(self) -> None:
        self.folder("64 - 3x2.__old__.x.1", 1)
        self.assertIsNone(self.verify().restorable)

    def test_nothing_to_restore_for_a_valid_output_or_other_widths(self) -> None:
        self.make_valid_target()
        self.folder("64 - 3x2.__old__.x.1")
        self.assertIsNone(self.verify().restorable, "the current output is fine")
        (self.target / "1.jpg").unlink()
        shutil.rmtree(self.root / "64 - 3x2.__old__.x.1")
        self.folder("320 - 3x2.__old__.x.1")
        self.assertIsNone(self.verify().restorable, "an old copy of another width is not this target")

    def test_nothing_to_restore_when_the_source_changed(self) -> None:
        self.make_valid_target()
        shutil.rmtree(self.target)
        self.folder("64 - 3x2.__old__.x.1")
        from modules.verify import verify_video
        changed = meta(self.video, size=2, mtime_ns=self.video.stat().st_mtime_ns, duration=170.0)
        row = verify_video(changed, self.st, self.state)
        self.assertEqual(row.status, "SOURCE_CHANGED")
        self.assertIsNone(row.restorable)

    def test_restore_is_refused_when_the_target_holds_other_files(self) -> None:
        from modules.verify import restore_previous
        old = self.folder("64 - 3x2.__old__.x.1")
        row = self.verify()
        self.target.mkdir(parents=True)
        (self.target / "notes.txt").write_text("mine")
        self.assertIn("other than sprite images", restore_previous(row.restorable))
        self.assertTrue(old.exists() and (self.target / "notes.txt").exists())

    def test_restore_is_refused_when_the_old_copy_changed_since_it_was_checked(self) -> None:
        from modules.verify import restore_previous
        old = self.folder("64 - 3x2.__old__.x.1")
        row = self.verify()
        (old / "extra.txt").write_text("x")
        self.assertIsNotNone(restore_previous(row.restorable))
        (old / "extra.txt").unlink()
        (old / "1.jpg").unlink()
        self.assertIn("no longer valid", restore_previous(row.restorable))
        self.assertFalse(self.target.exists())

    @needs_symlinks
    def test_restore_is_refused_when_the_target_is_a_link(self) -> None:
        from modules.verify import restore_previous
        self.folder("64 - 3x2.__old__.x.1")
        row = self.verify()
        elsewhere = self.root.parent / "elsewhere"
        elsewhere.mkdir()
        self.target.symlink_to(elsewhere, target_is_directory=True)
        self.assertIn("link", restore_previous(row.restorable))

    def test_old_copy_is_kept_when_the_current_output_is_damaged(self) -> None:
        self.make_valid_target()
        (self.target / "1.jpg").unlink()
        old = self.folder("64 - 3x2.__old__.x.1")
        row = self.verify()
        self.assertEqual(row.status, "INVALID")
        self.assertFalse(row.leftover_items[0].safe)
        self.assertTrue(old.exists())

    def test_folders_with_other_files_are_never_removable(self) -> None:
        from modules.verify import remove_leftover
        self.make_valid_target()
        failed = self.folder("64 - 3x2.__failed_finalize__.x", 1)
        (failed / "notes.txt").write_text("mine")
        (item,) = self.verify().leftover_items
        self.assertFalse(item.safe)
        self.assertIsNotNone(remove_leftover(item))
        self.assertTrue((failed / "notes.txt").exists())

    def test_removal_rechecks_everything_at_the_last_moment(self) -> None:
        from modules.verify import remove_leftover
        self.make_valid_target()
        old = self.folder("64 - 3x2.__old__.x.1")
        failed = self.folder("64 - 3x2.__failed_finalize__.x", 1)
        items = {i.kind: i for i in self.verify().leftover_items}
        (failed / "surprise.txt").write_text("appeared after the check")
        self.assertIn("changed", remove_leftover(items["failed"]) or "")
        (self.target / "0.jpg").write_bytes(b"")                      # current output got damaged in the meantime
        self.assertIn("no longer valid", remove_leftover(items["old"]) or "")
        self.assertTrue(old.exists() and failed.exists())

    def test_old_copy_of_another_width_depends_on_that_widths_output(self) -> None:
        from modules.verify import remove_leftover
        self.make_valid_target()
        other = settings(width=32)
        write_sprites(self.root / other.folder_name, other, 3)
        gone = self.folder(other.folder_name + ".__old__.x.1", st=other)           # width 32 output is valid -> removable
        write_sprites(self.root / "16 - 3x2.__old__.x.1", settings(width=16), 3)   # no 16-wide output exists -> kept
        items = {i.path.name: i for i in self.verify().leftover_items}
        self.assertTrue(items[gone.name].safe)
        self.assertFalse(items["16 - 3x2.__old__.x.1"].safe)
        self.assertIsNone(remove_leftover(items[gone.name]))

    def test_odd_names_and_wrong_locations_are_refused(self) -> None:
        from modules.verify import Leftover, remove_leftover
        lonely = self.media / "64 - 3x2.__failed_finalize__.x"
        write_sprites(lonely, self.st, 1)
        self.assertIn(".trickplay", remove_leftover(Leftover(lonely, "failed", True, "x")) or "")
        self.assertIn("name", remove_leftover(Leftover(self.root / "whatever", "failed", True, "x")) or "")
        self.assertTrue(lonely.exists())

    def test_zero_dimension_folder_name_does_not_crash(self) -> None:
        self.make_valid_target()
        write_sprites(self.root / "0 - 0x0.__old__.x.1", self.st, 1)
        (item,) = self.verify().leftover_items
        self.assertFalse(item.safe)

    @needs_symlinks
    def test_symlinked_leftover_is_never_followed(self) -> None:
        from modules.verify import remove_leftover
        self.make_valid_target()
        precious = self.root.parent / "precious"
        write_sprites(precious, self.st, 1)
        link = self.root / "64 - 3x2.__failed_finalize__.link"
        link.symlink_to(precious, target_is_directory=True)
        for item in self.verify().leftover_items:
            self.assertFalse(item.safe)
            self.assertIsNotNone(remove_leftover(item))
        self.assertTrue((precious / "0.jpg").exists())

    def test_verify_screen_asks_before_removing_and_logs_each_removal(self) -> None:
        from modules.ui import Application
        self.make_valid_target()
        failed = self.folder("64 - 3x2.__failed_finalize__.x", 1)
        old = self.folder("64 - 3x2.__old__.x.1")
        kept = self.folder("32 - 3x2.__old__.x.1", st=settings(width=32))   # no 32-wide output -> kept
        items = list(self.verify().leftover_items)
        app = Application()
        self.addCleanup(app.state.close)
        logger = new_logger("verify")
        self.addCleanup(logger.close)
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=["n"]), contextlib.redirect_stdout(out):
            app._offer_leftover_cleanup(items, logger)
        self.assertTrue(failed.exists() and old.exists() and kept.exists(), "answering n must change nothing")
        self.assertIn("Safe to remove: 2", out.getvalue())
        self.assertIn("Kept (not safe to remove): 1", out.getvalue())
        with mock.patch("builtins.input", side_effect=["y"]), contextlib.redirect_stdout(out):
            app._offer_leftover_cleanup(items, logger)
        self.assertFalse(failed.exists() or old.exists())
        self.assertTrue(kept.exists() and self.target.exists())
        events = logger.events_path.read_text(encoding="utf-8")
        self.assertEqual(events.count("LEFTOVER_REMOVED"), 2)
        self.assertIn("Removed 2 of 2", out.getvalue())

    def test_nothing_to_offer_asks_nothing(self) -> None:
        from modules.ui import Application
        app = Application()
        self.addCleanup(app.state.close)
        logger = new_logger("verify")
        self.addCleanup(logger.close)
        with mock.patch("builtins.input", side_effect=AssertionError("must not prompt")), contextlib.redirect_stdout(io.StringIO()):
            app._offer_leftover_cleanup([], logger)


class Maintenance(IsolatedApp):
    def test_selection_and_removal(self) -> None:
        logs = constants.DIRS["logs"]
        old, new = logs / "old.log", logs / "new.log"
        old.write_text("o")
        new.write_text("n")
        (logs / ".gitkeep").write_text("")
        sub = logs / "sub"
        sub.mkdir()
        (sub / "f").write_text("x")
        ancient = time.time() - 100 * 86400
        os.utime(old, (ancient, ancient))
        os.utime(sub, (ancient, ancient))
        self.assertEqual(sorted(p.name for p in old_items(logs, 30, include_dirs=False)), ["old.log"])
        self.assertEqual(sorted(p.name for p in old_items(logs, 30, include_dirs=True)), ["old.log", "sub"])
        self.assertEqual(sorted(p.name for p in all_items(logs)), ["new.log", "old.log", "sub"])
        removed, failed = remove_items(old_items(logs, 30, include_dirs=True))
        self.assertEqual((removed, failed), (2, []))
        self.assertTrue(new.exists() and (logs / ".gitkeep").exists() and not sub.exists())

    @needs_symlinks
    def test_symlinked_folders_are_unlinked_not_followed(self) -> None:
        logs = constants.DIRS["logs"]
        precious = self.root / "precious"
        precious.mkdir()
        (precious / "keep.txt").write_text("keep")
        (logs / "link").symlink_to(precious)
        removed, failed = remove_items([logs / "link"])
        self.assertEqual((removed, failed), (1, []))
        self.assertTrue((precious / "keep.txt").exists())


class FrameCountDeviation(unittest.TestCase):
    """The number of thumbnails must fit the length of the video; the real files from the first library run are the examples."""

    def test_the_usual_difference_of_a_frame_or_two_is_fine(self) -> None:
        for frames, duration in ((255, 2553.215), (254, 2545.983), (576, 5800.0), (577, 5800.0), (25, 250.0), (3, 30.0)):
            self.assertIsNone(frame_count_deviation(frames, duration, 10.0), (frames, duration))

    def test_real_files_with_a_wrong_count_are_found(self) -> None:
        self.assertEqual(frame_count_deviation(524, 2545.983, 10.0), (255, 524))        # damaged time stamps: pictures repeated
        self.assertEqual(frame_count_deviation(449, 5591.466, 10.0), (559, 449))        # video stream ends before the container says
        self.assertEqual(frame_count_deviation(776, 8425.192, 10.0), (843, 776))
        self.assertEqual(frame_count_deviation(663, 6840.0, 10.0), (684, 663))

    def test_unknown_length_or_interval_is_never_suspect(self) -> None:
        self.assertIsNone(frame_count_deviation(5, 0.0, 10.0))
        self.assertIsNone(frame_count_deviation(5, 100.0, 0.0))

    def test_description_says_what_is_wrong(self) -> None:
        self.assertIn("short", describe_frame_deviation(559, 449, 10.0))
        self.assertIn("4,490 s", describe_frame_deviation(559, 449, 10.0))
        self.assertIn("too many", describe_frame_deviation(255, 524, 10.0))


class ParseFrameRate(unittest.TestCase):
    def test_fractions_from_ffprobe(self) -> None:
        self.assertAlmostEqual(parse_frame_rate("13978/583"), 23.976, places=3)
        self.assertAlmostEqual(parse_frame_rate("30000/1001"), 29.97, places=2)
        self.assertEqual(parse_frame_rate("25/1"), 25.0)
        self.assertEqual(parse_frame_rate("24"), 24.0)

    def test_unusable_rates_are_none(self) -> None:
        for text in ("0/0", "0/1", "1/0", "", "N/A", None, "-5/1"):
            self.assertIsNone(parse_frame_rate(text), text)


if __name__ == "__main__":
    unittest.main()


class PromptHelpers(unittest.TestCase):
    def run_choose(self, entries, answers, default=None):
        from modules.prompts import choose
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=list(answers)), contextlib.redirect_stdout(out):
            return choose(entries, default), out.getvalue()

    def test_choose_prints_a_vertical_menu_and_returns_the_key(self) -> None:
        key, out = self.run_choose([("1", "Quick"), ("2", "Extended"), ("0", "Back")], ["2"], "1")
        self.assertEqual(key, "2")
        self.assertEqual(out, "\n[1] Quick\n[2] Extended\n[0] Back\n\n")

    def test_choose_uses_the_default_on_enter_and_asks_again_on_nonsense(self) -> None:
        self.assertEqual(self.run_choose([("1", "A"), ("2", "B")], [""], "2")[0], "2")
        key, out = self.run_choose([("1", "A"), ("2", "B")], ["9", "x", "1"], "2")
        self.assertEqual(key, "1")
        self.assertEqual(out.count("Choose one of"), 2)

    def test_pause_waits_for_enter(self) -> None:
        from modules.prompts import pause
        with mock.patch("builtins.input", return_value="") as asked, contextlib.redirect_stdout(io.StringIO()):
            pause()
        self.assertEqual([c.args[0] for c in asked.call_args_list], ["\nPress Enter to continue..."])


class MainMenu(IsolatedApp):
    """Same behavior as the sibling programs: [0] leaves, no default choice, an empty answer only redraws the screen."""

    def run_menu(self, answers: list[str], **patched):
        from modules.ui import Application
        app = Application()
        self.addCleanup(app.state.close)
        out = io.StringIO()
        clears: list[int] = []
        prompts: list[str] = []
        queue = list(answers)

        def fake_input(prompt: str = "") -> str:
            prompts.append(prompt)
            if not queue: raise AssertionError(f"menu asked for more input than expected: {prompt!r}")
            answer = queue.pop(0)
            if isinstance(answer, BaseException): raise answer           # Ctrl+C at this prompt
            return answer

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch("builtins.input", side_effect=fake_input))
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(mock.patch("modules.platform_api.clear_screen", side_effect=lambda: clears.append(1)))
            for name, value in patched.items():
                stack.enter_context(mock.patch.object(Application, name, side_effect=value))
            app.main_menu()
        self.assertEqual(queue, [], "unused answers left over")
        return out.getvalue(), len(clears), prompts

    def test_the_header_says_where_you_are_in_nested_menus(self) -> None:
        text, _, _ = self.run_menu(["4", "2", "3", "0", "0", "0", "0"])
        for where in (" Configuration\n", " Configuration > Trickplay settings\n", " Configuration > Trickplay settings > Key frames\n"):
            self.assertIn(where, text)
        self.assertEqual(text.count(" MEDIA TRICKPLAY GENERATOR"), text.count("=" * 72) // 2, "every header has the program name once")
        self.assertNotIn("Main menu", text)
        self.assertNotIn(constants.APP_VERSION, text)

    def test_the_main_menu_header_shows_the_tagline(self) -> None:
        text, _, _ = self.run_menu(["0"])
        self.assertIn(" MEDIA TRICKPLAY GENERATOR\n" + "-" * 72 + "\n Let your PC suffer so your movie doesn't buffer\n" + "=" * 72, text)

    def test_the_header_never_gets_wider_than_the_line(self) -> None:
        from modules.ui import Application
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            Application.show_header("Configuration > Processing > " + "Hardware acceleration > " * 8 + "Key frames")
        lines = out.getvalue().splitlines()
        self.assertTrue(all(len(line) <= 72 for line in lines), lines)
        self.assertTrue(lines[3].startswith(" ...") and lines[3].endswith("Key frames"), "a long path keeps its end")

    def test_ctrl_c_in_a_submenu_goes_one_level_back(self) -> None:
        text, _, _ = self.run_menu(["4", KeyboardInterrupt(), "0"])
        self.assertEqual(text.count(" Let your PC suffer so your movie doesn't buffer\n"), 2, "back in the main menu, which was then left with 0")

    def test_ctrl_c_in_the_main_menu_ends_the_program(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            self.run_menu([KeyboardInterrupt()])

    def test_ctrl_c_during_an_action_ends_only_that_action(self) -> None:
        text, _, _ = self.run_menu(["4", "1", "", "0", "0"], configure_library=KeyboardInterrupt())
        self.assertIn("Canceled.", text)
        self.assertEqual(text.count("\n Configuration\n"), 2, "the Configuration menu is drawn again after the cancelled action")

    def test_main_menu_is_short_and_exit_is_zero(self) -> None:
        text, clears, _ = self.run_menu(["0"])
        self.assertEqual(clears, 1)
        for line in ("[1] Generate / Resume trickplay", "[2] Verify existing trickplay", "[3] Benchmark", "[4] Configuration", "[5] Maintenance", "[0] Exit"):
            self.assertIn(line, text)
        self.assertNotIn("[6]", text)
        self.assertGreater(text.index("[0] Exit"), text.index("[5] Maintenance"))

    def test_prompt_has_no_default_value(self) -> None:
        _, _, prompts = self.run_menu(["0"])
        self.assertEqual(prompts, ["Choice: "])

    def test_enter_alone_leaves_a_submenu_but_never_the_main_menu(self) -> None:
        # Benchmark + Enter, Configuration + Enter, Maintenance + Enter, then [0] in the main menu
        _, _, prompts = self.run_menu(["3", "", "4", "", "5", "", "0"])
        main, sub = "Choice: ", "Choice [0]: "
        self.assertEqual(prompts, [main, sub, main, sub, main, sub, main])

    def test_empty_input_only_redraws_the_screen(self) -> None:
        text, clears, _ = self.run_menu(["", "   ", "0"])
        self.assertEqual(clears, 3)
        self.assertNotIn("Invalid selection", text)

    def test_invalid_input_is_rejected_without_redrawing(self) -> None:
        text, clears, _ = self.run_menu(["abc", "6", "-1", "0"])
        self.assertEqual(clears, 1)
        self.assertEqual(text.count("Invalid selection"), 3)

    def test_main_items_run_their_screens(self) -> None:
        calls: list[str] = []
        patched = {n: (lambda *a, n=n, **k: calls.append(n)) for n in ("generate", "verify")}
        self.run_menu(["1", "2", "0"], **patched)
        self.assertEqual(calls, ["generate", "verify"])

    def test_benchmark_submenu(self) -> None:
        calls: list[str] = []
        patched = {"run_diagnostic": lambda *a, **k: calls.append("diag:" + a[-1]),
                   "run_decode_check": lambda *a, **k: calls.append("decode")}
        text, clears, _ = self.run_menu(["3", "1", "2", "3", "0", "0"], **patched)
        self.assertEqual(calls, ["diag:benchmark", "diag:hdr", "decode"])
        for line in ("[1] CPU vs GPU benchmark", "[2] HDR compatibility check", "[3] Backend compatibility check", "[0] Back"):
            self.assertIn(line, text)
        self.assertIn(" Benchmark\n", text)
        self.assertNotIn("Self-test", text)

    def test_self_test_is_the_first_entry_of_maintenance(self) -> None:
        calls: list[str] = []
        text, _, _ = self.run_menu(["5", "1", "0", "0"], self_test=lambda *a, **k: calls.append("self_test"))
        self.assertEqual(calls, ["self_test"])
        self.assertIn("[1] Self-test", text)

    def test_maintenance_is_in_the_main_menu_and_keeps_results_visible(self) -> None:
        calls: list[str] = []
        text, _, prompts = self.run_menu(["5", "2", "", "0", "0"], _show_usage=lambda *a, **k: calls.append("usage"))
        self.assertEqual(calls, ["usage"])
        self.assertIn("[5] Maintenance", text)
        self.assertIn(" Maintenance\n", text)
        self.assertEqual(prompts.count("\nPress Enter to continue..."), 1)

    def test_actions_that_wait_for_enter_themselves_are_not_paused_twice(self) -> None:
        _, _, prompts = self.run_menu(["1", "0"], generate=lambda *a, **k: None)
        self.assertNotIn("\nPress Enter to continue...", prompts)

    def test_every_configuration_entry_is_reachable(self) -> None:
        calls: list[str] = []
        names = ["configure_roots", "configure_excluded_folders", "configure_preset", "configure_trickplay_image", "configure_key_frames", "configure_existing_outputs",
                 "configure_acceleration", "configure_storage_workers", "configure_fingerprint"]
        patched = {n: (lambda *a, n=n, **k: calls.append(n)) for n in names}
        # Library: roots (pauses), excluded; the other screens return to their menu at once
        self.run_menu(["4", "1", "1", "", "2", "0", "2", "1", "2", "3", "4", "0", "3", "1", "2", "3", "0", "0", "0"], **patched)
        self.assertEqual(calls, names)

    def test_maintenance_delete_goes_through_the_menu_and_asks_for_confirmation(self) -> None:
        logs = constants.DIRS["logs"]
        (logs / "a.log").write_text("x")
        (logs / "b.log").write_text("y")
        # Maintenance -> [4] Clean logs and diagnostics -> [3] Everything: "n" keeps them, "y" removes them
        self.run_menu(["5", "4", "3", "n", "0", "0"])           # declined: back in the menu at once, no Enter
        self.assertTrue((logs / "a.log").exists() and (logs / "b.log").exists())
        self.run_menu(["5", "4", "3", "y", "", "0", "0"])
        remaining = [p.name for p in logs.iterdir() if p.suffix == ".log" and p.name in {"a.log", "b.log"}]
        self.assertEqual(remaining, [])

    def test_maintenance_questions_answered_with_no_or_back_need_no_enter(self) -> None:
        self.run_menu(["5", "4", "0", "0", "0"])            # Clean logs: Back
        self.run_menu(["5", "5", "n", "0", "0"])            # Reset processing history: no
        self.run_menu(["5", "6", "n", "0", "0"])            # Forget FFmpeg build history: no

    def test_maintenance_reset_history_is_confirmed_before_it_runs(self) -> None:
        from modules.ui import Application
        calls: list[str] = []
        with mock.patch.object(Application, "_reset_processing_history", side_effect=lambda: calls.append("reset")):
            self.run_menu(["5", "5", "", "0", "0"])
        self.assertEqual(calls, ["reset"])
