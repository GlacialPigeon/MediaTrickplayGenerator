# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""The interactive screens, driven with scripted keyboard input.

Every test lists the exact answers the screen is going to get. A screen that asks one more question than expected fails
("unexpected prompt"), and so does one that leaves answers unused - so a changed flow is noticed, not silently accepted.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import unittest
from pathlib import Path
from typing import Callable
from unittest import mock

from tests.helpers import IsolatedApp, make_video, needs_ffmpeg
from tests.test_pipeline import PipelineBase

from modules import constants, platform_api
from modules.config import ConfigManager
from modules.tiles import target_dir_for
from modules.verify import VALID, verify_video


def problem_list() -> Path:
    """The problems.txt of the latest Verify run (each run has a folder of its own in logs/)."""
    found = sorted(constants.DIRS["logs"].glob("*/problems.txt"), key=lambda path: path.stat().st_mtime)
    return found[-1] if found else constants.DIRS["logs"] / "problems.txt"


class ScriptedInput:
    """Replaces input() with a list of answers and collects what the screen printed and asked."""

    def __init__(self, test: unittest.TestCase, answers: list[str]) -> None:
        self.test, self.answers = test, list(answers)
        self.prompts: list[str] = []
        self.out = io.StringIO()

    def _input(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"unexpected prompt {prompt!r}; asked so far: {self.prompts}")
        return self.answers.pop(0)

    def run(self, fn: Callable[[], object]) -> str:
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch("builtins.input", side_effect=self._input))
            stack.enter_context(mock.patch.object(platform_api, "clear_screen"))
            stack.enter_context(contextlib.redirect_stdout(self.out))
            fn()
        self.test.assertEqual(self.answers, [], f"answers not used: {self.answers}; prompts: {self.prompts}")
        return self.out.getvalue()


class ScreenTest(IsolatedApp):
    def setUp(self) -> None:
        super().setUp()
        from modules.ui import Application
        self.app = Application()
        self.addCleanup(self.app.state.close)

    def feed(self, fn: Callable[[], object], *answers: str) -> str:
        return ScriptedInput(self, list(answers)).run(fn)

    def saved(self) -> dict:
        """The configuration as it is on disk (what the next start will see)."""
        return ConfigManager().load()


class MediaRootsScreen(ScreenTest):
    def test_manual_entry_is_cleaned_up_and_saved(self) -> None:
        other = self.root / "other"
        other.mkdir()
        offline = self.root / "offline_drive"                              # not there right now: kept, a drive may just be unplugged
        out = self.feed(self.app.configure_roots, "1",
                        str(self.media), f'"{other}"', str(self.media), str(self.media / "sub"),
                        str(constants.APP_ROOT / "inside"), str(offline), "")
        self.assertEqual(self.saved()["media_roots"], [str(self.media), str(other), str(offline)])
        self.assertIn("Saved 3 media folders.", out)
        self.assertIn("Ignoring duplicate folder", out)
        self.assertIn("Ignoring nested media folder", out)
        self.assertIn("inside the program folder", out)

    def test_a_drive_that_holds_the_program_is_saved_and_says_the_program_folder_is_skipped(self) -> None:
        out = self.feed(self.app.configure_roots, "1", str(constants.APP_ROOT.parent), "")
        self.assertEqual(self.saved()["media_roots"], [str(constants.APP_ROOT.parent)])
        self.assertIn("contains the program folder; it is skipped when scanning.", out)
        self.assertNotIn("inside the program folder", out)

    def test_the_excluded_folders_screen_names_what_is_always_skipped(self) -> None:
        out = self.feed(self.app.configure_excluded_folders, "0")
        self.assertIn("Nothing is excluded.", out)
        self.assertIn("Always skipped, whatever is listed:", out)
        self.assertIn(f"the program folder: {constants.APP_ROOT}\n", out, "the whole path, not a dot")
        self.assertIn("$RECYCLE.BIN and System Volume Information", out)
        self.assertIn(".trickplay", out)

    def test_roots_from_a_text_file(self) -> None:
        listing = self.root / "roots.txt"
        listing.write_text(f"# my libraries\n{self.media}\n\n", encoding="utf-8")
        self.feed(self.app.configure_roots, "2", str(listing))
        self.assertEqual(self.saved()["media_roots"], [str(self.media)])

    def test_unreadable_text_file_changes_nothing(self) -> None:
        self.app.config["media_roots"] = [str(self.media)]
        out = self.feed(self.app.configure_roots, "2", str(self.root / "missing.txt"))
        self.assertIn("Could not read TXT file", out)
        self.assertEqual(self.app.config["media_roots"], [str(self.media)])

    def test_back_and_keep_change_nothing(self) -> None:
        self.app.config["media_roots"] = [str(self.media)]
        self.feed(self.app.configure_roots, "0")
        self.feed(self.app.configure_roots, "")           # Enter = Back (the default when roots exist)
        self.assertEqual(self.app.config["media_roots"], [str(self.media)])

    def test_invalid_menu_answer_is_asked_again(self) -> None:
        out = self.feed(self.app.configure_roots, "7", "x", "0")
        self.assertIn("Choose one of", out)


class ExcludedFoldersScreen(ScreenTest):
    def test_names_are_added_cleaned_and_saved(self) -> None:
        out = self.feed(self.app.configure_excluded_folders, "1",
                        "extras", ' "Trailers" ', "EXTRAS", "a/b", "C:\\x", "movie.trickplay", "..", "samples/", "",    # then back to the menu
                        "0")
        self.assertEqual(self.saved()["excluded_folders"], ["extras", "Trailers", "samples"])
        self.assertIn("Already in the list", out)
        self.assertIn("not a path", out)
        self.assertIn("always skipped", out)
        self.assertIn("not a folder name", out)

    def test_some_or_all_can_be_removed(self) -> None:
        self.app.config["excluded_folders"] = ["a", "b", "c", "d"]
        self.feed(self.app.configure_excluded_folders, "2", "2-3", "0")
        self.assertEqual(self.saved()["excluded_folders"], ["a", "d"])
        self.feed(self.app.configure_excluded_folders, "2", "", "0")                  # nothing picked: nothing removed
        self.assertEqual(self.saved()["excluded_folders"], ["a", "d"])
        self.feed(self.app.configure_excluded_folders, "3", "n", "0")
        self.assertEqual(self.saved()["excluded_folders"], ["a", "d"])
        self.feed(self.app.configure_excluded_folders, "3", "y", "0")
        self.assertEqual(self.saved()["excluded_folders"], [])

    def test_enter_goes_back_and_changes_nothing(self) -> None:
        self.app.config["excluded_folders"] = ["a"]
        out = self.feed(self.app.configure_excluded_folders, "")
        self.assertIn("Excluded folders", out)
        self.assertEqual(self.app.config["excluded_folders"], ["a"])

    def test_an_empty_list_offers_only_adding(self) -> None:
        out = self.feed(self.app.configure_excluded_folders, "0")
        self.assertIn("Nothing is excluded.", out)
        self.assertNotIn("Remove all", out)


class TrickplaySettingsScreen(ScreenTest):
    def test_image_values_are_validated_asked_again_and_saved(self) -> None:
        out = self.feed(self.app.configure_trickplay_image,
                        "1", "abc", "50", "5000", "n", "320",     # width: not a number, too small, too big, not a number again, then ok
                        "2", "0", "x", "7.5",                     # interval: out of range, not a number, ok
                        "0")
        t = self.saved()["trickplay"]
        self.assertEqual((t["width"], t["interval_seconds"]), (320, 7.5))
        self.assertIn("Enter a whole number", out)
        self.assertIn("Allowed range: 160..2048", out)
        self.assertIn("Enter a number", out)

    def test_unusually_high_value_must_be_confirmed(self) -> None:
        self.feed(self.app.configure_trickplay_image, "1", "1500", "n", "1200", "y", "0")
        self.assertEqual(self.saved()["trickplay"]["width"], 1200)

    def test_every_image_setting_has_its_own_entry(self) -> None:
        self.feed(self.app.configure_trickplay_image, "3", "12", "4", "8", "5", "5", "6", "70", "0")
        t = self.saved()["trickplay"]
        self.assertEqual((t["tile_columns"], t["tile_rows"], t["ffmpeg_qscale"], t["jpeg_quality"]), (12, 8, 5, 70))

    def test_key_frames_only_is_off_by_default_and_kept_when_answered_with_enter(self) -> None:
        self.assertIs(self.app.config["trickplay"]["keyframe_only"], False)
        out = self.feed(self.app.configure_key_frames, "1", "", "0")
        self.assertIs(self.saved()["trickplay"]["keyframe_only"], False)
        self.assertNotIn("Switch to every frame", out, "only offered when key frames are on")

    def test_key_frames_on_offers_how_many_repeats_bring_back_every_frame(self) -> None:
        self.feed(self.app.configure_key_frames, "1", "y", "2", "15", "0")
        t = self.saved()["trickplay"]
        self.assertEqual((t["keyframe_only"], t["keyframe_repeat_limit_percent"]), (True, 15.0))
        out = self.feed(self.app.configure_key_frames, "1", "n", "0")
        self.assertIs(self.saved()["trickplay"]["keyframe_only"], False)
        self.assertNotIn("Switch to every frame", out.split("Key frames only: off")[-1])

    def test_existing_output_policies(self) -> None:
        self.feed(self.app.configure_existing_outputs, "1", "2", "2", "2", "0")
        t = self.saved()["trickplay"]
        self.assertEqual((t["same_width_policy"], t["other_width_policy"]), ("rebuild_all", "replace"))

    def test_enter_everywhere_keeps_the_current_settings(self) -> None:
        before = dict(self.app.config["trickplay"])
        self.feed(self.app.configure_trickplay_image, "")
        self.feed(self.app.configure_key_frames, "1", "", "0")
        self.feed(self.app.configure_existing_outputs, "1", "", "2", "", "0")
        self.assertEqual(self.app.config["trickplay"], before)

    def test_the_menu_that_leads_on_shows_names_only(self) -> None:
        self.app.config["trickplay"].update(width=400, keyframe_only=True)
        out = self.feed(self.app.configure_trickplay, "")
        for entry in ("[1] Preset\n", "[2] Image\n", "[3] Key frames\n", "[4] Existing outputs\n"):
            self.assertIn(entry, out)
        self.assertNotIn("400", out)

    def test_the_last_menus_show_the_values_that_are_set(self) -> None:
        self.app.config["trickplay"].update(width=400, interval_seconds=7.5, keyframe_only=True, keyframe_repeat_limit_percent=15.0, other_width_policy="replace")
        image = self.feed(self.app.configure_trickplay_image, "")
        for entry in ("[1] Thumbnail width: 400 px", "[2] Interval: 7.5 s", "[3] Tile columns: 10", "[4] Tile rows: 10", "[5] FFmpeg JPEG qscale: 4", "[6] Final JPEG quality: 90"):
            self.assertIn(entry, image)
        key_frames = self.feed(self.app.configure_key_frames, "")
        self.assertIn("[1] Key frames only: on", key_frames)
        self.assertIn("[2] Switch to every frame above: 15 % repeated", key_frames)
        outputs = self.feed(self.app.configure_existing_outputs, "")
        self.assertIn("[1] Same width already exists: skip valid", outputs)
        self.assertIn("[2] Other width already exists: replace after success", outputs)

    def test_the_menu_shows_a_changed_value_at_once(self) -> None:
        out = self.feed(self.app.configure_trickplay_image, "1", "400", "0")
        self.assertIn("[1] Thumbnail width: 320 px", out)
        self.assertIn("[1] Thumbnail width: 400 px", out)

    def test_the_default_is_the_jellyfin_preset(self) -> None:
        t = self.app.config["trickplay"]
        self.assertEqual((t["width"], t["interval_seconds"], t["tile_columns"], t["tile_rows"], t["ffmpeg_qscale"], t["jpeg_quality"]), (320, 10.0, 10, 10, 4, 90))
        self.assertIn("\nNow: Jellyfin default\n", self.feed(self.app.configure_preset, ""))

    def test_the_preset_replaces_only_the_image_settings(self) -> None:
        t = self.app.config["trickplay"]
        t.update(width=640, interval_seconds=5.0, jpeg_quality=70, keyframe_only=True, other_width_policy="replace")
        out = self.feed(self.app.configure_preset, "1", "0")
        self.assertIn("\nNow: custom\n", out)
        self.assertIn("\nNow: Jellyfin default\n", out, "shown again at once")
        saved = self.saved()["trickplay"]
        self.assertEqual((saved["width"], saved["interval_seconds"], saved["jpeg_quality"]), (320, 10.0, 90))
        self.assertEqual((saved["keyframe_only"], saved["other_width_policy"]), (True, "replace"), "everything else stays")

    def test_going_back_from_the_preset_changes_nothing(self) -> None:
        self.app.config["trickplay"]["width"] = 640
        self.feed(self.app.configure_preset, "")                       # Enter = Back
        self.assertEqual(self.app.config["trickplay"]["width"], 640)
        self.feed(self.app.configure_preset, "0")
        self.assertEqual(self.app.config["trickplay"]["width"], 640)

    def test_a_configuration_saved_before_keeps_its_width(self) -> None:
        self.app.config["trickplay"]["width"] = 640
        self.app.config_mgr.save()
        from modules.ui import Application
        reopened = Application()
        self.addCleanup(reopened.state.close)
        self.assertEqual(reopened.config["trickplay"]["width"], 640)


class AccelerationScreen(ScreenTest):
    def test_mode_limits_and_threshold_are_saved(self) -> None:
        self.feed(self.app.configure_acceleration, "1", "3", "2", "3", "3", "50", "0")      # mode: CPU only; 3 jobs; 50 % faster
        a = self.saved()["acceleration"]
        self.assertEqual((a["mode"], a["max_gpu_jobs"], a["gpu_speed_threshold"]), ("cpu", 3, 1.5))

    def test_out_of_range_values_are_rejected(self) -> None:
        out = self.feed(self.app.configure_acceleration, "2", "99", "0", "2", "3", "999", "20", "0")
        a = self.saved()["acceleration"]
        self.assertEqual((a["mode"], a["max_gpu_jobs"], a["gpu_speed_threshold"]), ("auto", 2, 1.2))
        self.assertIn("Allowed range: 1..16", out)
        self.assertIn("Allowed range: 1..200", out)

    def test_the_menu_shows_the_values_that_are_set(self) -> None:
        self.app.config["acceleration"].update(mode="gpu", max_gpu_jobs=3, gpu_speed_threshold=1.15)
        out = self.feed(self.app.configure_acceleration, "")
        for entry in ("[1] Mode: gpu", "[2] Videos on the GPU at once: 3", "[3] GPU must be faster by: 15 %"):
            self.assertIn(entry, out)
        self.assertNotIn("GPU backend", out, "no GPU path to choose from")

    def test_preferred_backend_is_chosen_from_the_available_ones(self) -> None:
        self.app.tools = mock.Mock()
        self.app.tools.backends.return_value = ["cuda", "qsv"]
        out = self.feed(self.app.configure_acceleration, "2", "3", "0")
        self.assertEqual(self.saved()["acceleration"]["preferred_backend"], "qsv")
        self.assertIn("[2] GPU backend: qsv", out)
        self.feed(self.app.configure_acceleration, "2", "1", "0")
        self.assertEqual(self.saved()["acceleration"]["preferred_backend"], "auto")


class SmallConfigurationScreens(ScreenTest):
    def test_fingerprint_toggle(self) -> None:
        out = self.feed(self.app.configure_fingerprint, "1", "0")
        self.assertTrue(self.saved()["source_detection"]["strong_fingerprint"])
        self.assertIn("each video: on", out)
        self.feed(self.app.configure_fingerprint, "")      # Enter goes back and keeps the setting
        self.assertTrue(self.saved()["source_detection"]["strong_fingerprint"])
        self.feed(self.app.configure_fingerprint, "1", "0")
        self.assertFalse(self.saved()["source_detection"]["strong_fingerprint"])

    def test_storage_workers_manual_groups_get_safe_names(self) -> None:
        self.app.config["media_roots"] = [str(self.media), str(self.root / "b")]
        self.feed(self.app.configure_storage_workers, "1", "3", "my disk/1", "", "2", "2", "2", "0")   # a fixed number: the same on every disk
        w = self.saved()["workers"]
        self.assertEqual(w["storage_mode"], "manual")
        self.assertEqual(w["manual_groups"], {"my_disk_1": [str(self.media)], "disk2": [str(self.root / "b")]})
        self.assertEqual(w["per_storage_group"], 2)

    def test_manual_storage_without_roots_falls_back_to_auto(self) -> None:
        out = self.feed(self.app.configure_storage_workers, "1", "3", "", "0")      # the message waits for Enter
        self.assertEqual(self.saved()["workers"]["storage_mode"], "auto")
        self.assertIn("No media folders are configured.", out)

    def test_the_storage_menu_shows_the_values_that_are_set(self) -> None:
        self.app.config["workers"].update(per_storage_group=2, storage_mode="drive")
        out = self.feed(self.app.configure_storage_workers, "")
        self.assertIn("[1] Disks: each drive or share separate", out)
        self.assertIn("[2] Videos at once per disk: 2", out)


class ConfigurationMenu(ScreenTest):
    def test_entry_opens_the_screen_and_returns_to_the_menu(self) -> None:
        # 2 = Trickplay settings, 2 = Image, 1 = width, 400, then back out of the three menus
        self.feed(self.app.configuration_menu, "2", "2", "1", "400", "0", "0", "0")
        self.assertEqual(self.saved()["trickplay"]["width"], 400)

    def test_entry_that_pauses_waits_for_enter_before_the_menu_returns(self) -> None:
        self.app.config["media_roots"] = []
        # 1 = Library, 1 = Media folders, 1 = enter paths, a blank line, then the menu's own "Press Enter", then [0] twice
        out = self.feed(self.app.configuration_menu, "1", "1", "1", "", "", "0", "0")
        self.assertIn("Media folders", out)

    def test_going_back_from_a_screen_does_not_wait_for_enter(self) -> None:
        out = self.feed(self.app.configuration_menu, "1", "1", "0", "0", "0")       # Library, Media folders, Back, then back out of both menus
        self.assertIn("Media folders", out)

    def test_every_menu_entry_opens_its_screen(self) -> None:
        paths = {"configure_roots": ["1", "1", "", "0", "0"], "configure_excluded_folders": ["1", "2", "0", "0"],
                 "configure_preset": ["2", "1", "0", "0"], "configure_trickplay_image": ["2", "2", "0", "0"],
                 "configure_key_frames": ["2", "3", "0", "0"], "configure_existing_outputs": ["2", "4", "0", "0"], "configure_acceleration": ["3", "1", "0", "0"],
                 "configure_storage_workers": ["3", "2", "0", "0"], "configure_fingerprint": ["3", "3", "0", "0"]}
        for name, answers in paths.items():
            with self.subTest(screen=name), mock.patch.object(type(self.app), name) as screen:
                self.feed(self.app.configuration_menu, *answers)
                screen.assert_called_once_with()

    def test_invalid_entry_is_rejected_and_enter_goes_back(self) -> None:
        out = self.feed(self.app.configuration_menu, "99", "")
        self.assertIn("Invalid selection", out)

    def test_menus_that_lead_on_show_names_only(self) -> None:
        self.app.config.update(media_roots=["D:\\Media", "X:\\Movies"], excluded_folders=["extras"])
        self.app.config["acceleration"].update(mode="gpu")
        out = self.feed(self.app.configuration_menu, "")
        for entry in ("[1] Library\n", "[2] Trickplay settings\n", "[3] Processing\n"):
            self.assertIn(entry, out)
        self.assertNotIn("Maintenance", out)
        self.assertNotIn("gpu", out)
        library = self.feed(self.app.configure_library, "")
        self.assertIn("[1] Media folders\n", library)
        self.assertIn("[2] Excluded folders\n", library)
        self.assertNotIn("Movies", library)
        processing = self.feed(self.app.configure_processing, "")
        for entry in ("[1] Hardware acceleration\n", "[2] Videos at once\n", "[3] Detecting changed videos\n"):
            self.assertIn(entry, processing)


class HdrReviewScreen(ScreenTest):
    def rows(self) -> list[dict]:
        return [{"backend": "cpu", "hdr_label": "HDR10", "status": "PASS"}, {"backend": "cuda", "hdr_label": "HDR10", "status": "PASS"},
                {"backend": "cpu", "hdr_label": "Dolby Vision P8", "status": "PASS"}, {"backend": "cuda", "hdr_label": "Dolby Vision P8", "status": "PASS"},
                {"backend": "cuda", "hdr_label": "HLG", "status": "FAILED"}]

    def test_one_yes_rates_every_path_of_a_kind_of_hdr_as_correct(self) -> None:
        data = {"rows": self.rows(), "visual": {}}
        self.feed(lambda: self.app.review_hdr_images(data, self.root, lambda: None), "n", "", "n", "1", "3")      # HDR10: Enter = all correct; Dolby Vision: no, cpu ok, cuda too dark
        self.assertEqual(data["visual"], {"HDR10 | cpu": "ACCEPTABLE", "HDR10 | cuda": "ACCEPTABLE",
                                          "Dolby Vision P8 | cpu": "ACCEPTABLE", "Dolby Vision P8 | cuda": "TOO_DARK"})

    def test_every_kind_and_path_is_assessed_separately(self) -> None:
        data = {"rows": self.rows(), "visual": {}}
        saved: list[int] = []
        # Open the folder? no; HDR10: not all correct, then cpu ok and cuda too bright; Dolby Vision: not all correct, cpu ok, cuda wrong colors
        out = self.feed(lambda: self.app.review_hdr_images(data, self.root, lambda: saved.append(1)), "n", "n", "1", "2", "n", "1", "4")
        self.assertEqual(data["visual"], {"HDR10 | cpu": "ACCEPTABLE", "HDR10 | cuda": "TOO_BRIGHT",
                                          "Dolby Vision P8 | cpu": "ACCEPTABLE", "Dolby Vision P8 | cuda": "WRONG_COLORS"})
        self.assertNotIn("HLG", out, "a failed run has no images to review")
        self.assertTrue(saved)

    def test_reports_from_before_the_per_kind_test_still_open(self) -> None:
        data = {"rows": [{"backend": "cpu", "status": "PASS"}, {"backend": "cuda", "status": "PASS"}], "visual": {"cuda": "TOO_DARK"}}
        self.feed(lambda: self.app.review_hdr_images(data, self.root, lambda: None), "n", "1", "")      # Enter keeps the saved answer for cuda
        self.assertEqual(data["visual"], {"cpu": "ACCEPTABLE", "cuda": "TOO_DARK"})


@needs_ffmpeg
class GenerateAndVerifyScreens(PipelineBase):
    def setUp(self) -> None:
        super().setUp()
        from modules.ui import Application
        self.app = Application()
        self.addCleanup(self.app.state.close)
        self.app.state.open()
        self.app.config = self.config
        self.app.tools = self.tools
        self.app.ffmpeg_candidate = self.tools.candidate
        patcher = mock.patch.object(type(self.app), "ensure_ffmpeg", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        make_video(self.media / "a.mp4", 25)
        self.settings = self.generator().settings

    def feed(self, fn: Callable[[], object], *answers: str) -> str:
        return ScriptedInput(self, list(answers)).run(fn)

    def target(self) -> Path:
        return target_dir_for(self.scan()[0].path, self.settings)

    def test_generate_asks_for_confirmation_then_pauses_on_the_summary(self) -> None:
        out = self.feed(self.app.generate, "y", "")       # Proceed? -> yes; Press Enter after FINAL SUMMARY
        self.assertIn("PREFLIGHT", out)
        self.assertIn("FINAL SUMMARY", out)
        self.assertEqual(verify_video(self.scan()[0], self.settings, self.app.state).status, VALID)

    def test_the_preflight_lists_jobs_workers_and_time_and_ends_with_the_one_question(self) -> None:
        with mock.patch("modules.storage.storage_kind", return_value="hdd"):
            out = self.feed(self.app.generate, "y", "")
        for label in ("  Videos on the GPU at once:", "  Videos at once per disk:", "  Time:", "  Videos:", "  Disk:"):
            self.assertIn(label, out)
        self.assertRegex(out, r"Videos at once per disk:\s+\S+ HDD 1")
        self.assertRegex(out, r"Time:\s+(about|less than)")
        self.assertLess(out.index("PREFLIGHT"), out.index("FINAL SUMMARY"))

    def test_the_preflight_gives_an_ssd_two_videos_at_once_and_a_number_set_by_hand_wins(self) -> None:
        with mock.patch("modules.storage.storage_kind", return_value="ssd"):
            self.assertRegex(self.feed(self.app.generate, "n"), r"Videos at once per disk:\s+\S+ SSD 2")
            self.app.config["workers"]["per_storage_group"] = 3
            self.assertRegex(self.feed(self.app.generate, "n"), r"Videos at once per disk:\s+\S+ SSD 3")

    def test_the_final_summary_names_the_videos_found_the_log_and_why_some_did_not_start(self) -> None:
        out = self.feed(self.app.generate, "y", "")
        self.assertIn("Videos found:", out)
        self.assertNotIn("Scanned", out)
        self.assertRegex(out, r"Log:\s+\S*logs\S*")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            type(self.app)._print_final_summary({"scanned": 5, "completed": 2, "failed": 0, "canceled": 3, "elapsed_seconds": 5.0})
        self.assertRegex(buffer.getvalue(), r"Not started:\s+3  \(stopped with Ctrl\+C\)")

    def test_the_scan_line_gives_the_time_in_plain_words(self) -> None:
        out = self.feed(self.app.generate, "")
        self.assertRegex(out, r"Scan complete: 1 videos from \d+ files in \S+\.")

    def test_declining_the_preflight_generates_nothing_and_does_not_pause(self) -> None:
        self.feed(self.app.generate, "")                  # Enter = default answer "no"
        self.assertFalse(self.target().exists())

    def test_not_enough_space_must_be_confirmed_separately(self) -> None:
        full = shutil._ntuple_diskusage(100, 100, 0)
        with mock.patch("shutil.disk_usage", return_value=full):
            out = self.feed(self.app.generate, "n")       # Continue anyway? -> no (Proceed? is not even asked)
        self.assertIn("exceeds free space", out)
        self.assertFalse(self.target().exists())
        with mock.patch("shutil.disk_usage", return_value=full):
            self.feed(self.app.generate, "y", "y", "")    # Continue anyway? yes; Proceed? yes; Enter
        self.assertTrue(self.target().exists())

    def test_dry_run_lists_the_plan_and_only_waits_for_enter(self) -> None:
        out = self.feed(lambda: self.app.generate(dry_run=True), "")          # the Enter keeps the list on the screen
        self.assertIn("DRY RUN", out)
        self.assertIn("a.mp4", out)
        self.assertFalse(self.target().exists())

    def test_generate_and_verify_without_videos_keep_their_message_on_the_screen(self) -> None:
        with mock.patch.object(type(self.app), "scan_library", return_value=[]):
            out = self.feed(self.app.generate, "")
            self.assertIn("No videos found.", out)
            out = self.feed(self.app.verify, "")
            self.assertIn("No videos found.", out)

    def test_verify_lists_no_problem_on_the_screen_only_the_summary_and_the_file(self) -> None:
        from modules.verify import MISSING, VerifyRow
        videos = [mock.MagicMock(path=Path(f"/v/{i}.mp4")) for i in range(25)]
        rows = [VerifyRow(Path(f"/v/{i}.mp4"), MISSING, "no folder") for i in range(25)]

        def fake_verify(vids, settings, state, verify_all, on_row):
            for number, row in enumerate(rows, 1): on_row(number, len(rows), row)
            summary = {"total": 25, "valid": 0, "missing": 25, "invalid": 0, "source_changed": 0, "settings_changed": 0, "unexpected": 0,
                       "stale_building": 0, "leftover_dirs": 0, "orphan_dirs": 0, "unreadable_videos": 0, "suspect": 0}
            return summary, rows
        with mock.patch.object(type(self.app), "scan_library", return_value=videos), mock.patch("modules.screens.verify.verify_videos", fake_verify):
            out = self.feed(self.app.verify, "n")
        self.assertNotIn("/v/", out)
        self.assertNotIn("] MISSING", out)
        self.assertIn("25 videos checked: 25 missing", out)

    def test_verify_reports_trickplay_of_other_sizes_next_to_the_target(self) -> None:
        self.feed(self.app.generate, "y", "")
        (self.target().parent / "640 - 10x10").mkdir()
        out = self.feed(self.app.verify, "")
        self.assertIn("OTHER TRICKPLAY SIZES", out)
        self.assertRegex(out, r"640 - 10x10\s+on 1 video")
        self.assertIn("keeps them", out)
        self.assertNotIn("OTHER TRICKPLAY SIZES", self.feed(lambda: (self.target().parent / "640 - 10x10").rmdir() or self.app.verify(), ""))

    def test_missing_output_is_no_problem_the_file_has_the_counts_and_no_problems(self) -> None:
        out = self.feed(self.app.verify, "n")            # everything is missing: Generate is offered (declined), nothing is wrong
        self.assertIn("missing", out)
        self.assertNotIn("Problem list", out)
        text = problem_list().read_text(encoding="utf-8")
        self.assertIn("VERIFY SUMMARY", text)
        self.assertIn("missing", text)
        self.assertIn("PROBLEMS: none", text)
        self.assertNotIn("MISSING\t", text)

    def test_paths_inside_the_program_folder_are_shown_relative_and_the_others_in_full(self) -> None:
        from modules.formatting import display_path
        root = constants.APP_ROOT
        self.assertEqual(display_path(root / "diagnostics" / "benchmark" / "job_1"), str(Path("diagnostics") / "benchmark" / "job_1"))
        self.assertEqual(display_path(root), ".")
        self.assertEqual(display_path(root.parent / "elsewhere" / "a.mkv"), str(root.parent / "elsewhere" / "a.mkv"))
        self.assertEqual(display_path(Path(str(root) + "_other") / "x"), str(Path(str(root) + "_other") / "x"), "a folder that only starts with the same name")

    def test_the_usage_screen_names_the_program_folder_once(self) -> None:
        out = self.feed(self.app._show_usage)
        self.assertIn(f"Program folder: {constants.APP_ROOT}", out)

    def test_generate_without_media_roots_goes_to_the_roots_screen_and_back(self) -> None:
        self.config["media_roots"] = []
        out = self.feed(self.app.generate, "0")
        self.assertIn("No media folders are configured", out)

    def test_verify_of_a_healthy_library_only_waits_for_enter(self) -> None:
        self.feed(self.app.generate, "y", "")
        out = self.feed(self.app.verify, "")
        self.assertIn("VERIFY SUMMARY", out)
        self.assertNotIn("Regenerate", out)

    def test_verify_offers_to_fix_damage_and_the_fix_works(self) -> None:
        self.feed(self.app.generate, "y", "")
        sorted(self.target().glob("*.jpg"))[0].write_bytes(b"")
        self.assertNotEqual(verify_video(self.scan()[0], self.settings, self.app.state).status, VALID)
        # Regenerate the 1 invalid video? yes; Proceed? yes; Press Enter (generation summary is not paused: verify pauses once)
        out = self.feed(self.app.verify, "y", "y", "")
        self.assertIn("1 invalid", out)
        self.assertIn(" Generate\n", out.split("VERIFY SUMMARY")[1], "the fix continues on a Generate screen")
        self.assertIn("PREFLIGHT", out.split("VERIFY SUMMARY")[1])
        self.assertEqual(verify_video(self.scan()[0], self.settings, self.app.state).status, VALID)

    def test_declining_the_preflight_after_verify_needs_no_extra_enter(self) -> None:
        self.feed(self.app.generate, "y", "")
        sorted(self.target().glob("*.jpg"))[0].write_bytes(b"")
        out = self.feed(self.app.verify, "y", "n")           # Regenerate? yes; Start generating? no: the end, no Enter
        self.assertIn("PREFLIGHT", out)
        self.assertNotEqual(verify_video(self.scan()[0], self.settings, self.app.state).status, VALID)

    def orphan(self, name: str = "gone") -> Path:
        """The trickplay folder of a video that no longer exists: a copy of a real output under another name."""
        folder = self.media / f"{name}.trickplay"
        shutil.copytree(self.target().parent, folder)
        return folder

    def test_verify_offers_to_remove_the_folder_of_a_deleted_video(self) -> None:
        self.feed(self.app.generate, "y", "")
        orphan = self.orphan()
        out = self.feed(self.app.verify, "y", "")                # Remove the folder of videos that no longer exist? yes; Enter
        self.assertIn("TRICKPLAY FOLDERS WITHOUT A VIDEO", out)
        self.assertIn("Folders of deleted videos:", out)
        self.assertIn("Removed 1 of 1", out)
        self.assertFalse(orphan.exists())
        self.assertEqual(verify_video(self.scan()[0], self.settings, self.app.state).status, VALID, "the folder of an existing video stays")

    def test_declining_keeps_the_folder_of_a_deleted_video(self) -> None:
        self.feed(self.app.generate, "y", "")
        orphan = self.orphan()
        self.feed(self.app.verify, "n")
        self.assertTrue(orphan.exists())

    def test_a_folder_with_other_files_is_listed_but_never_offered_for_removal(self) -> None:
        self.feed(self.app.generate, "y", "")
        orphan = self.orphan()
        (orphan / "notes.txt").write_text("mine")
        out = self.feed(self.app.verify, "")                     # no question about removing: only Enter at the end
        self.assertIn("Kept (not safe to remove): 1", out)
        self.assertNotIn("Remove the", out)
        self.assertTrue((orphan / "notes.txt").exists())

    def test_the_folder_of_a_video_that_still_exists_is_never_an_orphan(self) -> None:
        self.feed(self.app.generate, "y", "")
        out = self.feed(self.app.verify, "")
        self.assertNotIn("WITHOUT A VIDEO", out)
        self.assertNotIn("Folders of deleted videos", out)
        self.assertIn("checked: all valid", out)

    def test_headless_verify_reports_but_never_removes_and_does_not_change_the_exit_code(self) -> None:
        from modules import cli
        from modules.cli import parse_args
        self.feed(self.app.generate, "y", "")
        orphan = self.orphan()
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=AssertionError("must not ask")), contextlib.redirect_stdout(out):
            self.assertEqual(self.app.run_command(parse_args(["--verify"])), cli.EXIT_OK)
        self.assertTrue(orphan.exists())
        self.assertRegex(out.getvalue(), r"Folders of deleted videos:\s+1")
        report = problem_list().read_text(encoding="utf-8")
        self.assertIn("ORPHAN", report)
        self.assertIn(str(orphan), report)

    def test_verify_lists_unreadable_videos_and_the_cli_exit_code_is_1(self) -> None:
        from modules import cli
        from modules.cli import parse_args
        self.feed(self.app.generate, "y", "")
        broken = self.media / "broken.mkv"
        broken.write_bytes(b"this is not a video" * 100)
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=AssertionError("must not ask")), contextlib.redirect_stdout(out):
            self.assertEqual(self.app.run_command(parse_args(["--verify"])), cli.EXIT_PROBLEMS)
        text = out.getvalue()
        self.assertIn("could not be read", text)
        self.assertIn(str(broken), text)
        self.assertRegex(text, r"Unreadable videos:\s+1")
        report = problem_list().read_text(encoding="utf-8")
        self.assertIn("UNREADABLE", report)

    def test_generate_in_the_cli_still_works_but_reports_exit_1_for_an_unreadable_video(self) -> None:
        from modules import cli
        from modules.cli import parse_args
        (self.media / "broken.mkv").write_bytes(b"this is not a video" * 100)
        with mock.patch("builtins.input", side_effect=AssertionError("must not ask")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.app.run_command(parse_args(["--generate"])), cli.EXIT_PROBLEMS)
        self.assertTrue(self.target().exists(), "the readable video was generated")
        self.assertEqual(self.app.last_summary["unreadable_videos"], 1)

    def interrupted_replacement(self) -> Path:
        """A finished library where the output was renamed to .__old__ and the new one never arrived (crash / failed rollback)."""
        self.feed(self.app.generate, "y", "")
        target = self.target()
        old = target.parent / f"{target.name}.__old__.2026-10-26_220000.abc123"
        target.rename(old)
        return old

    def test_verify_offers_to_restore_the_previous_output_instead_of_regenerating(self) -> None:
        old = self.interrupted_replacement()
        before = sorted(p.name for p in old.iterdir())
        out = self.feed(self.app.verify, "y", "")                # Restore ...? yes; Press Enter  (nothing left to regenerate)
        self.assertIn("PREVIOUS OUTPUT CAN BE RESTORED", out)
        self.assertIn("Restored 1 of 1", out)
        self.assertNotIn("Regenerate", out)
        self.assertFalse(old.exists())
        self.assertEqual(sorted(p.name for p in self.target().iterdir()), before)
        self.assertEqual(verify_video(self.scan()[0], self.settings, self.app.state).status, VALID)

    def test_declining_the_restore_offers_regeneration_and_keeps_the_old_copy(self) -> None:
        old = self.interrupted_replacement()
        out = self.feed(self.app.verify, "n", "n")           # Restore? no; Regenerate? no: nothing to read afterwards, so no Enter
        self.assertIn("1 missing", out)
        self.assertTrue(old.exists())
        self.assertFalse(self.target().exists())

    def test_headless_verify_never_restores(self) -> None:
        from modules import cli
        from modules.cli import parse_args
        old = self.interrupted_replacement()
        with mock.patch("builtins.input", side_effect=AssertionError("must not ask")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.app.run_command(parse_args(["--verify"])), cli.EXIT_PROBLEMS)
        self.assertTrue(old.exists())

    def test_verify_declining_leaves_the_damage_alone(self) -> None:
        self.feed(self.app.generate, "y", "")
        sprite = sorted(self.target().glob("*.jpg"))[0]
        sprite.write_bytes(b"")
        self.feed(self.app.verify, "n")
        self.assertEqual(sprite.read_bytes(), b"")


class EveryScreenStartsWithTheHeader(ScreenTest):
    def test_single_screens_clear_the_screen_and_show_the_header(self) -> None:
        cases = [(self.app.configure_preset, [""], "Preset"), (self.app.configure_key_frames, [""], "Key frames"),
                 (self.app.configure_existing_outputs, [""], "Existing outputs"), (self.app.configure_fingerprint, [""], "Detecting changed videos"),
                 (self.app._show_usage, [], "Disk usage"), (self.app._clean_temp_now, [], "Delete temporary files")]
        for fn, answers, title in cases:
            with self.subTest(title):
                with mock.patch.object(self.app, "_screen", wraps=self.app._screen) as cleared:
                    out = self.feed(fn, *answers)
                self.assertIn("MEDIA TRICKPLAY GENERATOR", out)
                self.assertIn(title, out)
                self.assertGreaterEqual(cleared.call_count, 1)

    def test_the_ffmpeg_not_found_screen_is_a_fresh_screen_each_time_it_is_shown(self) -> None:
        with mock.patch.object(self.app.registry, "discover", return_value=[]), mock.patch.object(self.app, "_screen", wraps=self.app._screen) as cleared:
            out = self.feed(lambda: self.app.ensure_ffmpeg(), "3", "4")           # re-scan once, then Back
        self.assertEqual(out.count("FFmpeg / FFprobe not found."), 2)
        self.assertEqual(out.count("MEDIA TRICKPLAY GENERATOR"), 2)
        self.assertEqual(cleared.call_count, 2)


if __name__ == "__main__":
    unittest.main()


class FFmpegLine(ScreenTest):
    """Which FFmpeg is in use takes one line; the check of every build is spoken about only when it takes time or fails."""

    def candidate(self, valid: bool = True):
        return mock.Mock(root=Path("/ff"), valid=valid, validation_error="" if valid else "cannot run", sha256="H",
                         version_line="ffmpeg version 9.0.2-essentials_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers")

    def ensure(self, cached: bool, candidate) -> str:
        from modules.screens import core
        tools = mock.Mock(gpu_names=["NVIDIA GeForce RTX 5060", "Intel(R) UHD Graphics 770"], hw_profile={"intel_gpu": True}, candidate=candidate)
        tools.backends.return_value = ["cuda", "d3d11va"]
        self.app.config["media_roots"] = ["D:\\Media", "X:\\Media"]
        with mock.patch.object(self.app.registry, "discover", return_value=[candidate]), \
                mock.patch.object(self.app.registry, "hydrate_cached", return_value=cached), \
                mock.patch.object(self.app.registry, "validate", return_value=candidate), \
                mock.patch.object(self.app.registry, "choose", return_value=candidate if candidate.valid else None), \
                mock.patch.object(core, "FFmpegTools", return_value=tools), \
                mock.patch.object(core.system_info, "host", return_value=core.system_info.Host("Intel Core i7-14700K", 20, 28, 31 * 2 ** 30 + 800 * 2 ** 20)), \
                mock.patch.object(core.system_info, "all_disks", return_value=["C: SSD", "D: SSD", "X: HDD"]), \
                mock.patch.object(core, "library_disks", return_value=["D: SSD", "X: HDD"]):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.app.ensure_ffmpeg(interactive=False)
        return out.getvalue()

    def test_a_known_build_is_one_line(self) -> None:
        out = self.ensure(True, self.candidate())
        lines = out.strip().splitlines()
        self.assertEqual(lines[0], "FFmpeg 9.0.2-essentials_build | GPU backends: cuda, d3d11va")
        self.assertEqual([line.split(None, 1) for line in (lines[2], lines[3], lines[5])], [["CPU:", "Intel Core i7-14700K, 20 cores / 28 threads"], ["GPU:", "NVIDIA GeForce RTX 5060"], ["RAM:", "32 GB"]])
        self.assertEqual(lines[4].strip(), "Intel(R) UHD Graphics 770")
        self.assertEqual(lines[6].split(None, 1), ["Disks:", "C: SSD, D: SSD, X: HDD"])
        self.assertEqual(lines[7].split(None, 2), ["Selected", "disks:", "D: SSD, X: HDD"])
        self.assertEqual(lines[2].index("Intel"), lines[3].index("NVIDIA"), "the values are in one column")
        self.assertEqual(lines[4].index("Intel"), lines[3].index("NVIDIA"), "a second graphics card continues under the first")
        self.assertIn("QSV is not available", out, "an Intel graphics chip without the QSV path is said")

    def test_the_first_check_of_a_build_is_announced_because_it_takes_a_moment(self) -> None:
        out = self.ensure(False, self.candidate())
        self.assertEqual(out.splitlines()[0], "Checking FFmpeg availability and version ...")
        self.assertIn("FFmpeg 9.0.2", out.splitlines()[1])
        self.assertIn("CPU:", out)

    def test_a_build_that_cannot_be_used_says_why(self) -> None:
        out = self.ensure(False, self.candidate(valid=False))
        self.assertIn("Not usable: cannot run", out)

    def test_the_line_for_a_computer_without_hardware_paths(self) -> None:
        from modules.screens.core import ffmpeg_summary
        self.assertEqual(ffmpeg_summary("ffmpeg version 6.1.1-3ubuntu5 Copyright", [], []), "FFmpeg 6.1.1-3ubuntu5 | CPU only")
        self.assertEqual(ffmpeg_summary("ffmpeg version 7.0", ["RTX"], []), "FFmpeg 7.0 | no GPU backend")
        self.assertEqual(ffmpeg_summary("unknown build", [], []), "unknown build | CPU only")
        self.assertEqual(ffmpeg_summary("ffmpeg version 9.0.2-essentials_build-www.gyan.dev Copyright (c) 2000-2026", [], []), "FFmpeg 9.0.2-essentials_build | CPU only")
        self.assertEqual(ffmpeg_summary("ffmpeg version 7.1-static https://johnvansickle.com/ffmpeg/ Copyright", [], []), "FFmpeg 7.1-static | CPU only")
        self.assertEqual(ffmpeg_summary("ffmpeg version N-118000-g1a2b3c4 Copyright", [], []), "FFmpeg N-118000-g1a2b3c4 | CPU only")


class LongLinesWrap(unittest.TestCase):
    """A long line on a screen is wrapped to the terminal width and what continues is indented under its value."""

    def test_labelled_values_wrap_under_the_value(self) -> None:
        from modules import prompts
        rows = [("Now", "off"), ("Why", "Generate uses the faster one for each kind and skips a GPU that is slower than the CPU.")]
        with mock.patch("shutil.get_terminal_size", return_value=shutil.os.terminal_size((60, 24))):
            lines = prompts.format_labelled(rows, indent=4, minimum=12)
        self.assertEqual(lines[0], "    Now:        off")
        self.assertGreater(len(lines), 2)
        self.assertTrue(all(len(line) <= 59 for line in lines))
        self.assertTrue(all(line.startswith(" " * 16) for line in lines[2:]))

    def test_fit_wraps_with_hanging_indent_and_keeps_short_lines(self) -> None:
        from modules import prompts
        with mock.patch("shutil.get_terminal_size", return_value=shutil.os.terminal_size((40, 24))):
            self.assertEqual(prompts.fit("  short line"), "  short line")
            wrapped = prompts.fit("  " + "word " * 20).splitlines()
        self.assertGreater(len(wrapped), 1)
        self.assertTrue(all(len(line) <= 39 for line in wrapped))
        self.assertTrue(all(line.startswith("    ") for line in wrapped[1:]))
