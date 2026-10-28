# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import contextlib
import io
import shutil
import unittest
from unittest import mock

from tests.helpers import IsolatedApp, host_adapter, make_video, needs_ffmpeg
from tests.test_pipeline import PipelineBase, quiet

from modules import cli, constants
from modules.application import run_application
from modules.cli import CliArgs, parse_args
from modules.tiles import target_dir_for


def no_prompts():
    return mock.patch("builtins.input", side_effect=AssertionError("command-line mode must never ask anything"))


class ParseArguments(unittest.TestCase):
    def test_no_arguments_means_the_interactive_menu(self) -> None:
        args = parse_args([])
        self.assertTrue(args.interactive)
        self.assertIsNone(args.command)

    def test_actions(self) -> None:
        self.assertEqual(parse_args(["--generate"]), CliArgs("generate", False))
        self.assertEqual(parse_args(["--generate", "--dry-run"]), CliArgs("generate", True))
        self.assertEqual(parse_args(["--verify"]), CliArgs("verify", False))
        self.assertEqual(parse_args(["--self-test"]), CliArgs("self-test", False))
        self.assertFalse(parse_args(["--verify"]).interactive)

    def test_wrong_arguments_exit_with_code_2(self) -> None:
        for argv in (["--dry-run"], ["--verify", "--dry-run"], ["--generate", "--verify"], ["--nonsense"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                parse_args(argv)
            self.assertEqual(raised.exception.code, 2)

    def test_version_prints_and_exits_with_0(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as raised:
            parse_args(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn(constants.APP_VERSION, out.getvalue())


class StartupWithoutPerson(IsolatedApp):
    def run_cli(self, *argv: str) -> int:
        with no_prompts(), quiet():
            return run_application(host_adapter(), platform_name="test", argv=list(argv))

    def test_no_media_roots_is_not_run(self) -> None:
        self.assertEqual(self.run_cli("--verify"), cli.EXIT_NOT_RUN)
        self.assertEqual(self.run_cli("--generate"), cli.EXIT_NOT_RUN)

    def test_broken_configuration_is_not_repaired_without_asking(self) -> None:
        (constants.DIRS["config"] / "config.json").write_text("{ not json", encoding="utf-8")
        self.assertEqual(self.run_cli("--generate"), cli.EXIT_NOT_RUN)
        self.assertIn("{ not json", (constants.DIRS["config"] / "config.json").read_text(encoding="utf-8"))

    def test_missing_ffmpeg_is_not_run(self) -> None:
        from modules.config import ConfigManager
        mgr = ConfigManager()
        cfg = mgr.load()
        cfg["media_roots"] = [str(self.media)]
        mgr.save()
        self.assertEqual(self.run_cli("--verify"), cli.EXIT_NOT_RUN)


@needs_ffmpeg
class CommandsThroughTheRealPipeline(PipelineBase):
    def setUp(self) -> None:
        super().setUp()
        from modules.ui import Application
        self.app = Application()
        self.addCleanup(self.app.state.close)
        self.app.state.open()
        self.app.config = self.config
        self.app.tools = self.tools
        self.app.ffmpeg_candidate = self.tools.candidate
        patcher = mock.patch.object(Application, "ensure_ffmpeg", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def command(self, *argv: str) -> int:
        with no_prompts(), quiet():
            return self.app.run_command(parse_args(list(argv)))

    def test_generate_then_verify_then_generate_again(self) -> None:
        make_video(self.media / "a.mp4", 25)
        make_video(self.media / "b.mp4", 12)
        self.assertEqual(self.command("--verify"), cli.EXIT_PROBLEMS, "nothing generated yet")
        self.assertEqual(self.command("--generate"), cli.EXIT_OK)
        self.assertEqual(self.command("--verify"), cli.EXIT_OK)
        videos = self.scan()
        settings = self.generator().settings
        before = {v.path.name: sorted(p.name for p in target_dir_for(v.path, settings).iterdir()) for v in videos}
        self.assertEqual(self.command("--generate"), cli.EXIT_OK, "second run: everything valid")
        after = {v.path.name: sorted(p.name for p in target_dir_for(v.path, settings).iterdir()) for v in videos}
        self.assertEqual(before, after)

    def test_generate_repairs_damage_and_verify_reports_it(self) -> None:
        make_video(self.media / "a.mp4", 25)
        self.assertEqual(self.command("--generate"), cli.EXIT_OK)
        video = self.scan()[0]
        target = target_dir_for(video.path, self.generator().settings)
        sorted(target.glob("*.jpg"))[0].write_bytes(b"")
        self.assertEqual(self.command("--verify"), cli.EXIT_PROBLEMS)
        self.assertEqual(self.command("--generate"), cli.EXIT_OK)
        self.assertEqual(self.command("--verify"), cli.EXIT_OK)

    def test_a_stopped_generate_exits_with_the_canceled_code_and_the_next_run_finishes_the_job(self) -> None:
        from modules.generator import TrickplayGenerator
        make_video(self.media / "a.mp4", 25)
        make_video(self.media / "b.mp4", 12)
        real_plan = TrickplayGenerator.plan

        def plan_then_ctrl_c(generator, videos):
            plan = real_plan(generator, videos)
            generator.request_stop()                                   # the first Ctrl+C arrives before anything started
            return plan
        with mock.patch.object(TrickplayGenerator, "plan", plan_then_ctrl_c):
            self.assertEqual(self.command("--generate"), cli.EXIT_CANCELED)
        self.assertEqual(self.app.last_summary["canceled"], 2)
        self.assertEqual(self.command("--verify"), cli.EXIT_PROBLEMS)
        self.assertEqual(self.command("--generate"), cli.EXIT_OK)
        self.assertEqual(self.command("--verify"), cli.EXIT_OK)

    def test_dry_run_changes_nothing(self) -> None:
        make_video(self.media / "a.mp4", 25)
        self.assertEqual(self.command("--generate", "--dry-run"), cli.EXIT_OK)
        self.assertFalse((self.media / "a.trickplay").exists())

    def test_folder_with_unexpected_files_is_skipped_not_replaced(self) -> None:
        make_video(self.media / "a.mp4", 25)
        video = self.scan()[0]
        target = target_dir_for(video.path, self.generator().settings)
        target.mkdir(parents=True)
        (target / "my_notes.txt").write_text("keep me")
        self.assertEqual(self.command("--generate"), cli.EXIT_PROBLEMS, "left alone, so there is still no valid output")
        self.assertEqual(self.app.last_summary["skipped_unclear_folder"], 1)
        self.assertEqual([p.name for p in target.iterdir()], ["my_notes.txt"])

    def test_dry_run_shows_the_skip_of_a_folder_with_unexpected_files(self) -> None:
        """Unattended, the real run leaves such a folder alone; the dry run must not announce a rebuild."""
        import contextlib, io
        make_video(self.media / "a.mp4", 25)
        target = target_dir_for(self.scan()[0].path, self.generator().settings)
        target.mkdir(parents=True)
        (target / "my_notes.txt").write_text("keep me")
        out = io.StringIO()
        with no_prompts(), contextlib.redirect_stdout(out):
            self.assertEqual(self.app.run_command(parse_args(["--generate", "--dry-run"])), cli.EXIT_OK)
        listing = [line for line in out.getvalue().splitlines() if "a.mp4" in line]
        self.assertEqual(len(listing), 1)
        self.assertIn("left alone: unexpected files", listing[0])
        self.assertIn("1 left alone", out.getvalue())
        self.assertEqual([p.name for p in target.iterdir()], ["my_notes.txt"])

    def test_old_widths_with_unexpected_files_are_kept_even_when_policy_is_replace(self) -> None:
        make_video(self.media / "a.mp4", 25)
        self.config["trickplay"]["other_width_policy"] = "replace"
        video = self.scan()[0]
        old = target_dir_for(video.path, self.generator().settings).parent / "320 - 3x3"
        old.mkdir(parents=True)
        (old / "1.jpg").write_bytes(b"x")
        (old / "notes.txt").write_text("keep me")
        self.assertEqual(self.command("--generate"), cli.EXIT_OK)
        self.assertTrue((old / "notes.txt").exists())

    def test_not_enough_disk_space_does_not_start(self) -> None:
        make_video(self.media / "a.mp4", 25)
        full = shutil._ntuple_diskusage(100, 100, 0)
        with mock.patch("shutil.disk_usage", return_value=full):
            self.assertEqual(self.command("--generate"), cli.EXIT_NOT_RUN)
        self.assertFalse((self.media / "a.trickplay").exists())

    def test_unavailable_media_folders_are_not_run(self) -> None:
        self.config["media_roots"] = [str(self.media / "missing_drive")]
        self.assertEqual(self.command("--verify"), cli.EXIT_NOT_RUN)

    def test_self_test_exit_code(self) -> None:
        self.assertEqual(self.command("--self-test"), cli.EXIT_OK)
        with mock.patch.object(type(self.app), "_run_self_test", return_value={"failed": 2, "passed": 1, "checks": {}}):
            self.assertEqual(self.command("--self-test"), cli.EXIT_PROBLEMS)


if __name__ == "__main__":
    unittest.main()
