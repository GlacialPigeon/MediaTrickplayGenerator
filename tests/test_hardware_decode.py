# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""The GPU decoding check: does a GPU path really decode on the GPU?"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import make_video, needs_ffmpeg
from tests.test_core import meta
from tests.test_pipeline import PipelineBase, quiet
from tests.test_screens import ScreenTest

from modules import hardware_decode as hd
from modules import hdr_validation as hv
from modules.generator import TrickplayGenerator
from modules.models import TrickplaySettings


def video(name: str, codec: str = "hevc", bits: int = 10, width: int = 3840, height: int = 2160, duration: float = 600.0):
    return meta(Path(name), codec=codec, bits=bits, width=width, height=height, duration=duration, pix_fmt="yuv420p10le" if bits == 10 else "yuv420p")


class Pure(unittest.TestCase):
    def test_one_sample_per_codec_and_bit_depth_with_the_most_pixels(self) -> None:
        picked = hd.pick_samples([
            video("a.mkv", "hevc", 10, 1920, 1080), video("b.mkv", "hevc", 10, 3840, 2160), video("c.mkv", "hevc", 8, 1280, 720),
            video("d.mkv", "h264", 8), video("e.avi", "mpeg4", 8), video("f.mkv", "av1", 10),
        ])
        self.assertEqual([m.path.name for m in picked], ["f.mkv", "d.mkv", "b.mkv", "c.mkv", "e.avi"], "sorted by the name of the kind; every codec the library has is in, not a fixed list")

    def test_the_check_takes_every_codec_there_is_so_the_ones_no_gpu_decodes_show_too(self) -> None:
        videos = [video("a.avi", "mpeg4", 8), video("b.rm", "rv40", 8), video("c.mkv", "h264", 8)]
        self.assertEqual([m.path.name for m in hd.pick_samples(videos, None)], ["c.mkv", "a.avi", "b.rm"])

    def test_a_codec_no_gpu_decodes_is_a_result_like_any_other(self) -> None:
        rows = [{"backend": "cuda", "kind": "rv40 8-bit", "ok": False, "decoder_starts": False, "unsupported": True},
                {"backend": "cuda", "kind": "h264 8-bit", "ok": True}]
        self.assertEqual(hd.verdicts(rows), {"cuda": {"rv40 8-bit": "UNSUPPORTED", "h264 8-bit": "PASS"}})
        self.assertEqual(hd.reason(rows[0]), "not supported by the GPU")

    def test_nothing_to_test_without_videos(self) -> None:
        self.assertEqual(hd.pick_samples([]), [])

    def test_frames_are_kept_in_gpu_memory(self) -> None:
        self.assertEqual(hd.with_surface_format(["-hwaccel", "d3d11va"], "d3d11va"), ["-hwaccel", "d3d11va", "-hwaccel_output_format", "d3d11"])
        self.assertEqual(hd.with_surface_format(["-hwaccel", "vaapi"], "vaapi")[-2:], ["-hwaccel_output_format", "vaapi"])
        already = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-extra_hw_frames", "24"]
        self.assertEqual(hd.with_surface_format(already, "cuda"), already)

    def test_segment_is_inside_the_video(self) -> None:
        for duration in (600.0, 25.0, 5.0):
            start, length = hd.segment_of(video("v.mkv", duration=duration))
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(start + length, duration)
        self.assertEqual(hd.segment_of(video("v.mkv", duration=600.0)), (180.0, hd.SEGMENT_SECONDS))

    def test_commands(self) -> None:
        m = video("v.mkv")
        software = hd.decode_command(Path("ffmpeg"), m, None)
        self.assertNotIn("-hwaccel", software)
        self.assertNotIn("-vf", software)
        gpu = hd.decode_command(Path("ffmpeg"), m, ["-hwaccel", "d3d11va", "-hwaccel_output_format", "d3d11"])
        self.assertLess(gpu.index("-hwaccel"), gpu.index("-i"), "hardware options belong before the input")
        self.assertIn("hwdownload,format=p010le", gpu, "without hardware frames hwdownload fails: that is the proof")
        self.assertEqual(gpu[-3:], ["-f", "null", "-"])

    def test_the_download_format_is_the_one_the_hardware_frames_have(self) -> None:
        # A format filter that allows a choice makes FFmpeg pick one hwdownload then refuses ("Invalid output format yuv420p").
        self.assertEqual(hd.download_format(video("a.mkv", bits=8)), "nv12")
        self.assertEqual(hd.download_format(video("a.mkv", bits=10)), "p010le")
        self.assertEqual(hd.download_format(video("a.mkv", bits=12)), "p016le")
        self.assertEqual(hd.download_format(meta(Path("a"), bits=8, chroma="4:4:4")), "yuv444p")
        for bits in (8, 10, 12):
            command = hd.decode_command(Path("ffmpeg"), video("a.mkv", bits=bits), ["-hwaccel", "cuda"])
            self.assertNotIn("|", command[command.index("-vf") + 1], "exactly one format")

    def test_summary_names_what_was_and_was_not_confirmed(self) -> None:
        rows = [{"backend": "cuda", "kind": "hevc 10-bit", "ok": True}, {"backend": "cuda", "kind": "vp9 8-bit", "ok": False},
                {"backend": "d3d11va", "kind": "hevc 10-bit", "ok": True}]
        self.assertEqual(hd.summarize(rows, ["cuda", "d3d11va", "qsv"]), [
            "cuda      1 of 2 kinds decode on the GPU",
            "          failed: vp9 8-bit",
            "d3d11va   1 of 1 kind decodes on the GPU",
            "qsv       not tested"])


class Running(unittest.TestCase):
    def tools(self):
        t = mock.Mock()
        t.ffmpeg = Path("ffmpeg")
        t.backend_filter.return_value = (["-hwaccel", "d3d11va"], "vf")
        return t

    def run_with(self, completed=None, error=None):
        with mock.patch("subprocess.run", side_effect=error, return_value=completed) as run:
            result = hd.check_decode(self.tools(), TrickplaySettings(640, 10.0, 10, 10, 4, 90), video("v.mkv"), "d3d11va")
        return result, run

    def test_success_is_confirmation(self) -> None:
        result, run = self.run_with(subprocess.CompletedProcess([], 0, "", ""))
        self.assertTrue(result.ok)
        self.assertGreater(result.speed, 0)
        self.assertIn("d3d11", run.call_args.args[0])

    def test_failure_is_not_confirmation_and_is_explained(self) -> None:
        result, _ = self.run_with(subprocess.CompletedProcess([], 1, "", "Impossible to convert between the formats supported by the filter"))
        self.assertFalse(result.ok)
        self.assertIn("cannot handle this video", result.error)

    def test_unknown_failures_show_the_last_line_ffmpeg_printed_and_keep_everything_for_the_log(self) -> None:
        stderr = "[h264 @ 0x1] first line\n[h264 @ 0x1] Failed setup for format d3d11: hwaccel initialisation returned error.\nError opening filters!"
        windows = ("[hwdownload @ 000001] Invalid output format yuv420p for hwframe download.\n[vf#0:0 @ 0x2] Error reinitializing filters!\n"
                   "[out#0/null @ 0x3] Nothing was written into output file, because at least one of its streams received no packets.")
        self.assertEqual(hd._one_line(windows, 1), "Invalid output format yuv420p for hwframe download.", "the cause, not the aftermath")
        result, run = self.run_with(subprocess.CompletedProcess([], 1, "", stderr))
        self.assertEqual(result.error, "first line")
        self.assertIn("first line", result.detail)
        self.assertIn("-hwaccel_output_format d3d11", result.detail, "the command is in the log so it can be repeated by hand")
        self.assertEqual(run.call_count, 2, "the second run is the one without the download step")
        self.assertNotIn("hwdownload", run.call_args.args[0])

    def test_a_decoder_that_runs_without_the_proof_is_told_apart_from_one_that_does_not_run(self) -> None:
        fails, works = subprocess.CompletedProcess([], 1, "", "boom"), subprocess.CompletedProcess([], 0, "", "")
        with mock.patch("subprocess.run", side_effect=[fails, works]):
            result = hd.check_decode(self.tools(), TrickplaySettings(640, 10.0, 10, 10, 4, 90), video("v.mkv"), "d3d11va")
        self.assertFalse(result.ok)
        self.assertTrue(result.decoder_starts)
        self.assertIn("could not be confirmed", result.error)
        with mock.patch("subprocess.run", side_effect=[fails, fails]):
            result = hd.check_decode(self.tools(), TrickplaySettings(640, 10.0, 10, 10, 4, 90), video("v.mkv"), "d3d11va")
        self.assertFalse(result.decoder_starts)

    def test_a_card_that_cannot_decode_the_kind_at_all_is_told_apart_from_a_failure_of_another_kind(self) -> None:
        refuses = subprocess.CompletedProcess([], 1, "", "[hevc @ 0x1] Hardware is lacking required capabilities; this format is not supported")
        with mock.patch("subprocess.run", side_effect=[refuses, refuses]):
            result = hd.check_decode(self.tools(), TrickplaySettings(640, 10.0, 10, 10, 4, 90), video("v.mkv"), "d3d11va")
        self.assertTrue(result.unsupported)
        self.assertFalse(result.decoder_starts)
        boom = subprocess.CompletedProcess([], 1, "", "boom")
        with mock.patch("subprocess.run", side_effect=[refuses, boom]):
            self.assertFalse(hd.check_decode(self.tools(), TrickplaySettings(640, 10.0, 10, 10, 4, 90), video("v.mkv"), "d3d11va").unsupported)
        with mock.patch("subprocess.run", side_effect=[boom, boom]):
            self.assertFalse(hd.check_decode(self.tools(), TrickplaySettings(640, 10.0, 10, 10, 4, 90), video("v.mkv"), "d3d11va").unsupported)

    def test_timeout_and_missing_program(self) -> None:
        self.assertFalse(self.run_with(error=subprocess.TimeoutExpired("ffmpeg", 1))[0].ok)
        self.assertFalse(self.run_with(error=OSError("no such file"))[0].ok)


@needs_ffmpeg
class RealFfmpeg(PipelineBase):
    def test_software_decoding_of_a_real_file_works(self) -> None:
        make_video(self.media / "a.mp4", 25)
        result = hd.check_decode(self.tools, self.generator().settings, self.scan()[0], None)
        self.assertTrue(result.ok, result.error)
        self.assertGreater(result.speed, 0)

    def test_a_gpu_path_that_does_not_exist_here_is_reported_not_raised(self) -> None:
        make_video(self.media / "a.mp4", 25)
        with mock.patch.object(type(self.tools), "backend_filter", return_value=(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"], "x")):
            result = hd.check_decode(self.tools, self.generator().settings, self.scan()[0], "cuda")
        if not result.ok: self.assertTrue(result.error)


class Screen(ScreenTest):
    def setUp(self) -> None:
        super().setUp()
        self.app.tools = mock.Mock()
        self.app.tools.candidate.sha256 = "H"
        self.app.tools.gpu_names = ["RTX"]
        patcher = mock.patch.object(type(self.app), "ensure_ffmpeg", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.library = self.root / "library"
        self.library.mkdir()

    def test_without_gpu_paths_nothing_is_asked(self) -> None:
        self.app.tools.backends.return_value = []
        out = self.feed(self.app.run_decode_check, "")            # only the Enter at the end
        self.assertIn("No GPU backend is available", out)

    def test_runs_every_path_for_every_codec_and_saves_the_report(self) -> None:
        self.app.tools.backends.return_value = ["cuda", "d3d11va"]
        found = [video("a.mkv", "hevc", 10), video("b.mkv", "h264", 8)]
        calls: list[tuple[str, str]] = []

        def check(tools, settings, m, backend, **kw):
            calls.append((m.path.name, backend or "cpu"))
            return hd.DecodeResult(backend != "d3d11va" or m.codec == "hevc", 2.0, 10.0, "" if backend != "d3d11va" or m.codec == "hevc" else "not supported")

        with mock.patch.object(type(self.app), "_collect_test_videos", return_value=found), mock.patch.object(hd, "check_decode", check):
            out = self.feed(self.app.run_decode_check, str(self.library), "y", "")
        self.assertEqual(calls, [("b.mkv", "cpu"), ("b.mkv", "cuda"), ("b.mkv", "d3d11va"), ("a.mkv", "cpu"), ("a.mkv", "cuda"), ("a.mkv", "d3d11va")])
        self.assertIn("cuda      all 2 kinds decode on the GPU", out)
        self.assertIn("d3d11va   1 of 2 kinds decode on the GPU\n            not supported: h264 8-bit", out)
        self.assertIn("h264 8-bit  | 4K", out)
        self.assertIn("not GPU", out)
        self.assertEqual(self.saved()["diagnostics"]["benchmark_folder"], str(self.library.resolve()))

    def test_declining_runs_nothing(self) -> None:
        self.app.tools.backends.return_value = ["cuda"]
        with mock.patch.object(type(self.app), "_collect_test_videos", return_value=[video("a.mkv")]), \
                mock.patch.object(hd, "check_decode") as check:
            self.feed(self.app.run_decode_check, str(self.library), "")      # Enter = default no
        check.assert_not_called()

    def test_no_usable_video(self) -> None:
        self.app.tools.backends.return_value = ["cuda"]
        with mock.patch.object(type(self.app), "_collect_test_videos", return_value=[]):
            out = self.feed(self.app.run_decode_check, str(self.library), "")
        self.assertIn("test NOT RUN", out)

    def run_check(self, results: dict[str, bool], *answers: str) -> str:
        """Run the check for one h264 10-bit video; results {backend: decodes on the GPU}."""
        self.app.tools.backends.return_value = list(results)
        def check(tools, settings, m, backend, **kw):
            ok = results.get(backend, True) if backend else True
            return hd.DecodeResult(ok, 2.0, 10.0, "" if ok else "cannot confirm", decoder_starts=not ok)
        with mock.patch.object(type(self.app), "_collect_test_videos", return_value=[video("a.mkv", "h264", 10)]), mock.patch.object(hd, "check_decode", check):
            return self.feed(self.app.run_decode_check, str(self.library), "y", *answers)

    def saved_decoding(self):
        return self.saved()["acceleration"]["gpu_decoding"]

    def test_a_path_that_decodes_on_the_cpu_is_saved_for_generate_after_confirmation(self) -> None:
        out = self.run_check({"cuda": True, "d3d11va": False}, "")       # Enter = yes (default); no Enter after a question
        self.assertIn("    d3d11va   h264 10-bit", out)
        self.assertEqual(self.saved_decoding()["results"], {"cuda": {"h264 10-bit": "PASS"}, "d3d11va": {"h264 10-bit": "FAIL"}})

    def test_the_manual_check_also_remembers_what_the_card_cannot_do(self) -> None:
        self.app.tools.backends.return_value = ["cuda"]
        def check(tools, settings, m, backend, **kw):
            return hd.DecodeResult(backend is None, 1.0, 5.0, "" if backend is None else "not supported", unsupported=backend is not None)
        with mock.patch.object(type(self.app), "_collect_test_videos", return_value=[video("a.mkv", "av1", 10)]), mock.patch.object(hd, "check_decode", check):
            out = self.feed(self.app.run_decode_check, str(self.library), "y", "")
        self.assertIn("    cuda   av1 10-bit", out)
        self.assertEqual(self.saved_decoding()["results"], {"cuda": {"av1 10-bit": "UNSUPPORTED"}})

    def test_declining_keeps_trying_the_path(self) -> None:
        out = self.run_check({"d3d11va": False}, "n")
        self.assertIn("Nothing changed", out)
        self.assertIsNone(self.app.config["acceleration"].get("gpu_decoding"))

    def test_a_result_that_is_already_saved_is_not_asked_again(self) -> None:
        self.run_check({"d3d11va": False}, "")                                # yes
        out = self.run_check({"d3d11va": False}, "")                              # only the final Enter: nothing to decide
        self.assertIn("Already saved", out)
        self.assertNotIn("Saved.\n", out)
        self.assertEqual(self.saved_decoding()["results"], {"d3d11va": {"h264 10-bit": "FAIL"}})
        out = self.run_check({"d3d11va": False, "cuda": False}, "")           # a new bad path is asked about, the old one is not repeated
        self.assertNotIn("Already saved", out)
        self.assertIn("    cuda   h264 10-bit", out)
        self.assertNotIn("d3d11va   h264 10-bit", out)

    def test_all_confirmed_is_saved_without_asking(self) -> None:
        self.run_check({"cuda": True}, "")                                       # only the final Enter
        self.assertEqual(self.saved_decoding()["results"], {"cuda": {"h264 10-bit": "PASS"}})

    def test_the_acceleration_screen_shows_the_results_and_can_forget_them(self) -> None:
        self.run_check({"d3d11va": False}, "")
        self.app.tools.backends.return_value = []
        out = self.feed(self.app.configure_acceleration, "4", "n", "0")
        self.assertIn("d3d11va: h264 10-bit decoded on the CPU", out)
        self.assertIsNotNone(self.saved_decoding())
        self.feed(self.app.configure_acceleration, "4", "y", "0")
        self.assertIsNone(self.saved_decoding())


def row(kind: str, backend: str, ok: bool, starts: bool = False) -> dict:
    return {"kind": kind, "backend": backend, "ok": ok, "decoder_starts": starts}


class Verdicts(unittest.TestCase):
    def test_only_confirmed_and_silently_software_paths_decide_something(self) -> None:
        results = hd.verdicts([
            row("hevc 10-bit", "cpu", True), row("hevc 10-bit", "cuda", True), row("h264 10-bit", "d3d11va", False, starts=True),
            row("av1 10-bit", "d3d11va", False, starts=False),            # failed for another reason: unknown, decides nothing
        ])
        self.assertEqual(results, {"cuda": {"hevc 10-bit": "PASS"}, "d3d11va": {"h264 10-bit": "FAIL"}})
        self.assertEqual(hd.verdicts([]), {})
        self.assertEqual(hd.describe(results), ["cuda: hevc 10-bit on the GPU", "d3d11va: h264 10-bit decoded on the CPU"])

    def test_blocks_only_that_path_and_that_kind_on_that_system(self) -> None:
        saved = hv.merge(None, {"d3d11va": {"h264 10-bit": "FAIL", "hevc 10-bit": "PASS"}}, "H", ["RTX"])
        blocked = lambda backend, m, h="H", gpus=("RTX",): hd.is_blocked(saved, h, list(gpus), backend, m)
        self.assertTrue(blocked("d3d11va", video("a.mkv", "h264", 10)))
        self.assertFalse(blocked("d3d11va", video("a.mkv", "h264", 8)), "other bit depth")
        self.assertFalse(blocked("d3d11va", video("a.mkv", "hevc", 10)), "decodes on the GPU")
        self.assertFalse(blocked("cuda", video("a.mkv", "h264", 10)), "other path")
        self.assertFalse(blocked("d3d11va", video("a.mkv", "h264", 10), h="other"), "other FFmpeg build")
        self.assertFalse(blocked("d3d11va", video("a.mkv", "h264", 10), gpus=("Arc",)), "other GPU")
        self.assertFalse(hd.is_blocked(None, "H", ["RTX"], "d3d11va", video("a.mkv", "h264", 10)))


class Unsupported(unittest.TestCase):
    def test_a_path_the_card_cannot_use_is_a_result_of_its_own(self) -> None:
        results = hd.verdicts([{"kind": "av1 10-bit", "backend": "cuda", "ok": False, "decoder_starts": False, "unsupported": True},
                               {"kind": "vp9 8-bit", "backend": "cuda", "ok": False, "decoder_starts": False}])
        self.assertEqual(results, {"cuda": {"av1 10-bit": "UNSUPPORTED"}}, "a failure of another kind decides nothing")
        self.assertEqual(hd.describe(results), ["cuda: av1 10-bit not supported by the GPU"])
        self.assertEqual(hd.blocked_kinds({"cuda": {"av1 10-bit": "UNSUPPORTED", "hevc 10-bit": "PASS", "h264 8-bit": "FAIL"}}),
                         [("cuda", "av1 10-bit"), ("cuda", "h264 8-bit")])

    def test_generate_skips_it(self) -> None:
        saved = hv.merge(None, {"cuda": {"av1 10-bit": "UNSUPPORTED"}}, "H", ["RTX"])
        self.assertTrue(hd.is_blocked(saved, "H", ["RTX"], "cuda", video("a.mkv", "av1", 10)))
        self.assertFalse(hd.is_blocked(saved, "H", ["RTX"], "cuda", video("a.mkv", "av1", 8)))
        self.assertFalse(hd.is_blocked(saved, "H", ["Another"], "cuda", video("a.mkv", "av1", 10)), "another card may well support it")
        self.assertEqual(hd.stored_result(saved, "H", ["RTX"], "cuda", video("a.mkv", "av1", 10)), "UNSUPPORTED")
        self.assertIsNone(hd.stored_result(saved, "H", ["RTX"], "cuda", video("a.mkv", "hevc", 10)))


class Precheck(unittest.TestCase):
    def run_precheck(self, videos, outcomes, saved=None, backends=("cuda", "d3d11va"), known=lambda b, m: False):
        """outcomes: {(kind, backend): DecodeResult}; returns (verdicts, the (kind, backend) pairs that were actually checked)."""
        tools = mock.Mock()
        tools.candidate.sha256 = "H"
        tools.gpu_names = ["RTX"]
        checked: list[tuple[str, str]] = []

        def check(t, settings, m, backend, **kw):
            checked.append((hd.sample_key(m), backend))
            return outcomes[(hd.sample_key(m), backend)]

        with mock.patch.object(hd, "check_decode", check):
            found = hd.precheck(tools, TrickplaySettings(640, 10.0, 10, 10, 4, 90), videos, saved, lambda m: list(backends), known)
        return found, checked

    OK = hd.DecodeResult(True, 1.0, 20.0)
    CPU = hd.DecodeResult(False, 1.0, 0.0, "software", decoder_starts=True)
    NO = hd.DecodeResult(False, 1.0, 0.0, "not supported", unsupported=True)
    UNKNOWN = hd.DecodeResult(False, 1.0, 0.0, "boom")

    def test_every_path_is_tested_for_a_kind_even_when_the_first_works(self) -> None:
        found, checked = self.run_precheck([video("a.mkv", "hevc", 10)], {("hevc 10-bit", "cuda"): self.OK, ("hevc 10-bit", "d3d11va"): self.CPU})
        self.assertEqual(found, {"cuda": {"hevc 10-bit": "PASS"}, "d3d11va": {"hevc 10-bit": "FAIL"}})
        self.assertEqual(checked, [("hevc 10-bit", "cuda"), ("hevc 10-bit", "d3d11va")])

    def test_a_path_the_first_cannot_use_is_tested_too(self) -> None:
        found, checked = self.run_precheck([video("a.mkv", "av1", 10)], {("av1 10-bit", "cuda"): self.NO, ("av1 10-bit", "d3d11va"): self.CPU})
        self.assertEqual(found, {"cuda": {"av1 10-bit": "UNSUPPORTED"}, "d3d11va": {"av1 10-bit": "FAIL"}})
        self.assertEqual(len(checked), 2)

    def test_what_is_known_is_not_checked_again(self) -> None:
        saved = hv.merge(None, {"cuda": {"hevc 10-bit": "PASS", "av1 10-bit": "UNSUPPORTED"}}, "H", ["RTX"])
        found, checked = self.run_precheck([video("a.mkv", "hevc", 10), video("b.mkv", "av1", 10)], {("av1 10-bit", "d3d11va"): self.OK, ("hevc 10-bit", "d3d11va"): self.OK}, saved)
        self.assertEqual(sorted(checked), [("av1 10-bit", "d3d11va"), ("hevc 10-bit", "d3d11va")], "what cuda is known to do is not tested again, d3d11va is new")
        self.assertEqual(found, {"d3d11va": {"av1 10-bit": "PASS", "hevc 10-bit": "PASS"}})

    def test_results_of_another_ffmpeg_or_card_are_checked_again(self) -> None:
        saved = hv.merge(None, {"cuda": {"hevc 10-bit": "PASS"}}, "other build", ["RTX"])
        _, checked = self.run_precheck([video("a.mkv", "hevc", 10)], {("hevc 10-bit", "cuda"): self.OK, ("hevc 10-bit", "d3d11va"): self.OK}, saved)
        self.assertEqual(checked, [("hevc 10-bit", "cuda"), ("hevc 10-bit", "d3d11va")])

    def test_a_failure_for_another_reason_decides_nothing(self) -> None:
        found, checked = self.run_precheck([video("a.mkv", "hevc", 10)], {("hevc 10-bit", "cuda"): self.UNKNOWN, ("hevc 10-bit", "d3d11va"): self.OK})
        self.assertEqual(found, {"d3d11va": {"hevc 10-bit": "PASS"}})
        self.assertEqual(checked, [("hevc 10-bit", "cuda"), ("hevc 10-bit", "d3d11va")])

    def test_signatures_generate_leaves_out_anyway_are_not_checked(self) -> None:
        found, checked = self.run_precheck([video("a.mkv", "hevc", 10), video("b.avi", "mpeg4", 8)], {("hevc 10-bit", "d3d11va"): self.OK, ("mpeg4 8-bit", "d3d11va"): self.NO},
                                           known=lambda backend, m: backend == "cuda")
        self.assertEqual(checked, [("hevc 10-bit", "d3d11va"), ("mpeg4 8-bit", "d3d11va")])

    def test_a_codec_from_an_old_format_is_checked_too(self) -> None:
        found, checked = self.run_precheck([video("a.rm", "rv40", 8)], {("rv40 8-bit", "cuda"): self.NO, ("rv40 8-bit", "d3d11va"): self.NO})
        self.assertEqual(found, {"cuda": {"rv40 8-bit": "UNSUPPORTED"}, "d3d11va": {"rv40 8-bit": "UNSUPPORTED"}})

    def test_a_kind_the_benchmark_gave_to_the_cpu_is_not_checked_on_the_gpu(self) -> None:
        tools = mock.Mock()
        tools.candidate.sha256, tools.gpu_names = "H", ["RTX"]
        with mock.patch.object(hd, "check_decode", side_effect=AssertionError("no GPU check for a CPU kind")):
            found = hd.precheck(tools, TrickplaySettings(640, 10.0, 10, 10, 4, 90), [video("a.mkv", "h264", 8)], None, lambda m: [], lambda b, m: False)
        self.assertEqual(found, {})

    def test_one_check_per_kind_not_per_video(self) -> None:
        _, checked = self.run_precheck([video("a.mkv", "hevc", 10), video("b.mkv", "hevc", 10, 1920, 1080)], {("hevc 10-bit", "cuda"): self.OK, ("hevc 10-bit", "d3d11va"): self.OK})
        self.assertEqual(checked, [("hevc 10-bit", "cuda"), ("hevc 10-bit", "d3d11va")])


class GenerateChecksNewKinds(ScreenTest):
    def setUp(self) -> None:
        super().setUp()
        self.app.tools = mock.Mock()
        self.app.tools.candidate.sha256 = "H"
        self.app.tools.gpu_names = ["RTX"]
        self.app.state.open()
        self.addCleanup(self.app.state.close)
        self.generator = mock.Mock()
        self.paths = ["cuda", "d3d11va"]
        self.generator._preferred_backends.side_effect = lambda: self.paths
        self.generator._backends_for.side_effect = lambda meta, mode: self.paths
        self.logger = mock.Mock()

    def plan(self, *metas, action="GENERATE"):
        return [mock.Mock(meta=m, action=action) for m in metas]

    def run_it(self, plan, outcomes):
        def check(tools, settings, m, backend, **kw): return outcomes[(hd.sample_key(m), backend)]
        with mock.patch.object(hd, "check_decode", check):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                done = self.app._check_new_kinds_of_video(plan, self.generator, self.logger)
        return done, out.getvalue()

    def remembered(self):
        return self.app.config["acceleration"].get("gpu_decoding")

    def test_new_kinds_are_checked_without_questions_and_remembered_including_what_the_card_cannot_do(self) -> None:
        plan = self.plan(video("a.mkv", "hevc", 10), video("b.mkv", "av1", 10))
        done, out = self.run_it(plan, {("hevc 10-bit", "cuda"): Precheck.OK, ("hevc 10-bit", "d3d11va"): Precheck.CPU, ("av1 10-bit", "cuda"): Precheck.NO, ("av1 10-bit", "d3d11va"): Precheck.OK})
        self.assertTrue(done)
        self.assertIn("BACKEND COMPATIBILITY", out)
        rows = {line.split()[0] + " " + line.split()[1]: line.split()[2:] for line in out.splitlines() if line.startswith(("  hevc 10-bit", "  av1 10-bit"))}
        self.assertEqual(rows["hevc 10-bit"], ["OK", "no", "OK"])
        self.assertEqual(rows["av1 10-bit"], ["no", "OK", "OK"])
        self.assertEqual(self.remembered()["results"], {"cuda": {"hevc 10-bit": "PASS", "av1 10-bit": "UNSUPPORTED"}, "d3d11va": {"hevc 10-bit": "FAIL", "av1 10-bit": "PASS"}})
        self.assertEqual(self.remembered()["ffmpeg_hash"], "H")
        self.assertEqual(self.remembered(), self.saved()["acceleration"]["gpu_decoding"], "saved to disk")

    def test_nothing_is_checked_twice(self) -> None:
        plan = self.plan(video("a.mkv", "hevc", 10))
        self.run_it(plan, {("hevc 10-bit", "cuda"): Precheck.OK, ("hevc 10-bit", "d3d11va"): Precheck.OK})
        done, out = self.run_it(plan, {})                    # a check now would raise KeyError
        self.assertTrue(done)
        self.assertEqual(out, "")

    def test_videos_that_need_no_generation_and_cpu_or_gpu_only_modes_check_nothing(self) -> None:
        self.assertEqual(self.run_it(self.plan(video("a.mkv"), action="SKIP"), {}), (True, ""))
        for mode in ("cpu", "gpu"):
            self.app.config["acceleration"]["mode"] = mode
            self.assertEqual(self.run_it(self.plan(video("a.mkv")), {}), (True, ""))
        self.app.config["acceleration"]["mode"] = "auto"
        self.paths = []
        self.assertEqual(self.run_it(self.plan(video("a.mkv")), {}), (True, ""), "no GPU path")

    def test_an_undecided_check_is_not_remembered_and_says_generate_will_try(self) -> None:
        done, out = self.run_it(self.plan(video("a.mkv", "hevc", 10)), {("hevc 10-bit", "cuda"): Precheck.UNKNOWN, ("hevc 10-bit", "d3d11va"): Precheck.OK})
        self.assertTrue(done)
        self.assertIn("hevc 10-bit  ", out)
        self.assertIn("? = could not be decided", out)
        self.assertEqual(self.remembered()["results"], {"d3d11va": {"hevc 10-bit": "PASS"}}, "only what was decided is remembered")

    def test_the_check_comes_before_the_benchmark_and_the_preflight(self) -> None:
        from modules.screens import generate as generate_screen
        order: list[str] = []
        with mock.patch.object(generate_screen, "TrickplayGenerator"), \
                mock.patch.object(self.app, "_require_tools"), \
                mock.patch.object(self.app, "_apply_storage_groups"), \
                mock.patch.object(self.app, "_check_new_kinds_of_video", side_effect=lambda *a: order.append("check") or True), \
                mock.patch.object(self.app, "_offer_benchmark_first", side_effect=lambda *a: order.append("benchmark") or False), \
                mock.patch.object(self.app, "_preflight", side_effect=lambda *a: order.append("preflight") or (False, set())):
            self.app._plan_and_run([], self.logger, offer_benchmark=True)
        self.assertEqual(order, ["check", "benchmark", "preflight"])

    def test_ctrl_c_stops_before_anything_is_started(self) -> None:
        with mock.patch.object(hd, "check_decode", side_effect=KeyboardInterrupt):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(self.app._check_new_kinds_of_video(self.plan(video("a.mkv")), self.generator, self.logger))


class CompatibilityTable(unittest.TestCase):
    def table(self, kinds, backends, cells):
        return "\n".join(hd.compatibility_lines(kinds, backends, lambda kind, backend: cells.get((kind, backend), "untested")))

    def test_every_kind_is_a_row_with_a_column_per_path_and_the_cpu(self) -> None:
        text = self.table(["av1 10-bit", "h264 8-bit"], ["cuda", "d3d11va"], {("av1 10-bit", "cuda"): "no", ("av1 10-bit", "d3d11va"): "ok", ("h264 8-bit", "cuda"): "ok"})
        lines = text.splitlines()
        self.assertEqual(lines[1].split(), ["Kind", "of", "video", "cuda", "d3d11va", "CPU"])
        self.assertEqual(lines[2].split(), ["av1", "10-bit", "no", "OK", "OK"])
        self.assertEqual(lines[3].split(), ["h264", "8-bit", "OK", "-", "OK"])
        self.assertEqual(lines[2].index("no"), lines[3].index("OK"), "the columns line up")
        self.assertIn("OK = decodes on the GPU", text)
        self.assertIn("no = does not decode on the GPU", text)
        self.assertIn("- = not tested", text)
        self.assertNotIn("? =", text)

    def test_nothing_is_printed_without_kinds(self) -> None:
        self.assertEqual(hd.compatibility_lines([], ["cuda"], lambda kind, backend: "ok"), [])



class GenerateObeysTheDecodeCheck(PipelineBase):
    def used_paths(self, video, saved, mode: str = "auto") -> list[str]:
        gen = self.generator()
        self.config["acceleration"]["mode"] = mode
        self.config["acceleration"]["gpu_decoding"] = saved
        calls: list[str] = []

        def attempt(self_, m, frames_dir, method, backend, label):
            calls.append(f"{method}/{backend}" if backend else method)
            return TrickplayGenerator._run_ffmpeg_extract(self_, m, frames_dir, "CPU", None, progress_label=label)

        with mock.patch.object(type(self.tools), "backends", return_value=["d3d11va"]), \
                mock.patch.object(TrickplayGenerator, "_attempt_extract_with_retry", attempt), quiet():
            gen._choose_method_and_extract(video, self.root / "frames", "t")
        return calls

    def test_a_path_that_decodes_on_the_cpu_is_skipped_only_for_that_kind(self) -> None:
        make_video(self.media / "a.mp4", 12)
        base = self.scan()[0]
        h264_10 = dataclasses.replace(base, codec="h264", bits=10, pix_fmt="yuv420p10le")
        h264_8 = dataclasses.replace(base, codec="h264", bits=8, pix_fmt="yuv420p")
        saved = hv.merge(None, {"d3d11va": {"h264 10-bit": "FAIL", "h264 8-bit": "PASS"}}, self.tools.candidate.sha256, list(self.tools.gpu_names))
        self.assertEqual(self.used_paths(h264_10, saved), ["CPU"])
        self.assertEqual(self.used_paths(h264_8, saved), ["GPU/d3d11va"])
        self.assertEqual(self.used_paths(h264_10, None), ["GPU/d3d11va"], "no check done: as before")

    def test_forced_gpu_mode_is_not_overruled(self) -> None:
        make_video(self.media / "a.mp4", 12)
        h264_10 = dataclasses.replace(self.scan()[0], codec="h264", bits=10, pix_fmt="yuv420p10le")
        saved = hv.merge(None, {"d3d11va": {"h264 10-bit": "FAIL"}}, self.tools.candidate.sha256, list(self.tools.gpu_names))
        self.assertEqual(self.used_paths(h264_10, saved, mode="gpu"), ["GPU/d3d11va"], "the person asked for GPU only")

    def test_results_of_another_ffmpeg_do_not_apply(self) -> None:
        make_video(self.media / "a.mp4", 12)
        h264_10 = dataclasses.replace(self.scan()[0], codec="h264", bits=10, pix_fmt="yuv420p10le")
        saved = hv.merge(None, {"d3d11va": {"h264 10-bit": "FAIL"}}, "some-other-build", list(self.tools.gpu_names))
        self.assertEqual(self.used_paths(h264_10, saved), ["GPU/d3d11va"])


if __name__ == "__main__":
    unittest.main()
