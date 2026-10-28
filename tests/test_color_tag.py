# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""A colour space tag that newer FFmpeg builds refuse (bt2020c) must not make a video fail or switch a GPU off."""
from __future__ import annotations

import unittest
from unittest import mock

from tests.helpers import make_video, needs_ffmpeg, open_state
from tests.test_core import meta
from tests.test_pipeline import PipelineBase, quiet

from modules.generator import TrickplayGenerator
from modules.models import ExtractResult

REFUSED = ("Impossible to convert between the formats supported by the filter 'Parsed_fps_0' and the filter 'auto_scale_0'\n"
           "Link 'Parsed_fps_0.default' -> 'auto_scale_0.default':\n  Color spaces:\n    src: bt2020c\n    dst: unknown gbr bt709")


class Retry(PipelineBase):
    def attempt(self, outcomes: list[ExtractResult], method: str = "CPU", backend=None, *, keyframes: bool = False):
        make_video(self.media / "a.mp4", 12)
        video = self.scan()[0]
        gen = self.generator()
        gen.keyframe_only = keyframes
        calls: list[bool] = []
        self.keyframe_calls = []

        def run(self_, m, frames, meth, back, *, ignore_color_tag=False, keyframes=False, **kw):
            calls.append(ignore_color_tag)
            self.keyframe_calls.append(keyframes)
            return outcomes[len(calls) - 1]

        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", run), quiet():
            result = gen._attempt_extract_with_retry(video, self.root / "frames", method, backend, "t")
        return result, calls, video

    def test_a_refused_colour_tag_is_ignored_on_the_second_try(self) -> None:
        ok = ExtractResult(True, "CPU", None, 1.0, 3)
        result, calls, _ = self.attempt([ExtractResult(False, "CPU", None, 0.1, 0, REFUSED), ok])
        self.assertTrue(result.success)
        self.assertEqual(calls, [False, True], "the first try is unchanged")

    def test_the_gpu_is_not_blamed_for_a_colour_tag(self) -> None:
        ok = ExtractResult(True, "GPU", "cuda", 1.0, 3)
        result, calls, video = self.attempt([ExtractResult(False, "GPU", "cuda", 0.1, 0, REFUSED, unsupported=True), ok], "GPU", "cuda")
        self.assertTrue(result.success)
        self.assertFalse(self.state.known_unsupported("cuda", video), "other videos of this codec keep using the GPU")

    def test_a_real_failure_is_not_retried_with_the_tag_ignored(self) -> None:
        bad = ExtractResult(False, "CPU", None, 0.1, 0, "Invalid data found when processing input")
        result, calls, _ = self.attempt([bad])
        self.assertFalse(result.success)
        self.assertEqual(calls, [False])

    def test_a_refused_colour_tag_does_not_cost_the_key_frame_speed_up(self) -> None:
        ok = ExtractResult(True, "GPU", "cuda", 1.0, 3)
        result, calls, _ = self.attempt([ExtractResult(False, "GPU", "cuda", 0.1, 0, REFUSED), ok], "GPU", "cuda", keyframes=True)
        self.assertTrue(result.success)
        self.assertEqual((calls, self.keyframe_calls), ([False, True], [True, True]), "the tag is dealt with first, the key frames stay")
        self.assertTrue(result.keyframe_only and result.color_tag_ignored)

    def test_key_frames_are_given_up_when_the_tag_is_not_the_problem(self) -> None:
        ok = ExtractResult(True, "CPU", None, 1.0, 3)
        result, calls, _ = self.attempt([ExtractResult(False, "CPU", None, 0.1, 0, "this decoder cannot skip pictures"), ok], keyframes=True)
        self.assertEqual((calls, self.keyframe_calls), ([False, False], [True, False]))
        self.assertFalse(result.keyframe_only)

    def test_the_reason_key_frames_failed_is_in_the_log(self) -> None:
        ok = ExtractResult(True, "CPU", None, 1.0, 3)
        gen_log: list[str] = []
        with mock.patch("modules.logger.RunLogger.log", lambda self_, level, message: gen_log.append(f"{level} {message}")):
            self.attempt([ExtractResult(False, "CPU", None, 0.1, 0, "this decoder cannot skip pictures"), ok], keyframes=True)
        self.assertTrue(any("Key-frame-only extraction failed" in line and "this decoder cannot skip pictures" in line for line in gen_log), gen_log)

    def test_when_ignoring_the_tag_does_not_help_the_old_rules_apply(self) -> None:
        bad = ExtractResult(False, "GPU", "cuda", 0.1, 0, REFUSED, unsupported=True)
        result, calls, video = self.attempt([bad, bad], "GPU", "cuda")
        self.assertFalse(result.success)
        self.assertEqual(calls, [False, True])
        self.assertTrue(self.state.known_unsupported("cuda", video), "a genuinely unsupported GPU path is still recorded")


@needs_ffmpeg
class RealFfmpeg(PipelineBase):
    def test_the_prefix_is_accepted_by_ffmpeg_and_gives_the_same_frames(self) -> None:
        make_video(self.media / "a.mp4", 25)
        video = self.scan()[0]
        gen = self.generator()
        plain = gen._run_ffmpeg_extract(video, self.root / "p", "CPU", None)
        fixed = gen._run_ffmpeg_extract(video, self.root / "f", "CPU", None, ignore_color_tag=True)
        self.assertTrue(plain.success and fixed.success, (plain.error, fixed.error))
        self.assertEqual(plain.frames, fixed.frames)
        self.assertEqual((self.root / "p" / "frame_00000001.jpg").read_bytes(), (self.root / "f" / "frame_00000001.jpg").read_bytes())


class StateCleanup(unittest.TestCase):
    def test_failures_caused_by_a_colour_tag_are_forgotten_when_the_database_opens(self) -> None:
        from tests.helpers import IsolatedApp
        class Case(IsolatedApp):
            def runTest(self) -> None:
                state = open_state()
                m = meta(self.root / "v.mkv")
                state.record_backend_failure("cuda", m, "unsupported", REFUSED)
                state.record_backend_failure("d3d11va", m, "unsupported", "Function not implemented")
                state.close()
                state = open_state()
                self.assertFalse(state.known_unsupported("cuda", m))
                self.assertTrue(state.known_unsupported("d3d11va", m), "other recorded failures stay")
                state.close()
        result = unittest.TextTestRunner(stream=__import__("io").StringIO()).run(Case())
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)


if __name__ == "__main__":
    unittest.main()
