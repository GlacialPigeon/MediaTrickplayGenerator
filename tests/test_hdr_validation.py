# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""What the HDR check's ratings do: stored per GPU path and kind of HDR, and used by Generate."""
from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import needs_ffmpeg, make_video
from tests.test_core import meta
from tests.test_pipeline import PipelineBase, quiet
from tests.test_screens import ScreenTest

from modules import hdr_validation as hv
from modules.generator import TrickplayGenerator
from modules.tiles import hdr_kind, hdr_label


def report(visual: dict[str, str]) -> dict:
    rows = []
    for label, kind in (("HDR10", "HDR10"), ("Dolby Vision P8", "Dolby Vision P8")):
        for backend in ("cpu", "cuda", "d3d11va"):
            rows.append({"backend": backend, "status": "PASS", "hdr_label": label, "hdr_kind": kind})
    rows.append({"backend": "cuda", "status": "FAILED", "hdr_label": "HLG", "hdr_kind": "HLG"})
    return {"rows": rows, "visual": visual, "ffmpeg_hash": "H", "gpu_names": ["RTX"]}


class ReviewResults(unittest.TestCase):
    def test_ratings_become_pass_or_fail_per_path_and_kind(self) -> None:
        data = report({"HDR10 | cpu": "TOO_BRIGHT", "HDR10 | cuda": "ACCEPTABLE", "HDR10 | d3d11va": "TOO_DARK",
                       "Dolby Vision P8 | cuda": "WRONG_COLORS", "Dolby Vision P8 | d3d11va": "PENDING"})
        self.assertEqual(hv.review_results(data), {"cuda": {"HDR10": "PASS", "Dolby Vision P8": "FAIL"}, "d3d11va": {"HDR10": "FAIL"}})
        self.assertEqual(sorted(hv.failing(hv.review_results(data))), [("cuda", "Dolby Vision P8"), ("d3d11va", "HDR10")])

    def test_the_cpu_and_failed_runs_decide_nothing(self) -> None:
        data = report({"HDR10 | cpu": "TOO_BRIGHT", "HLG | cuda": "TOO_DARK"})
        self.assertEqual(hv.review_results(data), {})

    def test_reports_from_before_the_per_kind_test_decide_nothing(self) -> None:
        old = {"rows": [{"backend": "cuda", "status": "PASS"}], "visual": {"cuda": "TOO_DARK"}}
        self.assertEqual(hv.review_results(old), {})


class MergeAndBlock(unittest.TestCase):
    def test_new_results_are_added_to_the_saved_ones_of_the_same_system(self) -> None:
        saved = hv.merge(None, {"cuda": {"HDR10": "FAIL"}}, "H", ["RTX"])
        merged = hv.merge(saved, {"cuda": {"HLG": "PASS"}, "d3d11va": {"HDR10": "PASS"}}, "H", ["RTX"])
        self.assertEqual(merged["results"], {"cuda": {"HDR10": "FAIL", "HLG": "PASS"}, "d3d11va": {"HDR10": "PASS"}})
        again = hv.merge(merged, {"cuda": {"HDR10": "PASS"}}, "H", ["RTX"])
        self.assertEqual(again["results"]["cuda"]["HDR10"], "PASS", "a new rating replaces the old one")

    def test_results_of_another_ffmpeg_or_gpu_are_dropped(self) -> None:
        saved = hv.merge(None, {"cuda": {"HDR10": "FAIL"}}, "H", ["RTX"])
        self.assertEqual(hv.merge(saved, {"qsv": {"HLG": "PASS"}}, "H2", ["RTX"])["results"], {"qsv": {"HLG": "PASS"}})
        self.assertEqual(hv.merge(saved, {"qsv": {"HLG": "PASS"}}, "H", ["RTX", "Arc"])["results"], {"qsv": {"HLG": "PASS"}})

    def test_only_a_bad_rating_for_that_path_and_kind_blocks(self) -> None:
        saved = hv.merge(None, {"cuda": {"HDR10": "FAIL", "HLG": "PASS"}}, "H", ["RTX"])
        blocked = lambda backend, kind, h="H", gpus=("RTX",): hv.is_blocked(saved, h, list(gpus), backend, kind)
        self.assertTrue(blocked("cuda", "HDR10"))
        self.assertFalse(blocked("cuda", "HLG"), "rated fine")
        self.assertFalse(blocked("cuda", "Dolby Vision P8"), "never tested: as before")
        self.assertFalse(blocked("d3d11va", "HDR10"), "another path")
        self.assertFalse(blocked("cuda", "HDR10", h="other"), "another FFmpeg build")
        self.assertFalse(blocked("cuda", "HDR10", gpus=("Arc",)), "another GPU")
        self.assertFalse(blocked("cuda", ""), "not HDR")
        for nothing in (None, {}, {"result": "FAIL", "backend": "cuda"}, "x"):
            self.assertFalse(hv.is_blocked(nothing, "H", ["RTX"], "cuda", "HDR10"))


class Kinds(unittest.TestCase):
    def test_names(self) -> None:
        self.assertEqual(hdr_kind(meta(Path("a"))), "")
        self.assertEqual(hdr_kind(meta(Path("a"), hdr=True, hdr_type="HDR10 / PQ")), "HDR10")
        self.assertEqual(hdr_kind(meta(Path("a"), hdr=True, hdr_type="Dolby Vision", dv_profile="5")), "Dolby Vision P5")
        self.assertEqual(hdr_kind(meta(Path("a"), hdr=True, hdr_type="Dolby Vision")), "Dolby Vision")
        self.assertEqual(hdr_kind(meta(Path("a"), hdr=True, hdr_type="")), "HDR")
        self.assertEqual(hdr_label(meta(Path("a"), hdr=True, hdr_type="HLG", hdr_pool="HDR Uncertain")), "HLG (uncertain)")


@needs_ffmpeg
class GenerateObeysTheRatings(PipelineBase):
    def used_paths(self, video, validation) -> list[str]:
        gen = self.generator()
        self.config["acceleration"]["mode"] = "auto"
        self.config["acceleration"]["hdr_validation"] = validation
        calls: list[str] = []

        def attempt(self_, m, frames_dir, method, backend, label):
            calls.append(f"{method}/{backend}" if backend else method)
            return TrickplayGenerator._run_ffmpeg_extract(self_, m, frames_dir, "CPU", None, progress_label=label)

        with mock.patch.object(type(self.tools), "backends", return_value=["cuda"]), \
                mock.patch.object(TrickplayGenerator, "_attempt_extract_with_retry", attempt), quiet():
            gen._choose_method_and_extract(video, self.root / "frames", "t")
        return calls

    def test_a_badly_rated_path_is_skipped_only_for_that_kind(self) -> None:
        make_video(self.media / "a.mp4", 12)
        base = self.scan()[0]
        hdr10 = dataclasses.replace(base, hdr=True, hdr_type="HDR10 / PQ")
        hlg = dataclasses.replace(base, hdr=True, hdr_type="HLG")
        validation = hv.merge(None, {"cuda": {"HDR10": "FAIL", "HLG": "PASS"}}, self.tools.candidate.sha256, list(self.tools.gpu_names))
        self.assertEqual(self.used_paths(hdr10, validation), ["CPU"])
        self.assertEqual(self.used_paths(hlg, validation), ["GPU/cuda"])
        self.assertEqual(self.used_paths(base, validation), ["GPU/cuda"], "SDR is not affected")
        self.assertEqual(self.used_paths(hdr10, None), ["GPU/cuda"], "no ratings: as before")

    def test_ratings_of_another_ffmpeg_do_not_apply(self) -> None:
        make_video(self.media / "a.mp4", 12)
        hdr10 = dataclasses.replace(self.scan()[0], hdr=True, hdr_type="HDR10 / PQ")
        validation = hv.merge(None, {"cuda": {"HDR10": "FAIL"}}, "some-other-build", list(self.tools.gpu_names))
        self.assertEqual(self.used_paths(hdr10, validation), ["GPU/cuda"])


class OfferAfterReview(ScreenTest):
    def setUp(self) -> None:
        super().setUp()
        self.app.tools = mock.Mock()
        self.app.tools.candidate.sha256 = "H"
        self.app.tools.gpu_names = ["RTX"]
        self.app.config.setdefault("acceleration", {})["hdr_validation"] = None

    def saved_validation(self):
        return self.saved()["acceleration"]["hdr_validation"]

    def test_bad_ratings_are_saved_after_confirmation(self) -> None:
        data = report({"HDR10 | cuda": "ACCEPTABLE", "Dolby Vision P8 | cuda": "WRONG_COLORS"})
        out = self.feed(lambda: self.app._offer_hdr_policy(data), "y")
        self.assertIn("Dolby Vision P8 on cuda", out)
        self.assertEqual(self.saved_validation()["results"], {"cuda": {"HDR10": "PASS", "Dolby Vision P8": "FAIL"}})

    def test_declining_changes_nothing(self) -> None:
        data = report({"Dolby Vision P8 | cuda": "TOO_DARK"})
        out = self.feed(lambda: self.app._offer_hdr_policy(data), "")          # Enter = default no
        self.assertIn("Nothing changed", out)
        self.assertIsNone(self.saved_validation())

    def test_good_ratings_are_saved_without_asking(self) -> None:
        data = report({"HDR10 | cuda": "ACCEPTABLE", "HDR10 | d3d11va": "ACCEPTABLE"})
        out = self.feed(lambda: self.app._offer_hdr_policy(data))
        self.assertIn("GPU stays in use", out)
        self.assertEqual(self.saved_validation()["results"], {"cuda": {"HDR10": "PASS"}, "d3d11va": {"HDR10": "PASS"}})

    def test_good_rating_lifts_an_earlier_bad_one(self) -> None:
        self.feed(lambda: self.app._offer_hdr_policy(report({"HDR10 | cuda": "TOO_BRIGHT"})), "y")
        self.feed(lambda: self.app._offer_hdr_policy(report({"HDR10 | cuda": "ACCEPTABLE"})))
        self.assertEqual(self.saved_validation()["results"], {"cuda": {"HDR10": "PASS"}})

    def test_nothing_rated_nothing_asked(self) -> None:
        self.feed(lambda: self.app._offer_hdr_policy(report({})))
        self.assertIsNone(self.saved_validation())

    def test_report_of_another_system_is_not_applied(self) -> None:
        data = report({"HDR10 | cuda": "TOO_DARK"})
        data["ffmpeg_hash"] = "old build"
        out = self.feed(lambda: self.app._offer_hdr_policy(data))
        self.assertIn("another FFmpeg build or GPU", out)
        self.assertIsNone(self.saved_validation())

    def test_acceleration_screen_shows_the_results_and_can_forget_them(self) -> None:
        self.feed(lambda: self.app._offer_hdr_policy(report({"HDR10 | cuda": "TOO_DARK"})), "y")
        self.app.tools.backends.return_value = []
        out = self.feed(self.app.configure_acceleration, "4", "n", "0")        # the saved results, Forget? no
        self.assertIn("cuda: HDR10 NOT OK", out)
        self.assertIsNotNone(self.saved_validation())
        self.feed(self.app.configure_acceleration, "4", "y", "0")
        self.assertIsNone(self.saved_validation())


if __name__ == "__main__":
    unittest.main()
