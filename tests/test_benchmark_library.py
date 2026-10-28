# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""The benchmark on the person's own library: the sample, the local copies, the key frame A/B and the suggested settings."""
from __future__ import annotations

import contextlib
import dataclasses
import io
import random
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import make_video, needs_ffmpeg
from tests.test_core import meta
from tests.test_pipeline import PipelineBase, quiet
from tests.test_screens import ScreenTest

from modules import benchmark_advice as ba
from modules import constants, library_sample as ls, path_choice as pc
from modules.diagnostics import DiagnosticJob, DiagnosticRunner, diagnostic_segments, report_name
from modules.generator import TrickplayGenerator
from modules.models import ExtractResult


def video(name: str, codec: str = "h264", bits: int = 8, width: int = 1920, height: int = 1080, duration: float = 3600.0, hdr: bool = False, **kw):
    return meta(Path(name), codec=codec, bits=bits, width=width, height=height, duration=duration, hdr=hdr,
                pix_fmt="yuv420p10le" if bits == 10 else "yuv420p", size=int(duration * 1_000_000), **kw)


class ReportNames(unittest.TestCase):
    def test_name_says_number_kind_file_segment_and_time(self) -> None:
        hdr = meta(Path("X:/Movies/Dune: Part Two (2024).mkv"), hdr=True, hdr_type="HDR10 / PQ", hdr_pool="HDR")
        self.assertEqual(report_name(1, hdr, 2, 605.4), "01 HDR10 - Dune_ Part Two (2024) - seg2 @10m05s")
        self.assertEqual(report_name(12, meta(Path("a.mkv")), 1, 7), "12 SDR - a - seg1 @7s")
        self.assertEqual(report_name(3, meta(Path("b.mkv")), 1, 3725), "03 SDR - b - seg1 @1h02m05s")
        long = report_name(1, meta(Path("x" * 200 + ".mkv")), 1, 0)
        self.assertLess(len(long), 80)
        self.assertFalse(set(long) & set('<>:"/\\|?*'))


class Sampling(unittest.TestCase):
    def library(self):
        return ([video(f"h264_{i}.mkv") for i in range(30)] + [video(f"hevc4k_{i}.mkv", "hevc", 10, 3840, 2160) for i in range(12)]
                + [video(f"sd_{i}.avi", "mpeg4", 8, 720, 480) for i in range(5)] + [video("av1.mkv", "av1", 10)]
                + [video(f"h264_4k_{i}.mkv", "h264", 8, 3840, 2160) for i in range(6)] + [video(f"hevc_8_{i}.mkv", "hevc", 8, 1280, 720) for i in range(6)]
                + [video("vp9.webm", "vp9", 8)]
                + [video("hdr.mkv", "hevc", 10, 3840, 2160, hdr=True, hdr_type="HDR10 / PQ", hdr_pool="HDR")])

    def test_ten_files_eight_by_kind_and_two_at_random(self) -> None:
        picked = ls.pick_sample(self.library(), rng=random.Random(1))
        self.assertEqual(len(picked), 10)
        self.assertEqual(len({p.meta.path for p in picked}), 10, "no file twice")
        kinds = [p.why for p in picked if not p.why.startswith("random")]
        self.assertEqual(len(kinds), 8)
        self.assertEqual(len(set(kinds)), 8, "every kind once")
        self.assertEqual(sum(p.why.startswith("random") for p in picked), 2)

    def test_fewer_kinds_than_eight_are_all_taken_and_the_rest_is_random(self) -> None:
        library = [video(f"a{i}.mkv") for i in range(20)] + [video("b.mkv", "hevc", 10)]
        picked = ls.pick_sample(library, rng=random.Random(5))
        self.assertEqual(len(picked), 10)
        self.assertEqual(sum(not p.why.startswith("random") for p in picked), 2)

    def test_every_codec_is_in_before_the_most_common_kinds_fill_up(self) -> None:
        kinds = " ".join(p.why for p in ls.pick_sample(self.library(), rng=random.Random(2)))
        for codec in ("h264", "hevc", "av1", "mpeg4"): self.assertIn(codec, kinds)

    def test_a_small_library_is_taken_as_it_is(self) -> None:
        picked = ls.pick_sample([video("a.mkv"), video("b.mkv", "hevc", 10)], rng=random.Random(3))
        self.assertEqual(sorted(p.meta.path.name for p in picked), ["a.mkv", "b.mkv"])
        self.assertEqual(ls.pick_sample([], rng=random.Random(3)), [])

    def test_longer_files_are_preferred_and_broken_ones_are_ignored(self) -> None:
        library = [video("short.mkv", duration=60.0), video("long.mkv", duration=3000.0), video("zero.mkv", duration=0.0), video("nan.mkv", duration=1.0)]
        library[3].duration = float("nan")
        picked = ls.pick_sample(library, rng=random.Random(4))
        self.assertEqual(picked[0].meta.path.name, "long.mkv")
        self.assertNotIn("zero.mkv", [p.meta.path.name for p in picked])
        self.assertNotIn("nan.mkv", [p.meta.path.name for p in picked])

    def test_the_segment_is_two_minutes_from_the_middle(self) -> None:
        self.assertEqual(ls.segment_of(video("a.mkv", duration=3600.0)), (1740.0, 120.0))
        self.assertEqual(ls.segment_of(video("a.mkv", duration=50.0)), (0.0, 50.0))
        self.assertEqual(diagnostic_segments(3600.0, "sample"), [(1740.0, 120.0)])
        self.assertEqual(diagnostic_segments(90.0, "sample"), [(0.0, 90.0)])

    def test_the_time_estimate_is_a_range_that_grows_with_the_work(self) -> None:
        few, many = ls.estimate_seconds([video("a.mkv")], 2), ls.estimate_seconds([video(f"{i}.mkv") for i in range(10)], 2)
        self.assertLess(few[0], few[1])
        self.assertLess(few[0], many[0])


class Advice(unittest.TestCase):
    def kf_rows(self, count=4, all_s=20.0, kf_s=5.0, share=0.0, failed=0):
        rows = [{"video": f"/v/{i}.mkv", "window": 1, "backend": "cuda", "status": "PASS", "all_seconds": all_s, "keyframe_seconds": kf_s, "share": share}
                for i in range(count - failed)]
        rows += [{"video": f"/v/f{i}.rmvb", "window": 1, "backend": "cpu", "status": "FAILED", "all_seconds": all_s, "keyframe_seconds": 0.5, "error_summary": "boom"}
                 for i in range(failed)]
        return rows

    def config(self, **kw):
        return {"trickplay": {"keyframe_only": False, "keyframe_repeat_limit_percent": 10.0, **kw.pop("trickplay", {})},
                "acceleration": {"max_gpu_jobs": 2, **kw.pop("acceleration", {})}}

    def test_key_frames_are_suggested_when_they_are_much_faster(self) -> None:
        advice = ba.build_advice({"keyframe_rows": self.kf_rows()}, self.config())
        self.assertEqual([(a.key, a.current, a.proposed) for a in advice], [("keyframe_only", "off", "on")])
        self.assertIn("4.00x faster", advice[0].reason)

    def test_files_that_must_be_redone_and_files_that_cannot_use_it_eat_the_gain(self) -> None:
        rows = self.kf_rows(4, share=50.0)                      # all four need a redo: 5 s + 20 s instead of 20 s
        self.assertEqual(ba.build_advice({"keyframe_rows": rows}, self.config()), [])
        summary = ba.keyframe_summary({"keyframe_rows": rows}, 10.0)
        self.assertEqual((summary["sparse"], summary["failed"]), (4, 0))
        mixed = ba.keyframe_summary({"keyframe_rows": self.kf_rows(4, failed=1)}, 10.0)
        self.assertEqual((mixed["ok"], mixed["failed"]), (3, 1))

    def test_with_the_redo_switched_off_sparse_files_are_not_counted_as_redone(self) -> None:
        self.assertEqual(ba.keyframe_summary({"keyframe_rows": self.kf_rows(4, share=50.0)}, 0.0)["sparse"], 0)

    def test_it_suggests_turning_it_off_when_it_does_not_pay(self) -> None:
        advice = ba.build_advice({"keyframe_rows": self.kf_rows(4, all_s=10.0, kf_s=9.5)}, self.config(trickplay={"keyframe_only": True}))
        self.assertEqual([(a.key, a.proposed) for a in advice], [("keyframe_only", "off")])

    def test_nothing_is_suggested_when_it_already_fits_or_too_few_files_were_measured(self) -> None:
        self.assertEqual(ba.build_advice({"keyframe_rows": self.kf_rows()}, self.config(trickplay={"keyframe_only": True})), [])
        self.assertEqual(ba.build_advice({"keyframe_rows": self.kf_rows(2)}, self.config()), [])
        self.assertEqual(ba.build_advice({}, self.config()), [])

    def test_a_repeat_limit_of_zero_is_questioned_when_files_would_keep_repeated_thumbnails(self) -> None:
        rows = self.kf_rows(4, share=30.0)
        advice = ba.build_advice({"keyframe_rows": rows}, self.config(trickplay={"keyframe_only": True, "keyframe_repeat_limit_percent": 0.0}))
        self.assertEqual([a.key for a in advice], ["keyframe_repeat_limit_percent"])
        config = self.config(trickplay={"keyframe_only": True, "keyframe_repeat_limit_percent": 0.0})
        ba.apply_advice(config, advice)
        self.assertEqual(config["trickplay"]["keyframe_repeat_limit_percent"], 10.0)

    def runs(self, *throughputs):
        return {"video": "/v/a.mkv", "backend": "cuda", "seconds": 60.0, "runs": [{"jobs": i, "wall": 10.0, "throughput": t} for i, t in enumerate(throughputs, 1)]}

    def test_gpu_jobs_stop_where_one_more_brings_little(self) -> None:
        self.assertEqual(ba.best_gpu_jobs(self.runs(6.0, 11.0, 11.5)), 2)
        self.assertEqual(ba.best_gpu_jobs(self.runs(6.0, 11.0, 15.0, 15.5)), 3)
        self.assertEqual(ba.best_gpu_jobs(self.runs(6.0, 6.2)), 1)
        # fewer jobs are never suggested: the test sees the GPU alone, a real job also waits for the drive and packs on the CPU
        self.assertEqual(ba.build_advice({"gpu_parallel": self.runs(6.0, 6.2)}, self.config()), [])
        advice = ba.build_advice({"gpu_parallel": self.runs(6.0, 11.0, 15.0, 15.5)}, self.config())
        self.assertEqual([(a.key, a.current, a.proposed) for a in advice], [("max_gpu_jobs", 2, 3)])
        self.assertEqual(ba.build_advice({"gpu_parallel": self.runs(6.0, 11.0, 11.5)}, self.config()), [], "2 is what is set")
        self.assertEqual(ba.build_advice({"gpu_parallel": {**self.runs(6.0), "error": "boom"}}, self.config()), [])

    @staticmethod
    def disks(*rows):
        """The disks of the library as the benchmark records them: (kind, videos at once, bytes, seconds)."""
        return [{"group": f"PHYSICAL_DISK_{i}", "disk": f"{'XYZ'[i]}:", "kind": kind, "workers": workers, "bytes": size, "seconds": seconds,
                 "mb_per_second": round(size / (1024 * 1024) / seconds, 1) if seconds else None} for i, (kind, workers, size, seconds) in enumerate(rows)]

    def test_more_gpu_jobs_than_videos_at_once_are_not_suggested(self) -> None:
        data = {"gpu_parallel": self.runs(6.0, 11.0, 15.0, 15.5)}
        one_hdd = {**data, "storage": self.disks(("hdd", 1, 0, 0))}
        self.assertEqual(ba.build_advice(one_hdd, self.config(acceleration={"max_gpu_jobs": 1})), [], "one video at a time never uses a second job")
        ssd = {**data, "storage": self.disks(("ssd", 2, 0, 0))}
        self.assertEqual([(a.current, a.proposed) for a in ba.build_advice(ssd, self.config(acceleration={"max_gpu_jobs": 1}))], [(1, 2)], "2 videos at once: 2 jobs, not 3")
        self.assertEqual(ba.build_advice(ssd, self.config()), [], "2 are set")
        two_disks = {**data, "storage": self.disks(("ssd", 2, 0, 0), ("hdd", 1, 0, 0))}
        self.assertEqual([(a.current, a.proposed) for a in ba.build_advice(two_disks, self.config())], [(2, 3)], "3 videos at once")
        self.assertEqual(ba.videos_at_once(two_disks), 3)
        self.assertIsNone(ba.videos_at_once(data))

    def test_the_gain_in_the_reason_is_the_one_of_the_suggested_number_of_jobs(self) -> None:
        data = {"gpu_parallel": self.runs(6.0, 9.0, 15.0), "storage": self.disks(("ssd", 2, 0, 0), ("ssd", 2, 0, 0))}
        advice = ba.build_advice(data, self.config(acceleration={"max_gpu_jobs": 1}))
        self.assertEqual([(a.current, a.proposed) for a in advice], [(1, 3)])
        self.assertIn("2.50x", advice[0].reason)

    def test_the_disks_are_listed_in_the_report_and_on_the_screen(self) -> None:
        data = {"storage": self.disks(("ssd", 2, 600 * 1024 * 1024, 0.5), ("hdd", 1, 300 * 1024 * 1024, 3.0)), "gpu_parallel": self.runs(6.0, 11.0)}
        report = "\n".join(ba.storage_lines(data))
        self.assertIn("X: SSD: 2 videos at once, the copies were read at 1200 MB/s", report)
        self.assertIn("Y: HDD: 1 video at once, the copies were read at 100 MB/s", report)
        screen = "\n".join(ba.screen_lines(data, self.config(), Path("/r/result.txt")))
        self.assertIn("X: SSD: 2 at once, reads 1200 MB/s", screen)
        self.assertIn("Y: HDD: 1 at once, reads 100 MB/s", screen)
        self.assertNotIn("Your disks give one video at a time", screen)
        single = "\n".join(ba.screen_lines({"storage": self.disks(("hdd", 1, 0, 0)), "gpu_parallel": self.runs(6.0, 11.0)}, self.config(), Path("/r/result.txt")))
        self.assertIn("Your disks give one video at a time", single)

    def test_the_read_speed_of_every_kind_of_disk_is_kept_for_the_time_estimate(self) -> None:
        config: dict = {}
        ba.remember_disk_speeds(config, {"storage": self.disks(("hdd", 1, 100 * 1024 * 1024, 1.0), ("hdd", 1, 300 * 1024 * 1024, 1.0), ("ssd", 2, 0, 0))})
        self.assertEqual(config["disk_speed"], {"hdd": 200.0}, "bytes over seconds of all the disks of that kind; none for one that was not read")
        ba.remember_disk_speeds(config, {"storage": self.disks(("hdd", 1, 50 * 1024 * 1024, 1.0))})
        self.assertEqual(config["disk_speed"]["hdd"], 50.0, "the newest measurement")
        ba.remember_disk_speeds(config, {})
        self.assertEqual(config["disk_speed"]["hdd"], 50.0)

    def test_applying_changes_only_what_was_suggested(self) -> None:
        config = self.config()
        ba.apply_advice(config, ba.build_advice({"keyframe_rows": self.kf_rows(), "gpu_parallel": self.runs(6.0, 11.0, 15.0, 15.5)}, config))
        self.assertEqual((config["trickplay"]["keyframe_only"], config["acceleration"]["max_gpu_jobs"]), (True, 3))
        self.assertEqual(config["trickplay"]["keyframe_repeat_limit_percent"], 10.0)

    def test_the_tables_read_well(self) -> None:
        data = {"keyframe_rows": self.kf_rows(2, share=30.0, failed=1), "gpu_parallel": self.runs(6.0, 11.0)}
        text = "\n".join(ba.keyframe_lines(data, 10.0))
        self.assertIn("4.00x", text)
        self.assertIn("key frames too far apart", text)
        self.assertIn("not possible for this file: boom", text)
        self.assertIn("1.83x of one job", "\n".join(ba.parallel_lines(data)))
        self.assertEqual(ba.keyframe_lines({}, 10.0), [])
        self.assertEqual(ba.parallel_lines({}), [])
        self.assertIn("Extract key frames only", "\n".join(ba.advice_lines(ba.build_advice({"keyframe_rows": self.kf_rows()}, self.config()))))


def bench_rows(kind: str, video: str, cpu: float, paths: dict[str, object]) -> list[dict]:
    """Rows of one video: paths {backend: seconds | "FAILED"}."""
    rows = [{"video": video, "kind": kind, "window": 1, "backend": "cpu", "status": "PASS", "extract_seconds": cpu, "pack_seconds": 0.0, "frames": 12}]
    for backend, seconds in paths.items():
        if seconds == "FAILED": rows.append({"video": video, "kind": kind, "window": 1, "backend": backend, "status": "FAILED", "extract_seconds": 0.0, "pack_seconds": 0.0})
        else: rows.append({"video": video, "kind": kind, "window": 1, "backend": backend, "status": "PASS", "extract_seconds": seconds, "pack_seconds": 0.0,
                           "frames": 12, "matching_frame_count": True})
    return rows


class PathChoice(unittest.TestCase):
    def data(self):
        rows = (bench_rows("hevc 10-bit 4K SDR", "/v/a.mkv", 12.0, {"cuda": 2.0, "d3d11va": 6.0})
                + bench_rows("hevc 10-bit 4K SDR", "/v/b.mkv", 10.0, {"cuda": 2.0, "d3d11va": 6.0})
                + bench_rows("h264 8-bit SD SDR", "/v/c.mp4", 0.5, {"cuda": 0.9, "d3d11va": 1.8})
                + bench_rows("rv40 8-bit SD SDR", "/v/d.rmvb", 0.5, {"cuda": "FAILED", "d3d11va": 0.9})
                + bench_rows("h264 8-bit 1080p SDR", "/v/e.mkv", 3.0, {"cuda": 2.8, "d3d11va": 4.0}))
        return {"rows": rows, "backends": ["cuda", "d3d11va"], "ffmpeg_hash": "H", "gpu_names": ["RTX"], "keyframe_rows": []}

    def test_the_fastest_path_per_kind_and_the_cpu_where_no_gpu_path_is_clearly_faster(self) -> None:
        results = pc.from_benchmark(self.data(), 1.15)
        self.assertEqual({k: v["best"] for k, v in results.items()},
                         {"hevc 10-bit 4K SDR": "cuda", "h264 8-bit SD SDR": "cpu", "rv40 8-bit SD SDR": "cpu", "h264 8-bit 1080p SDR": "cpu"})
        self.assertEqual(results["hevc 10-bit 4K SDR"]["files"], 2)
        self.assertEqual(results["hevc 10-bit 4K SDR"]["speedup"], 5.5, "22 s of CPU against 4 s")
        self.assertEqual(pc.from_benchmark(self.data(), 1.05)["h264 8-bit 1080p SDR"]["best"], "cuda", "3.0 s against 2.8 s passes a lower bar")

    def test_a_path_that_failed_on_a_file_of_the_kind_is_not_offered_for_it(self) -> None:
        data = self.data()
        data["rows"] += bench_rows("hevc 10-bit 4K SDR", "/v/f.mkv", 10.0, {"cuda": "FAILED", "d3d11va": 6.0})
        self.assertEqual(pc.from_benchmark(data, 1.15)["hevc 10-bit 4K SDR"]["best"], "d3d11va", "cuda failed on one file, so only d3d11va is offered for the kind")
        data["rows"] += bench_rows("hevc 10-bit 4K SDR", "/v/g.mkv", 10.0, {"cuda": 2.0, "d3d11va": "FAILED"})
        self.assertEqual(pc.from_benchmark(data, 1.15)["hevc 10-bit 4K SDR"]["best"], "cpu", "each GPU path failed on some file of the kind")

    def test_files_without_a_reference_are_ignored(self) -> None:
        data = {"rows": [{"video": "/v/x", "kind": "k", "window": 1, "backend": "cpu", "status": "FAILED", "extract_seconds": 0, "pack_seconds": 0}], "backends": ["cuda"]}
        self.assertEqual(pc.from_benchmark(data, 1.15), {})
        self.assertEqual(pc.from_benchmark({"rows": [{"video": "/v", "backend": "cpu"}]}, 1.15), {}, "reports from before the kinds were recorded")

    def test_the_choice_belongs_to_one_ffmpeg_and_card(self) -> None:
        stored = pc.merged(None, pc.from_benchmark(self.data(), 1.15), "H", ["RTX"])
        m = video("a.mkv", "hevc", 10, 3840, 2160)
        self.assertEqual(pc.kind_of(m), "hevc 10-bit 4K SDR")
        self.assertEqual(pc.choice_for(stored, "H", ["RTX"], m), "cuda")
        self.assertIsNone(pc.choice_for(stored, "other", ["RTX"], m))
        self.assertIsNone(pc.choice_for(stored, "H", ["Arc"], m))
        self.assertIsNone(pc.choice_for(stored, "H", ["RTX"], video("b.mkv", "av1", 10)), "not measured")
        self.assertIsNone(pc.choice_for(None, "H", ["RTX"], m))
        again = pc.merged(stored, {"av1 10-bit 4K SDR": {"best": "cpu", "speedup": 1.0}}, "H", ["RTX"])
        self.assertEqual(sorted(again["results"]), ["av1 10-bit 4K SDR", "h264 8-bit 1080p SDR", "h264 8-bit SD SDR", "hevc 10-bit 4K SDR", "rv40 8-bit SD SDR"])
        self.assertEqual(list(pc.merged(stored, {"x": {"best": "cpu", "speedup": 1.0}}, "new build", ["RTX"])["results"]), ["x"], "old results of another build are dropped")

    def test_it_is_suggested_once_and_applied_after_a_yes(self) -> None:
        data = self.data()
        config = {"trickplay": {}, "acceleration": {"gpu_speed_threshold": 1.15}}
        advice = [a for a in ba.build_advice(data, config) if a.key == "path_choice"]
        self.assertEqual(len(advice), 1)
        self.assertEqual(advice[0].proposed, "CPU for 3 kinds, GPU for 1 kind")
        ba.apply_advice(config, advice)
        self.assertEqual(pc.choice_for(config["acceleration"]["path_choice"], "H", ["RTX"], video("a.mkv", "hevc", 10, 3840, 2160)), "cuda")
        self.assertEqual([a for a in ba.build_advice(data, config) if a.key == "path_choice"], [], "already what is saved")
        self.assertIn("FASTEST PATH PER KIND OF VIDEO", "\n".join(ba.path_lines(data, config)))
        self.assertIn("cuda (5.50x)", "\n".join(ba.path_lines(data, config)))

    def kf_rows(self, n=4, share=0.0):
        return [{"video": f"/v/{i}.mkv", "window": 1, "backend": "cuda", "status": "PASS", "all_seconds": 20.0, "keyframe_seconds": 5.0, "share": share} for i in range(n)]

    def test_the_screen_shows_the_deciding_facts_and_leaves_the_rest_to_the_report(self) -> None:
        data = {**self.data(), "keyframe_rows": self.kf_rows(), "read_speed": [{"video": "/v/a.mkv", "bytes": 100 * 1024 * 1024, "seconds": 2.0, "mb_per_second": 50.0}],
                "gpu_parallel": {"video": "/v/a.mkv", "backend": "cuda", "seconds": 60.0, "runs": [{"jobs": 1, "wall": 1.0, "throughput": 60.0}, {"jobs": 2, "wall": 1.4, "throughput": 86.0}]}}
        text = "\n".join(ba.screen_lines(data, {"trickplay": {}, "acceleration": {}}, Path("/r/result.txt")))
        text = " ".join(text.split())                                                    # a line that is wrapped to the width is still one sentence
        self.assertIn("cuda (5.50x)", text)
        self.assertIn("Key frames only: 4.00x faster on 4 files", text)
        self.assertIn("Several videos at once on the GPU (cuda): 1 video, 2 videos = 1.43x of one video", text)
        self.assertIn("The library drive reads about 50 MB/s", text)
        self.assertTrue(text.endswith(f"Every measurement is in the report: {Path('/r/result.txt')}"))
        for left_out in ("FILE", "CPU VS GPU SPEED COMPARISON", "Recommendation", "SOURCE READ SPEED"): self.assertNotIn(left_out, text)

    def test_without_a_gpu_path_there_is_no_choice_between_paths_to_show_or_suggest(self) -> None:
        data = {**self.data(), "backends": [], "keyframe_rows": self.kf_rows()}
        self.assertNotIn("Kind of video", "\n".join(ba.screen_lines(data, {"trickplay": {}, "acceleration": {}}, Path("/r/result.txt"))))
        self.assertEqual([a for a in ba.build_advice(data, {"trickplay": {}, "acceleration": {}}) if a.key == "path_choice"], [])

    def test_a_changed_choice_says_which_kind_moves_where(self) -> None:
        config = {"trickplay": {}, "acceleration": {"gpu_speed_threshold": 1.15}}
        saved = {"hevc 10-bit 4K SDR": {"best": "cpu", "speedup": 1.0}, "h264 8-bit SD SDR": {"best": "cpu", "speedup": 1.0}}
        config["acceleration"]["path_choice"] = pc.merged(None, saved, "H", ["RTX"])
        advice = [x for x in ba.build_advice(self.data(), config) if x.key == "path_choice"][0]
        self.assertEqual(advice.details, ["hevc 10-bit 4K SDR: CPU -> cuda (5.50x faster than the CPU)"], "the kind that stays on the CPU is not listed")
        text = "\n".join(ba.advice_lines([advice]))
        self.assertIn("Now:        CPU for 2 kinds", text)
        self.assertIn("Moves:      hevc 10-bit 4K SDR: CPU -> cuda", text)

    def test_a_saved_choice_is_not_changed_when_the_new_measurement_finds_no_real_difference(self) -> None:
        config = {"trickplay": {}, "acceleration": {"gpu_speed_threshold": 1.15}}
        config["acceleration"]["path_choice"] = pc.merged(None, {"k 8-bit SD SDR": {"best": "cuda", "speedup": 1.3}}, "H", ["RTX"])
        even = {"rows": bench_rows("k 8-bit SD SDR", "/v/a.mkv", 1.0, {"cuda": 1.0}), "backends": ["cuda"], "ffmpeg_hash": "H", "gpu_names": ["RTX"], "keyframe_rows": []}
        self.assertEqual([x for x in ba.build_advice(even, config) if x.key == "path_choice"], [], "0.86 s against 0.86 s is no reason to move a kind")
        clearly = {**even, "rows": bench_rows("k 8-bit SD SDR", "/v/a.mkv", 0.5, {"cuda": 1.0})}
        moved = [x for x in ba.build_advice(clearly, config) if x.key == "path_choice"]
        self.assertEqual(moved[0].details, ["k 8-bit SD SDR: cuda -> CPU"])

    def test_the_key_frame_detail_names_only_what_applies(self) -> None:
        base = {"gain": 2.63, "files": 10, "sparse": 0, "failed": 0}
        self.assertEqual(ba.keyframe_detail(base), "2.63x faster on 10 files")
        self.assertEqual(ba.keyframe_detail({**base, "sparse": 2, "failed": 1}), "2.63x faster on 10 files; 2 files have key frames too far apart and are made the slow way anyway, 1 file cannot use key frames at all")

    def test_a_suggestion_is_one_block_not_a_wide_table(self) -> None:
        advice = ba.build_advice({"keyframe_rows": self.kf_rows()}, {"trickplay": {}, "acceleration": {}})
        lines = ba.advice_lines(advice)
        self.assertEqual(lines[:3], ["[1] Extract key frames only", "    Now:        off", "    Suggested:  on"])
        self.assertTrue(lines[3].startswith("    Why:        4.00x faster on 4 files"))
        self.assertTrue(all(len(line) < 140 for line in lines))

    def test_a_choice_grows_with_the_kinds_that_are_already_saved(self) -> None:
        config = {"trickplay": {}, "acceleration": {"gpu_speed_threshold": 1.15}}
        first = [a for a in ba.build_advice(self.data(), config) if a.key == "path_choice"]
        ba.apply_advice(config, first)
        more = {**self.data(), "rows": bench_rows("av1 10-bit 1080p SDR", "/v/f.mkv", 10.0, {"cuda": 2.0, "d3d11va": 6.0})}
        second = [a for a in ba.build_advice(more, config) if a.key == "path_choice"]
        self.assertEqual((second[0].current, second[0].proposed), ("CPU for 3 kinds, GPU for 1 kind", "CPU for 3 kinds, GPU for 2 kinds"))


@needs_ffmpeg
class GenerateUsesTheChoice(PipelineBase):
    def backends_for(self, stored, mode="auto", meta_kw=None):
        gen = self.generator()
        self.config["acceleration"]["path_choice"] = stored
        make_video(self.media / "a.mp4", 12)
        m = self.scan()[0]
        m = dataclasses.replace(m, **{"codec": "hevc", "bits": 10, "pix_fmt": "yuv420p10le", "width": 3840, "height": 2160, "resolution": "", **(meta_kw or {})})
        with mock.patch.object(type(self.tools), "backends", return_value=["cuda", "d3d11va"]):
            return gen._backends_for(m, mode), m

    def saved(self, best):
        return pc.merged(None, {"hevc 10-bit 4K SDR": {"best": best, "speedup": 3.0}}, self.tools.candidate.sha256, list(self.tools.gpu_names))

    def test_the_cpu_choice_skips_the_gpu_and_a_gpu_choice_goes_first(self) -> None:
        self.assertEqual(self.backends_for(self.saved("cpu"))[0], [])
        self.assertEqual(self.backends_for(self.saved("d3d11va"))[0], ["d3d11va", "cuda"])
        self.assertEqual(self.backends_for(self.saved("cuda"))[0], ["cuda", "d3d11va"])

    def test_nothing_changes_without_a_choice_for_the_kind_in_the_other_modes_or_for_another_ffmpeg(self) -> None:
        self.assertEqual(self.backends_for(None)[0], ["cuda", "d3d11va"])
        self.assertEqual(self.backends_for(self.saved("cpu"), mode="gpu")[0], ["cuda", "d3d11va"], "GPU only is the person's decision")
        other = pc.merged(None, {"hevc 10-bit 4K SDR": {"best": "cpu", "speedup": 1.0}}, "other build", list(self.tools.gpu_names))
        self.assertEqual(self.backends_for(other)[0], ["cuda", "d3d11va"])
        self.assertEqual(self.backends_for(self.saved("cpu"), meta_kw={"bits": 8, "pix_fmt": "yuv420p"})[0], ["cuda", "d3d11va"], "another kind: not measured")

    def test_a_video_of_a_cpu_kind_is_extracted_on_the_cpu_without_trying_the_gpu(self) -> None:
        make_video(self.media / "a.mp4", 12)
        gen = self.generator()
        m = self.scan()[0]
        kind = pc.kind_of(m)
        self.config["acceleration"]["path_choice"] = pc.merged(None, {kind: {"best": "cpu", "speedup": 1.0}}, self.tools.candidate.sha256, list(self.tools.gpu_names))
        self.config["acceleration"]["mode"] = "auto"
        calls: list[str] = []
        real = TrickplayGenerator._attempt_extract_with_retry

        def spy(self_, meta, frames_dir, method, backend, label):
            calls.append(method)
            return real(self_, meta, frames_dir, method, backend, label)

        with mock.patch.object(type(self.tools), "backends", return_value=["cuda"]), mock.patch.object(TrickplayGenerator, "_attempt_extract_with_retry", spy), quiet():
            result = gen._choose_method_and_extract(m, self.root / "frames", "t")
        self.assertEqual(calls, ["CPU"])
        self.assertTrue(result.success)

    def test_the_rows_of_a_benchmark_record_the_kind(self) -> None:
        make_video(self.media / "a.mp4", 40)
        job = DiagnosticJob("benchmark")
        self.addCleanup(job.close)
        with quiet():
            data = DiagnosticRunner(self.generator(), job).run(self.scan(), "quick", [], "benchmark")
        self.assertTrue(all(r["kind"].startswith("h264 8-bit") for r in data["rows"]), data["rows"][0]["kind"])
        self.assertIn("FASTEST PATH PER KIND OF VIDEO", job.result_txt.read_text(encoding="utf-8")) if len(data["backends"]) else None


    def test_the_benchmark_shows_one_row_per_file_and_keeps_the_details_for_the_report(self) -> None:
        import contextlib
        import io
        make_video(self.media / "a.mp4", 40)
        make_video(self.media / "b.mp4", 40)
        job = DiagnosticJob("benchmark")
        self.addCleanup(job.close)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            DiagnosticRunner(self.generator(), job).run(self.scan(), "quick", [], "benchmark")
        lines = out.getvalue().splitlines()
        self.assertIn("BENCHMARK", lines)
        self.assertTrue(any("Kind of video" in line and "Key frames" in line for line in lines), lines)
        rows = [line for line in lines if line.strip().startswith(("1/2", "2/2"))]
        self.assertEqual(len(rows), 2, lines)
        self.assertTrue(all("h264  8-bit  SD  SDR" in row and "faster (CPU)" in row for row in rows), rows)
        for detail in ("FILE ", "Segment ", "KEY FRAMES on", "Copied the segment", "Warm-up"): self.assertNotIn(detail, out.getvalue())
        self.assertTrue(all(len(line) <= 100 for line in lines), [line for line in lines if len(line) > 100])
        report = job.log_path.read_text(encoding="utf-8")                    # the details are in the report
        for detail in ("FILE 1/2", "Segment 1/1", "KEY FRAMES on"): self.assertIn(detail, report)


@needs_ffmpeg
class RealCopies(PipelineBase):
    def test_a_segment_is_copied_without_re_encoding_and_removed(self) -> None:
        make_video(self.media / "a.mp4", 40)
        original = self.scan()[0]
        staged = ls.stage_segment(self.tools.ffmpeg, original, 10.0, 20.0, self.root / "temp_copy")
        self.assertTrue(staged.path.exists())
        self.assertEqual(staged.meta.path, staged.path)
        self.assertEqual(staged.meta.codec, original.codec)
        self.assertEqual(staged.duration, 20.0)
        self.assertGreater(staged.read_speed, 0)
        staged.remove()
        self.assertFalse(staged.path.exists())

    def test_a_copy_that_cannot_be_made_says_why(self) -> None:
        make_video(self.media / "a.mp4", 40)
        original = self.scan()[0]
        original.path.unlink()
        with self.assertRaisesRegex(RuntimeError, "Copying the segment failed"):
            ls.stage_segment(self.tools.ffmpeg, original, 10.0, 20.0, self.root / "temp_copy")

    def test_not_enough_space_is_reported_before_copying(self) -> None:
        make_video(self.media / "a.mp4", 40)
        original = self.scan()[0]
        with mock.patch("shutil.disk_usage", return_value=mock.Mock(free=1000)), self.assertRaisesRegex(RuntimeError, "Not enough free space"):
            ls.stage_segment(self.tools.ffmpeg, original, 10.0, 20.0, self.root / "temp_copy")


@needs_ffmpeg
class RealBenchmark(PipelineBase):
    def run_benchmark(self, mode: str = "quick", backends=(), **kw):
        gen = self.generator()
        job = DiagnosticJob("benchmark")
        self.addCleanup(job.close)
        with quiet():
            return DiagnosticRunner(gen, job).run(self.scan(), mode, list(backends), "benchmark", **kw), job

    def test_every_file_is_also_run_with_key_frames_only_and_the_repeats_are_counted(self) -> None:
        make_video(self.media / "dense.mp4", 40, gop=10)
        make_video(self.media / "sparse.mp4", 40, gop=100)
        data, job = self.run_benchmark()
        self.assertEqual(data["status"], "COMPLETE")
        rows = {Path(r["video"]).name: r for r in data["keyframe_rows"]}
        self.assertEqual(sorted(rows), ["dense.mp4", "sparse.mp4"])
        self.assertTrue(all(r["status"] == "PASS" for r in rows.values()), rows)
        self.assertGreater(rows["sparse.mp4"]["share"], 30.0, "a key frame every 10 s for thumbnails every 5 s: half of them repeat")
        self.assertLess(rows["dense.mp4"]["share"], rows["sparse.mp4"]["share"])
        self.assertGreater(rows["dense.mp4"]["all_seconds"], 0)
        self.assertTrue(all(r["status"] == "PASS" and "variant" not in r for r in data["rows"]), "the A/B run is not mixed into the path measurements")
        self.assertIn("KEY FRAMES ONLY vs ALL FRAMES", job.result_txt.read_text(encoding="utf-8"))

    def test_a_path_the_compatibility_check_ruled_out_is_not_timed(self) -> None:
        make_video(self.media / "a.mp4", 40)
        gen = self.generator()
        job = DiagnosticJob("benchmark")
        self.addCleanup(job.close)
        runner = DiagnosticRunner(gen, job)
        measured: list[object] = []
        original = runner._measure
        with mock.patch.object(gen, "_gpu_decode_blocked", side_effect=lambda meta, backend: backend == "cuda"), \
                mock.patch.object(runner, "_measure", side_effect=lambda data, meta, kind, mode, window, backend: measured.append(backend) or original(data, meta, kind, mode, window, backend)), \
                quiet():
            data = runner.run(self.scan(), "quick", ["cuda", "d3d11va"], "benchmark")
        self.assertNotIn("cuda", measured)
        self.assertIn("d3d11va", measured)
        self.assertEqual(data["skipped_paths"][0]["backend"], "cuda")
        self.assertIn("does not decode this kind of video on the GPU", job.log_path.read_text(encoding="utf-8"))

    def test_the_benchmark_records_the_kind_of_disk_the_videos_workers_and_how_fast_it_read(self) -> None:
        make_video(self.media / "a.mp4", 40)
        for kind, workers in (("hdd", 1), ("ssd", 2)):
            with mock.patch("modules.storage.storage_kind", return_value=kind):
                data, job = self.run_benchmark("sample", local_copy=True, sample=[{"video": "x", "kind": "k", "why": "k"}])
            (disk,) = data["storage"]
            self.assertEqual((disk["kind"], disk["workers"]), (kind, workers))
            self.assertGreater(disk["bytes"], 0)
            self.assertGreater(disk["mb_per_second"], 0)
            self.assertIn(f"{kind.upper()}: {workers} video", "\n".join(ba.storage_lines(data)))
        self.assertIn("DISKS", job.result_txt.read_text(encoding="utf-8"))

    def test_segments_are_measured_on_local_copies_that_do_not_stay(self) -> None:
        make_video(self.media / "a.mp4", 40)
        seen: list[Path] = []
        real = TrickplayGenerator._run_ffmpeg_extract

        def spy(self_, meta, frames_dir, method, backend, **kw):
            seen.append(meta.path)
            return real(self_, meta, frames_dir, method, backend, **kw)

        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", spy):
            data, job = self.run_benchmark("sample", local_copy=True, sample=[{"video": "x", "kind": "k", "why": "k"}])
        self.assertEqual(data["status"], "COMPLETE")
        self.assertTrue(seen and all(p.parent == constants.DIRS["temp"] for p in seen), seen)
        self.assertFalse((constants.DIRS["temp"] / "benchmark_segment.mkv").exists())
        self.assertEqual(len(data["read_speed"]), 1)
        self.assertGreater(data["read_speed"][0]["mb_per_second"], 0)
        self.assertTrue(all(r["local_copy"] and r["video"].endswith("a.mp4") for r in data["rows"]), "the report names the original")
        self.assertIn("SOURCE READ SPEED", job.result_txt.read_text(encoding="utf-8"))
        self.assertEqual(data["sample"][0]["why"], "k")

    def test_a_segment_that_cannot_be_copied_is_tested_on_the_library_drive_and_says_so(self) -> None:
        make_video(self.media / "a.mp4", 40)
        make_video(self.media / "b.mp4", 40)
        real = ls.stage_segment

        def flaky(ffmpeg, meta, *a, **kw):
            if meta.path.name == "a.mp4": raise RuntimeError("Copying the segment failed: disk on fire")
            return real(ffmpeg, meta, *a, **kw)

        with mock.patch.object(ls, "stage_segment", flaky):
            data, job = self.run_benchmark("sample", local_copy=True)
        self.assertEqual(data["status"], "COMPLETE")
        self.assertTrue(all(r["status"] == "PASS" for r in data["rows"]), data["rows"])
        self.assertEqual({Path(r["video"]).name: r["local_copy"] for r in data["rows"]}, {"a.mp4": False, "b.mp4": True})
        self.assertEqual([Path(p).name for p in data["direct_files"]], ["a.mp4"])
        self.assertEqual([Path(r["video"]).name for r in data["read_speed"]], ["b.mp4"], "no read speed for what was not copied")
        self.assertEqual(sorted(Path(r["video"]).name for r in data["keyframe_rows"]), ["a.mp4", "b.mp4"])
        self.assertIn("TESTED ON THE LIBRARY DRIVE", job.result_txt.read_text(encoding="utf-8"))

    def test_files_whose_packets_have_no_time_stamps_can_still_be_copied(self) -> None:
        # MPEG-1 and MPEG-4 in AVI with B-frames: "Can't write packet with unknown timestamp" without +genpts
        for name, codec in (("a.mpg", "mpeg1video"), ("b.avi", "mpeg4")):
            path = self.media / name
            subprocess.run([self.tools.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25", "-t", "30", "-c:v", codec, "-bf", "2", "-y", str(path)], check=True)
            original = next(m for m in self.scan() if m.path == path)
            staged = ls.stage_segment(self.tools.ffmpeg, original, 5.0, 10.0, self.root / "temp_copy")
            self.assertGreater(staged.read_bytes, 1000, name)
            staged.remove()

    @staticmethod
    def gpu_wins(self_, data, meta, mode, window):
        """Stands in for the key frame run: records that the GPU was the fastest path of this segment, as a weight for the parallel test."""
        data["keyframe_rows"].append({"video": str(meta.path), "window": window["window_no"], "backend": "cuda", "status": "PASS",
                                      "all_seconds": 10.0 if meta.path.name == "b.mp4" else 1.0, "keyframe_seconds": 1.0, "share": 0.0})
        return "cuda"

    def test_the_parallel_jobs_test_starts_with_the_file_where_the_gpu_had_the_most_to_do(self) -> None:
        for name in ("a.mp4", "b.mp4"): make_video(self.media / name, 40)
        real = TrickplayGenerator._run_ffmpeg_extract

        def fake_gpu(self_, meta, frames_dir, method, backend, **kw):
            return real(self_, meta, frames_dir, "CPU", None, **kw)

        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", fake_gpu), mock.patch.object(DiagnosticRunner, "_measure_keyframes", self.gpu_wins):
            data, _ = self.run_benchmark("sample", ["cuda"], local_copy=True)
        test = data["gpu_parallel"]
        self.assertEqual(test["videos"][0], "b.mp4")
        self.assertEqual(sorted(test["videos"]), ["a.mp4", "b.mp4"], "the same kind of video again fills the test up to the segments it needs")
        self.assertFalse((constants.DIRS["temp"] / "benchmark_segment.mkv").exists())
        self.assertEqual([p.name for p in constants.DIRS["temp"].iterdir() if p.name.startswith("parallel_")], [], "the copies are removed")

    def test_a_gpu_path_that_wins_gets_the_parallel_jobs_test(self) -> None:
        for name in ("a.mp4", "b.mp4", "c.mp4"): make_video(self.media / name, 40)
        real = TrickplayGenerator._run_ffmpeg_extract
        jobs: list[str] = []

        def fake_gpu(self_, meta, frames_dir, method, backend, **kw):
            label = kw.get("progress_label", "")
            if "GPU job" in label: jobs.append(label)
            return real(self_, meta, frames_dir, "CPU", None, **kw)

        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", fake_gpu), \
                mock.patch.object(DiagnosticRunner, "_measure_keyframes", self.gpu_wins):
            data, job = self.run_benchmark("quick", ["cuda"])
        test = data["gpu_parallel"]
        self.assertEqual((test["backend"], test["segments"]), ("cuda", 3))
        self.assertGreaterEqual(len(test["runs"]), 2)
        self.assertEqual([r["jobs"] for r in test["runs"]], list(range(1, len(test["runs"]) + 1)))
        self.assertLessEqual(len(test["runs"]), 3, "a job needs a segment of its own")
        self.assertEqual(len(jobs), 3 * len(test["runs"]), "every run works through all the segments, n at a time")
        report = job.result_txt.read_text(encoding="utf-8")
        self.assertIn("GPU JOBS AT THE SAME TIME", report)
        self.assertIn("3 segments of different kinds of video", report)

    def test_one_gpu_segment_is_too_few_to_try_jobs_at_the_same_time(self) -> None:
        make_video(self.media / "a.mp4", 40)
        real = TrickplayGenerator._run_ffmpeg_extract

        def fake_gpu(self_, meta, frames_dir, method, backend, **kw): return real(self_, meta, frames_dir, "CPU", None, **kw)

        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", fake_gpu), mock.patch.object(DiagnosticRunner, "_measure_keyframes", self.gpu_wins):
            data, _ = self.run_benchmark("quick", ["cuda"])
        self.assertEqual(data["status"], "COMPLETE")
        self.assertIn("too few", data["gpu_parallel"]["error"])
        self.assertNotIn("max_gpu_jobs", [a.key for a in ba.build_advice(data, self.config)])

    def test_a_failing_parallel_test_is_noted_not_fatal(self) -> None:
        for name in ("a.mp4", "b.mp4"): make_video(self.media / name, 40)
        real = TrickplayGenerator._run_ffmpeg_extract

        def fake_gpu(self_, meta, frames_dir, method, backend, **kw):
            if "GPU job" in kw.get("progress_label", ""): raise RuntimeError("driver crashed")
            return real(self_, meta, frames_dir, "CPU", None, **kw)

        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", fake_gpu), mock.patch.object(DiagnosticRunner, "_measure_keyframes", self.gpu_wins):
            data, _ = self.run_benchmark("quick", ["cuda"])
        self.assertEqual(data["status"], "COMPLETE")
        self.assertIn("driver crashed", data["gpu_parallel"]["error"])
        self.assertNotIn("max_gpu_jobs", [a.key for a in ba.build_advice(data, self.config)])

    REFUSED = ("Impossible to convert between the formats supported by the filter 'graph -1 input from stream 0:0' and the filter 'auto_scale_0'\n"
               "Link 'graph -1 input from stream 0:0.default' -> 'auto_scale_0.default':\n  Color spaces:\n    src: bt2020c\n    dst: unknown gbr bt709")

    def refusing(self, calls: list[tuple[str, bool, bool]]):
        """FFmpeg as in the real run: it refuses the colour space tag of this file unless the tag is ignored."""
        real = TrickplayGenerator._run_ffmpeg_extract

        def run(self_, meta, frames_dir, method, backend, **kw):
            ignored = bool(kw.get("ignore_color_tag"))
            calls.append((method, ignored, bool(kw.get("keyframes"))))
            if not ignored:
                return ExtractResult(False, method, backend, 50.0, 0, self.REFUSED)
            return real(self_, meta, frames_dir, "CPU", None, **{k: v for k, v in kw.items() if k != "ignore_color_tag"})
        return run

    def test_a_file_whose_colour_tag_is_refused_is_measured_like_generate_does_it(self) -> None:
        make_video(self.media / "a.mp4", 40)
        calls: list[tuple[str, bool, bool]] = []
        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", self.refusing(calls)):
            data, job = self.run_benchmark("quick", ["cuda"])
        self.assertTrue(data["rows"] and all(r["status"] == "PASS" and r["color_tag_ignored"] for r in data["rows"]), data["rows"])
        self.assertTrue(all(r["status"] == "PASS" for r in data["keyframe_rows"]), data["keyframe_rows"])
        self.assertTrue(any(ignored and keyframes for _, ignored, keyframes in calls), "the key frame run keeps the tag ignored (on whichever path was fastest)")
        self.assertLess(calls.index(("CPU", False, False)), calls.index(("CPU", True, False)), "the first try of every path is unchanged")
        text = job.result_txt.read_text(encoding="utf-8")
        self.assertIn("COLOUR SPACE TAG IGNORED", text)
        self.assertIn("a.mp4", text.split("COLOUR SPACE TAG IGNORED", 1)[1].splitlines()[0])
        self.assertIn("colour space tag ignored", (job.path / "benchmark.log").read_text(encoding="utf-8"))

    def test_the_warm_up_of_a_file_whose_colour_tag_is_refused_also_ignores_the_tag(self) -> None:
        make_video(self.media / "a.mp4", 40)
        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", self.refusing([])):
            data, job = self.run_benchmark("quick", ["cuda"])
        self.assertEqual(data["warmup"]["cuda"]["status"], "PASS")
        self.assertIn("Warm-up GPU/cuda: ok", (job.path / "benchmark.log").read_text(encoding="utf-8"))

    def test_the_time_of_the_refused_first_try_is_not_measured(self) -> None:
        make_video(self.media / "a.mp4", 40)
        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", self.refusing([])):
            data, _ = self.run_benchmark("quick")
        cpu = next(r for r in data["rows"] if r["backend"] == "cpu")
        self.assertLess(cpu["extract_seconds"], 50.0, "the refused try (50 s here) is not part of the result")

    def test_a_real_failure_is_not_run_a_second_time(self) -> None:
        make_video(self.media / "a.mp4", 40)
        calls: list[tuple[str, bool, bool]] = []

        def broken(self_, meta, frames_dir, method, backend, **kw):
            calls.append((method, bool(kw.get("ignore_color_tag")), bool(kw.get("keyframes"))))
            return ExtractResult(False, method, backend, 0.1, 0, "Invalid data found when processing input")

        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", broken):
            data, _ = self.run_benchmark("quick")
        self.assertTrue(all(r["status"] == "FAILED" and not r.get("color_tag_ignored") for r in data["rows"]))
        self.assertFalse(any(c[1] for c in calls), calls)

    def test_a_path_that_fails_for_another_reason_after_the_tag_is_ignored_still_fails(self) -> None:
        make_video(self.media / "a.mp4", 40)
        calls: list[tuple[str, bool, bool]] = []

        def always_refused(self_, meta, frames_dir, method, backend, **kw):
            calls.append((method, bool(kw.get("ignore_color_tag")), bool(kw.get("keyframes"))))
            return ExtractResult(False, method, backend, 0.1, 0, self.REFUSED)

        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", always_refused):
            data, _ = self.run_benchmark("quick")
        self.assertTrue(all(r["status"] == "FAILED" and not r.get("color_tag_ignored") for r in data["rows"]))
        self.assertEqual(calls, [("CPU", False, False), ("CPU", True, False)], "one more try, no more")

    def test_a_file_the_key_frame_run_cannot_handle_is_reported_as_such(self) -> None:
        make_video(self.media / "a.mp4", 40)
        real = TrickplayGenerator._run_ffmpeg_extract

        def refuse(self_, meta, frames_dir, method, backend, **kw):
            if kw.get("keyframes"):
                return real(self_, meta, frames_dir, method, backend, **{**kw, "start_time": -5.0})      # FFmpeg refuses the seek
            return real(self_, meta, frames_dir, method, backend, **kw)

        with mock.patch.object(TrickplayGenerator, "_run_ffmpeg_extract", refuse):
            data, _ = self.run_benchmark()
        row = data["keyframe_rows"][0]
        if row["status"] == "FAILED": self.assertTrue(row["error_summary"])


class BenchmarkScreens(ScreenTest):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.object(type(self.app), "ensure_ffmpeg", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app.tools = mock.Mock()
        self.app.tools.backends.return_value = ["cuda"]
        self.app.tools.candidate.sha256 = "H"
        self.app.tools.gpu_names = ["RTX"]
        self.app.state.open()
        self.library = [video(f"{i}.mkv") for i in range(12)]
        self.results: list[bool] = []

    def fake_runner(self, data):
        runner = mock.Mock()
        runner.run.return_value = data
        return mock.patch("modules.screens.diagnostics.DiagnosticRunner", return_value=runner), runner

    def rows(self):
        return [{"video": f"/v/{i}.mkv", "window": 1, "backend": "cuda", "status": "PASS", "all_seconds": 20.0, "keyframe_seconds": 5.0, "share": 0.0} for i in range(4)]

    def complete(self, **kw):
        return {"kind": "benchmark", "status": "COMPLETE", "rows": [], "backends": ["cuda"], "visual": {}, "keyframe_rows": self.rows(), **kw}

    def test_with_media_folders_the_library_is_the_default_source_and_the_sample_runs_on_copies(self) -> None:
        self.app.config["media_roots"] = [str(self.media)]
        patcher, runner = self.fake_runner(self.complete())
        with patcher, mock.patch.object(type(self.app), "scan_library", return_value=self.library), \
                mock.patch("modules.screens.diagnostics.TrickplayGenerator"):
            out = self.feed(lambda: self.app.run_diagnostic("benchmark"), "", "y", "n")
        self.assertIn("PREFLIGHT", out)
        self.assertRegex(out, r"Files:\s+10 ")
        self.assertIn("deleted after each test", out)
        args, kwargs = runner.run.call_args
        self.assertEqual((args[1], args[3], kwargs["local_copy"]), ("sample", "benchmark", True))
        self.assertEqual(len(args[0]), 10)
        self.assertEqual(len(kwargs["sample"]), 10)
        self.assertIn("SUGGESTED SETTINGS", out)
        self.assertIn("Key frames only", out)
        self.assertFalse(self.app.config["trickplay"]["keyframe_only"], "declined")

    def test_a_test_folder_of_my_own_still_works_and_is_not_copied(self) -> None:
        self.app.config["media_roots"] = [str(self.media)]
        folder = self.root / "mine"
        folder.mkdir()
        patcher, runner = self.fake_runner(self.complete())
        with patcher, mock.patch.object(type(self.app), "_collect_test_videos", return_value=self.library[:2]), \
                mock.patch("modules.screens.diagnostics.TrickplayGenerator"):
            out = self.feed(lambda: self.app.run_diagnostic("benchmark"), "2", str(folder), "1", "y", "n")
        args, kwargs = runner.run.call_args
        self.assertEqual((args[1], kwargs["local_copy"]), ("quick", False))
        self.assertNotIn("Source: your library", out)

    def test_the_library_generate_has_scanned_is_not_read_again(self) -> None:
        job = mock.Mock()
        out = io.StringIO()
        with mock.patch.object(type(self.app), "scan_library") as scan, contextlib.redirect_stdout(out):
            videos = self.app._library_videos("benchmark", job, self.library)
        scan.assert_not_called()
        self.assertEqual(out.getvalue(), "", "no 'Reading the library', no scan line")
        self.assertEqual(len(videos), len(self.library))
        self.assertIs(job.say.call_args.kwargs["echo"], False)

    def test_an_hdr_check_without_an_hdr_video_says_so_and_waits_before_the_menu_returns(self) -> None:
        self.app.config["media_roots"] = [str(self.media)]
        with mock.patch.object(type(self.app), "_library_videos", return_value=self.library):
            out = self.feed(lambda: self.results.append(self.app.run_diagnostic("hdr")), "", "")      # library, then the Enter that keeps the message on the screen (no Quick/Full question: there is nothing to check)
        self.assertEqual(self.results, [False])
        self.assertIn("No HDR candidate detected; test NOT RUN.", out)
        self.assertIn("Choose a folder that has one", out)

    def test_a_failing_diagnostic_shows_the_error_and_waits_before_the_menu_returns(self) -> None:
        self.app.config["media_roots"] = [str(self.media)]
        with mock.patch.object(type(self.app), "_library_videos", side_effect=RuntimeError("boom")):
            out = self.feed(lambda: self.results.append(self.app.run_diagnostic("hdr")), "", "")
        self.assertEqual(self.results, [False])
        self.assertIn("Diagnostic error: RuntimeError: boom", out)

    def test_the_hdr_check_asks_quick_or_full_only_when_it_found_an_hdr_video(self) -> None:
        self.app.config["media_roots"] = [str(self.media)]
        with mock.patch.object(type(self.app), "_library_videos", return_value=self.library), \
                mock.patch("modules.screens.diagnostics.select_hdr_candidates", return_value=self.library[:1]):
            out = self.feed(lambda: self.results.append(self.app.run_diagnostic("hdr")), "", "0")      # library, Back at the Quick/Full question
        self.assertEqual(self.results, [False])
        self.assertIn("1 chosen, one of every kind of HDR:", out)
        self.assertRegex(out, r"\d+ HDR videos? found")

    def test_the_benchmark_before_generate_does_not_check_ffmpeg_again(self) -> None:
        with mock.patch.object(type(self.app), "ensure_ffmpeg") as ensure, mock.patch.object(type(self.app), "_library_videos", return_value=[]):
            self.feed(lambda: self.app.run_diagnostic("benchmark", before_generate=True))
        ensure.assert_not_called()

    def test_before_generate_nothing_is_paused_for(self) -> None:
        with mock.patch.object(type(self.app), "_library_videos", return_value=[]):
            out = self.feed(lambda: self.results.append(self.app.run_diagnostic("benchmark", before_generate=True)))
        self.assertEqual(self.results, [False])
        self.assertIn("No usable videos found; test NOT RUN.", out)

    def test_without_media_folders_only_a_test_folder_is_asked_for(self) -> None:
        self.app.config["media_roots"] = []
        out = self.feed(lambda: self.app.run_diagnostic("benchmark"), "0")
        self.assertIn("Enter a test folder", out)
        self.assertNotIn("Videos from my library", out)

    def test_suggested_settings_are_applied_only_after_a_yes_and_saved(self) -> None:
        data = self.complete(gpu_parallel={"video": "/v/a.mkv", "backend": "cuda", "seconds": 60.0, "runs": [
            {"jobs": 1, "wall": 10.0, "throughput": 6.0}, {"jobs": 2, "wall": 10.0, "throughput": 12.0}, {"jobs": 3, "wall": 10.0, "throughput": 13.0}]})
        out = self.feed(lambda: self.app._offer_benchmark_settings(data), "n")
        self.assertIn("Nothing changed", out)
        self.assertFalse(self.saved()["trickplay"]["keyframe_only"])
        self.assertEqual(self.saved()["acceleration"]["max_gpu_jobs"], 2)
        out = self.feed(lambda: self.app._offer_benchmark_settings(data), "y")
        self.assertIn("Saved.", out)
        self.assertTrue(self.saved()["trickplay"]["keyframe_only"])
        self.assertEqual(self.saved()["acceleration"]["max_gpu_jobs"], 2, "2 jobs are what the measurement says")

    def test_the_saved_choice_is_shown_under_acceleration_and_can_be_forgotten(self) -> None:
        self.app.config["acceleration"]["path_choice"] = pc.merged(None, {"hevc 10-bit 4K SDR": {"best": "cuda", "speedup": 3.5},
                                                                          "h264 8-bit SD SDR": {"best": "cpu", "speedup": 1.0}}, "H", ["RTX"])
        out = self.feed(self.app.configure_acceleration, "5", "y", "0")      # mode, backend, jobs, threshold, then the saved choice
        self.assertIn("hevc 10-bit 4K SDR: cuda (3.50x faster than the CPU)", out)
        self.assertIn("h264 8-bit SD SDR: CPU", out)
        self.assertIsNone(self.app.config["acceleration"]["path_choice"])

    def test_nothing_suggested_says_so(self) -> None:
        out = self.feed(lambda: self.app._offer_benchmark_settings({"keyframe_rows": []}))
        self.assertIn("too few files", out)
        self.app.config["trickplay"]["keyframe_only"] = True
        out = self.feed(lambda: self.app._offer_benchmark_settings(self.complete()))
        self.assertIn("Nothing to change", out)

    def test_the_decode_check_can_use_the_library_too(self) -> None:
        self.app.config["media_roots"] = [str(self.media)]
        with mock.patch.object(type(self.app), "scan_library", return_value=[video("a.mkv", "hevc", 10)]), \
                mock.patch("modules.hardware_decode.check_decode") as check:
            out = self.feed(self.app.run_decode_check, "", "")                  # library (default), then Enter = no
        check.assert_not_called()
        self.assertIn("PREFLIGHT", out)


class BenchmarkBeforeGenerate(BenchmarkScreens):
    """The first Generate for an FFmpeg build and GPU offers the benchmark; the answer is remembered."""

    def plan(self, count: int = 12, action: str = "GENERATE"):
        return [mock.Mock(meta=m, action=action, reason="MISSING") for m in self.library[:count]]

    def offer(self, *answers: str, plan=None, ran: bool = True):
        with mock.patch.object(type(self.app), "run_diagnostic", return_value=ran) as run:
            out = self.feed(lambda: self.results.append(self.app._offer_benchmark_first(plan or self.plan(), mock.Mock())), *answers)
        return out, run

    def setUp(self) -> None:
        super().setUp()
        self.results: list[bool] = []

    def test_the_default_answer_runs_the_benchmark_on_the_library_and_asks_for_a_new_plan(self) -> None:
        out, run = self.offer("")
        self.assertIn("BENCHMARK FIRST", out)
        self.assertRegex(out, r"about \d+ to \d+ minutes")
        run.assert_called_once_with("benchmark", before_generate=True, library=None)
        self.assertEqual(self.results, [True])

    def test_a_benchmark_that_did_not_finish_changes_nothing_for_the_run(self) -> None:
        _, run = self.offer("y", ran=False)
        self.assertEqual(self.results, [False])

    def test_a_no_is_not_remembered_only_a_finished_benchmark_stops_the_question(self) -> None:
        for _ in range(3):
            out, run = self.offer("n")
            self.assertIn("BENCHMARK FIRST", out, "asked again every time")
            run.assert_not_called()
        self.assertEqual(self.results, [False, False, False])
        self.assertIsNone(self.saved()["acceleration"].get("benchmark_seen"))
        out, run = self.offer("y", ran=False)                          # started but not finished: still asked next time
        self.assertIn("BENCHMARK FIRST", self.offer("n")[0])
        self.app.config["acceleration"]["benchmark_seen"] = {"ffmpeg_hash": "H", "gpu_names": ["RTX"]}      # what a finished benchmark leaves
        self.assertEqual(self.offer()[0], "")
        self.app.tools.candidate.sha256 = "OTHER"                    # another FFmpeg build: asked again
        self.assertIn("BENCHMARK FIRST", self.offer("n")[0])

    def test_nothing_is_offered_when_a_choice_is_saved_or_there_are_too_few_videos_to_generate(self) -> None:
        self.app.config["acceleration"]["path_choice"] = pc.merged(None, {"hevc 10-bit 4K SDR": {"best": "cuda", "speedup": 3.5}}, "H", ["RTX"])
        self.assertEqual(self.offer()[0], "")
        self.app.config["acceleration"]["path_choice"] = None
        self.assertEqual(self.offer(plan=self.plan(2))[0], "", "two videos are not worth a benchmark")
        self.assertEqual(self.offer(plan=self.plan(action="SKIP"))[0], "", "nothing to generate")
        self.assertEqual(self.results, [False, False, False])

    def test_a_finished_benchmark_is_not_offered_again_even_when_the_suggestions_were_declined(self) -> None:
        self.app.config["media_roots"] = [str(self.media)]
        patcher, runner = self.fake_runner(self.complete())
        with patcher, mock.patch.object(type(self.app), "scan_library", return_value=self.library), \
                mock.patch("modules.screens.diagnostics.TrickplayGenerator"):
            out = self.feed(lambda: self.results.append(self.app.run_diagnostic("benchmark", before_generate=True)), "n")      # only "Apply these changes?" is asked
        self.assertEqual(self.results, [True])
        self.assertNotIn("Start test?", out)
        self.assertNotIn("PREFLIGHT", out)
        args, kwargs = runner.run.call_args
        self.assertEqual((args[1], kwargs["local_copy"], len(args[0])), ("sample", True, 10))
        self.assertEqual(self.saved()["acceleration"]["benchmark_seen"]["ffmpeg_hash"], "H")
        self.assertEqual(self.offer()[0], "")

    def test_the_plan_is_made_again_with_the_settings_the_benchmark_saved(self) -> None:
        made: list[int] = []

        def generator(*a, **kw):
            made.append(1)
            return mock.Mock(plan=mock.Mock(return_value=self.plan()))

        for answer, expected in ((True, 2), (False, 1)):
            made.clear()
            with mock.patch("modules.screens.generate.TrickplayGenerator", generator), mock.patch.object(type(self.app), "_apply_storage_groups"), \
                    mock.patch.object(type(self.app), "_offer_benchmark_first", return_value=answer), mock.patch.object(type(self.app), "_check_new_kinds_of_video", return_value=True), \
                    mock.patch.object(type(self.app), "_preflight", return_value=(False, {})):
                self.app._plan_and_run(self.library, mock.Mock(), offer_benchmark=True)
            self.assertEqual(len(made), expected)

    def test_only_generate_asks_never_an_unattended_run_a_dry_run_or_a_single_file(self) -> None:
        with mock.patch("modules.screens.generate.TrickplayGenerator") as gen, mock.patch.object(type(self.app), "_apply_storage_groups"), \
                mock.patch.object(type(self.app), "_offer_benchmark_first", return_value=False) as offer, mock.patch.object(type(self.app), "_check_new_kinds_of_video", return_value=True), \
                mock.patch.object(type(self.app), "_preflight", return_value=(False, {})), mock.patch.object(type(self.app), "_preflight_unattended", return_value=(False, {})):
            gen.return_value.plan.return_value = self.plan()
            self.app._plan_and_run(self.library, mock.Mock())                                          # Verify's repair, for one
            self.app._plan_and_run(self.library, mock.Mock(), offer_benchmark=True, interactive=False)
            with contextlib.redirect_stdout(io.StringIO()):                                           # the dry run lists its plan on the screen
                self.app._plan_and_run(self.library, mock.Mock(), offer_benchmark=True, dry_run=True)
            offer.assert_not_called()
            self.app._plan_and_run(self.library, mock.Mock(), offer_benchmark=True)
            offer.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class AlignedKinds(unittest.TestCase):
    def test_the_parts_of_a_kind_line_up_in_columns(self) -> None:
        from modules.library_sample import aligned_kinds
        shown = aligned_kinds(["h264 8-bit SD SDR", "av1 10-bit 1080p SDR", "hevc 10-bit 4K Dolby Vision P8", "msmpeg4v3 8-bit SD SDR"])
        self.assertEqual(shown["h264 8-bit SD SDR"], "h264       8-bit   SD     SDR")
        self.assertEqual(shown["av1 10-bit 1080p SDR"], "av1        10-bit  1080p  SDR")
        self.assertEqual(shown["hevc 10-bit 4K Dolby Vision P8"], "hevc       10-bit  4K     Dolby Vision P8")
        self.assertEqual({text.index(text.split()[1]) for text in shown.values()}, {len("msmpeg4v3") + 2}, "the codec column is as wide as the longest codec")
        self.assertEqual(aligned_kinds(["something else"]), {"something else": "something else"})
