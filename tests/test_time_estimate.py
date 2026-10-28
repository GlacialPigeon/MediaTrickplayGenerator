# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""The time Generate shows in the preflight."""
from __future__ import annotations

import unittest
from pathlib import Path

from modules import path_choice, time_estimate
from modules.models import VideoMeta


def meta(height: int = 1080, duration: float = 3600.0) -> VideoMeta:
    m = VideoMeta.__new__(VideoMeta)
    m.height, m.duration, m.path = height, duration, Path("a.mkv")
    return m


class Estimate(unittest.TestCase):
    def test_measured_speed_gives_a_narrow_range(self) -> None:
        fast, slow, measured = time_estimate.estimate([meta()] * 10, lambda m: 40.0, 1, keyframes=False)
        self.assertTrue(measured)
        self.assertAlmostEqual(fast, 10 * (3600 / 40 + 1))
        self.assertAlmostEqual(slow, fast * time_estimate.SLOW_MARGIN)

    def test_key_frames_and_a_second_job_lower_the_fast_end_only(self) -> None:
        plain = time_estimate.estimate([meta()], lambda m: 40.0, 1, keyframes=False)
        quick = time_estimate.estimate([meta()], lambda m: 40.0, 4, keyframes=True)
        self.assertAlmostEqual(quick[0], plain[0] / 4)
        self.assertAlmostEqual(quick[1], plain[1])

    def test_a_slow_disk_sets_the_least_time_and_an_ssd_never_does(self) -> None:
        gib = 1024 ** 3
        hdd = time_estimate.disk_seconds("hdd", 66 * gib)
        self.assertAlmostEqual(hdd, 66 * 1024 / (time_estimate.DEFAULT_HDD_MB_PER_SECOND * time_estimate.HDD_EFFECTIVE_SHARE))
        self.assertAlmostEqual(time_estimate.disk_seconds("hdd", 66 * gib, 55.0), 66 * 1024 / (55.0 * time_estimate.HDD_EFFECTIVE_SHARE), msg="the measured speed wins")
        self.assertEqual(time_estimate.disk_seconds("ssd", 66 * gib), 0.0)
        self.assertEqual(time_estimate.disk_seconds("hdd", 0), 0.0)
        self.assertAlmostEqual(time_estimate.disk_seconds("hdd", 10 * gib, 0), time_estimate.disk_seconds("hdd", 10 * gib), msg="a nonsense speed is ignored")

    def test_the_disk_limits_the_estimate_only_when_it_is_the_slower_part(self) -> None:
        compute = time_estimate.estimate([meta()] * 10, lambda m: 40.0, 2, keyframes=False)
        slow_disk = time_estimate.estimate([meta()] * 10, lambda m: 40.0, 2, keyframes=False, disk_floor=compute[0] * 3)
        self.assertAlmostEqual(slow_disk[0], compute[0] * 3)
        self.assertAlmostEqual(slow_disk[1], compute[0] * 3 * time_estimate.DISK_MARGIN, msg="the upper end is not below the lower one")
        self.assertGreaterEqual(slow_disk[1], slow_disk[0])
        quick_disk = time_estimate.estimate([meta()] * 10, lambda m: 40.0, 2, keyframes=False, disk_floor=compute[0] / 4)
        self.assertEqual(quick_disk, compute)

    def test_unmeasured_kinds_use_typical_speeds_and_say_so(self) -> None:
        fast, slow, measured = time_estimate.estimate([meta(2160)], lambda m: None, 1, keyframes=False)
        self.assertFalse(measured)
        self.assertGreater(fast, 0)

    def test_the_text(self) -> None:
        self.assertEqual(time_estimate.describe(3 * 3600, 4 * 3600, True), "about 3 h 30 min")
        self.assertEqual(time_estimate.describe(2 * 3600, 6 * 3600, False), "about 2 to 6 hours")
        self.assertEqual(time_estimate.describe(600, 1500, True), "about 10 to 25 minutes")
        self.assertEqual(time_estimate.describe(10, 20, True), "less than a minute")


class Rate(unittest.TestCase):
    def test_the_benchmark_rate_is_video_seconds_per_second_of_the_best_path(self) -> None:
        rows = []
        for backend, seconds in (("cpu", 6.0), ("cuda", 2.0)):
            rows.append({"video": "v", "kind": "hevc 10-bit 1080p SDR", "backend": backend, "status": "PASS", "extract_seconds": seconds, "pack_seconds": 0.0,
                         "matching_frame_count": True, "duration_seconds": 60.0})
        result = path_choice.from_benchmark({"rows": rows, "backends": ["cuda"]}, 1.15)
        self.assertEqual(result["hevc 10-bit 1080p SDR"]["best"], "cuda")
        self.assertEqual(result["hevc 10-bit 1080p SDR"]["rate"], 30.0)


if __name__ == "__main__":
    unittest.main()
