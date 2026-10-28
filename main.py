# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import sys

from modules.application import run_application


def main() -> int:
    if sys.platform == "win32":
        from modules.platform_windows import WindowsPlatform
        return run_application(WindowsPlatform(), platform_name=sys.platform, argv=sys.argv[1:])

    if sys.platform == "linux":
        from modules.platform_linux import LinuxPlatform
        return run_application(LinuxPlatform(), platform_name=sys.platform, argv=sys.argv[1:])

    # Platform selection belongs only in this entry point. Additional adapters
    # can be added without changing application/core modules.
    raise RuntimeError(f"Unsupported platform: {sys.platform}")


if __name__ == "__main__":
    raise SystemExit(main())
