# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

"""Static checks, run when the tools are installed (they are optional on user machines)."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import unittest
from typing import Optional

from tests.helpers import ROOT


def _tool(name: str) -> Optional[list[str]]:
    """Command prefix for a tool installed either as a Python module or as a standalone program."""
    if importlib.util.find_spec(name) is not None:
        return [sys.executable, "-m", name]
    found = shutil.which(name)
    return [found] if found else None


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)


class StaticChecks(unittest.TestCase):
    def test_mypy_is_clean(self) -> None:
        mypy = _tool("mypy")
        if mypy is None: self.skipTest("mypy is not installed")
        proc = _run([*mypy, "--ignore-missing-imports", "--no-incremental", "modules", "main.py"])
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:])

    def test_ruff_finds_no_real_errors(self) -> None:
        ruff = _tool("ruff")
        if ruff is None: self.skipTest("ruff is not installed")
        proc = _run([*ruff, "check", "--no-cache", "--select", "F,E9,E702,E703,W291,W293", "modules", "main.py", "tests"])
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:])


if __name__ == "__main__":
    unittest.main()
