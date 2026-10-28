# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence

from .constants import APP_NAME, APP_VERSION

# Exit codes of the command-line mode (the interactive menu always exits with 0 unless it crashes or is canceled).
EXIT_OK = 0
EXIT_PROBLEMS = 1      # videos not valid / some failed / a self-test check failed
EXIT_NOT_RUN = 2       # could not start: no FFmpeg, no media roots, broken config, another instance running, bad arguments
EXIT_CRASH = 3         # unexpected error (a crash log is written)
EXIT_CANCELED = 130    # Ctrl+C


@dataclass(frozen=True)
class CliArgs:
    command: Optional[str] = None   # None = interactive menu; otherwise "generate", "verify", "self-test" or "check-language"
    dry_run: bool = False
    language: str = ""              # with "check-language": the language to check

    @property
    def interactive(self) -> bool:
        return self.command is None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=f"{APP_NAME} {APP_VERSION}. Without arguments the interactive menu opens. "
                    "With an action the program runs once without any questions and exits (for schedulers).",
        epilog="Exit codes: 0 = OK, 1 = problems found (not valid / failed / check failed), "
               "2 = could not run (no FFmpeg, no media roots, broken configuration, another instance running), "
               "3 = unexpected error, 130 = canceled.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--generate", action="store_true", help="generate missing, damaged and outdated trickplay for the configured media roots")
    action.add_argument("--verify", action="store_true", help="check existing trickplay output and report problems (changes nothing)")
    action.add_argument("--self-test", action="store_true", help="run the self-test")
    action.add_argument("--check-language", metavar="CODE", help="show what is missing or wrong in a language file of the locales folder (for example sl-SI)")
    parser.add_argument("--dry-run", action="store_true", help="with --generate: only show what would be done")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")
    return parser


def parse_args(argv: Sequence[str]) -> CliArgs:
    """Parse the command line. Invalid combinations end the program with exit code 2 (argparse behavior)."""
    parser = build_parser()
    ns = parser.parse_args(list(argv))
    if ns.dry_run and not ns.generate:
        parser.error("--dry-run can only be used together with --generate")
    command = "generate" if ns.generate else "verify" if ns.verify else "self-test" if ns.self_test else "check-language" if ns.check_language else None
    return CliArgs(command=command, dry_run=bool(ns.dry_run), language=str(ns.check_language or ""))
