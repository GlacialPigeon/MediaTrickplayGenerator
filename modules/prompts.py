# Media Trickplay Generator
# Author: GlacialPigeon
# GitHub: https://github.com/GlacialPigeon

from __future__ import annotations

import re
import shutil
import textwrap
from typing import Iterable, Optional, Sequence

from .localization import t


# --- Standard input (no msvcrt, no sys.__stdout__) ---
def prompt_choice(prompt: str, valid: Iterable[str], default: Optional[str] = None) -> str:
    valid_set = {str(v).lower() for v in valid}
    while True:
        suffix = f" [{default}]" if default is not None else ""
        answer = input(f"{prompt}{suffix}: ").strip().lower()
        if not answer and default is not None:
            return str(default).lower()
        if answer in valid_set:
            return answer
        print(t("prompt.choose_one_of", options=", ".join(sorted(valid_set))))

def heading(title: str) -> None:
    """The title of a part of a screen (PREFLIGHT, FINAL SUMMARY ...): an empty line, the title, a line under it."""
    print(f"\n{title}\n{'-' * min(len(title), 72)}")


def label_width(labels: Iterable[str], minimum: int = 10) -> int:
    """How wide the column of labels must be: the longest label, its colon and a space (and never narrower than `minimum`)."""
    return max(minimum, max((len(label) for label in labels), default=0) + 2)


def fit(text: str, hanging: int = 2) -> str:
    """One line of text for the screen: if it is longer than the terminal it is wrapped, and what continues is indented under it
    (`hanging` more than the line itself; a word that is longer than the room, such as a path, stays whole)."""
    room = shutil.get_terminal_size((80, 24)).columns - 1
    if room < 30 or len(text) <= room: return text
    lead = len(text) - len(text.lstrip(" "))
    return "\n".join(textwrap.wrap(text, room, subsequent_indent=" " * (lead + hanging), break_long_words=False, break_on_hyphens=False)) or text


def format_labelled(rows: Sequence[tuple[str, str]], indent: int = 2, minimum: int = 10, width: Optional[int] = None, wrap: bool = True) -> list[str]:
    """Rows of "Label: value" with the values in one column. The labels are translated and so of any length: the column is as wide as
    the longest of them needs (and never narrower than `minimum`; `width` from label_width() when several calls must line up).
    A value of several lines continues under itself, and a line that is longer than the terminal is wrapped under itself
    (a word that is longer than the room, such as a path, stays whole)."""
    width = width or label_width((label for label, _ in rows), minimum)
    room = shutil.get_terminal_size((80, 24)).columns - 1 - indent - width if wrap else 0
    out: list[str] = []
    for label, value in rows:
        lines: list[str] = []
        for line in value.split("\n"):
            lines.extend((textwrap.wrap(line, room, break_long_words=False, break_on_hyphens=False) or [""]) if room >= 20 else [line])
        out.append(" " * indent + f"{label}:".ljust(width) + lines[0])
        out.extend(" " * (indent + width) + line for line in lines[1:])
    return out


def labelled(rows: Sequence[tuple[str, str]], indent: int = 2, minimum: int = 10, width: Optional[int] = None) -> None:
    for line in format_labelled(rows, indent, minimum, width): print(line)


def pause() -> None:
    """Wait for Enter, so what a screen showed stays readable before the next screen clears it."""
    input("\n" + t("prompt.enter_to_continue"))


def choose(entries: Sequence[tuple[str, str]], default: Optional[str] = None, prompt: Optional[str] = None) -> str:
    """Show a vertical menu of (key, label) entries and return the chosen key. Every menu looks and answers the same way."""
    print()
    for key, label in entries:
        print(f"[{key}] {label}")
    print()
    return prompt_choice(prompt or t("prompt.choice"), [key for key, _ in entries], default)


def read_main_menu_choice(allowed: Iterable[int], prompt: Optional[str] = None, default: Optional[int] = None) -> int:
    """Menu input. An empty answer returns `default`; with no default it returns -1 (the caller just redraws).
    Anything invalid is rejected and asked again."""
    allowed_set = set(allowed)
    while True:
        value = input(prompt or t("prompt.choice") + ": ").strip()
        if not value:
            return -1 if default is None else default
        try:
            number = int(value)
            if number in allowed_set:
                return number
        except ValueError:
            pass
        print("\n" + t("prompt.invalid_selection") + "\n")

def _answers(language_words: str, english: str) -> set[str]:
    """The words a language accepts for yes or no (its own, from the language file) and always the English ones."""
    return {word.strip().lower() for word in (language_words + "," + english).split(",") if word.strip()}


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = t("prompt.yes_default") if default else t("prompt.no_default")
    yes, no = _answers(t("prompt.yes_words"), "y,yes"), _answers(t("prompt.no_words"), "n,no")
    while True:
        answer = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not answer: return default
        if answer in yes: return True
        if answer in no: return False
        print(t("prompt.answer_yes_or_no"))

def parse_number_selection(raw: str, maximum: int) -> set[int]:
    selected: set[int] = set()
    for part in raw.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not match:
            continue
        first = int(match.group(1))
        last = int(match.group(2) or first)
        selected.update(range(max(1, first), min(maximum, last) + 1))
    return selected

def prompt_int(prompt: str, default: int, minimum: int, maximum: int, warn_above: Optional[int] = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw: return default
        try:
            value = int(raw)
        except ValueError:
            print(t("prompt.whole_number"))
            continue
        if not (minimum <= value <= maximum):
            print(t("prompt.allowed_range", minimum=minimum, maximum=maximum))
            continue
        if warn_above is not None and value > warn_above:
            if not prompt_yes_no(t("prompt.unusually_high", value=value), False): continue
        return value

def prompt_float(prompt: str, default: float, minimum: float, maximum: float) -> float:
    while True:
        raw = input(f"{prompt} [{default:g}]: ").strip()
        if not raw: return default
        try:
            value = float(raw)
        except ValueError:
            print(t("prompt.number"))
            continue
        if not (minimum <= value <= maximum):
            print(t("prompt.allowed_range", minimum=f"{minimum:g}", maximum=f"{maximum:g}"))
            continue
        return value
