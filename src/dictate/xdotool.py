"""Xdotool operations for typing text and sending key presses."""

import subprocess

from .config import XDOTOOL_KEYSTROKE_DELAY


def type_text(text: str) -> None:
    """Type text at cursor position using xdotool.

    Splits on newlines and tabs, typing text segments normally and sending
    special characters as key presses for reliable X11 behavior.
    """
    _SPECIAL_KEYS = {"\n": "Return", "\t": "Tab"}
    parts = _split_special_chars(text, _SPECIAL_KEYS)
    for part in parts:
        if part in _SPECIAL_KEYS:
            subprocess.run(["xdotool", "key", _SPECIAL_KEYS[part]], check=False)
        elif part:
            subprocess.run(
                ["xdotool", "type", "--delay", str(XDOTOOL_KEYSTROKE_DELAY), part],
                check=False,
            )


def _split_special_chars(text: str, special: dict) -> list:
    """Split text into segments of regular text and special characters."""
    parts = []
    current = []
    for ch in text:
        if ch in special:
            if current:
                parts.append("".join(current))
                current = []
            parts.append(ch)
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def send_backspaces(count: int) -> None:
    """Send backspace key presses via xdotool."""
    subprocess.run(
        ["xdotool", "key", "--delay", "0"] + ["BackSpace"] * count, check=False
    )
