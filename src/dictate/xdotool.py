"""Xdotool operations for typing text and sending key presses."""

import logging
import subprocess

from .config import XDOTOOL_KEYSTROKE_DELAY

logger = logging.getLogger(__name__)

_SPECIAL_KEYS = {"\n": "Return", "\t": "Tab"}


def type_text(text: str) -> None:
    """Type text at cursor position using xdotool.

    Splits on newlines and tabs, typing text segments normally and sending
    special characters as key presses for reliable X11 behavior.
    """
    parts = _split_special_chars(text, _SPECIAL_KEYS)
    for part in parts:
        if part in _SPECIAL_KEYS:
            _run_xdotool(["key", _SPECIAL_KEYS[part]])
        elif part:
            _run_xdotool(
                ["type", "--delay", str(XDOTOOL_KEYSTROKE_DELAY), part]
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
    _run_xdotool(["key", "--delay", "0"] + ["BackSpace"] * count)


def _run_xdotool(args: list) -> None:
    """Run an xdotool command and log a warning if it fails."""
    result = subprocess.run(["xdotool"] + args, check=False, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        logger.warning(f"xdotool {' '.join(args)} failed: {stderr}")
