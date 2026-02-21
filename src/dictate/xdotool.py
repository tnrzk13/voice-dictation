"""Xdotool operations for typing text and sending key presses."""

import subprocess

from .config import XDOTOOL_KEYSTROKE_DELAY


def type_text(text: str) -> None:
    """Type text at cursor position using xdotool."""
    subprocess.run(
        ["xdotool", "type", "--delay", str(XDOTOOL_KEYSTROKE_DELAY), text],
        check=False,
    )


def send_backspaces(count: int) -> None:
    """Send backspace key presses via xdotool."""
    subprocess.run(
        ["xdotool", "key", "--delay", "0"] + ["BackSpace"] * count, check=False
    )
