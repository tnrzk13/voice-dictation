"""Utility functions for voice dictation."""

import subprocess


def notify(title: str, message: str) -> None:
    """Show desktop notification."""
    subprocess.run(["notify-send", title, message], check=False)


def type_text(text: str) -> None:
    """Type text at cursor position using xdotool."""
    subprocess.run(["xdotool", "type", "--delay", "5", text], check=False)
