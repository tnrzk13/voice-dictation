"""Utility functions for voice dictation."""

import shutil
import subprocess
from typing import List, Tuple

from .config import XDOTOOL_KEYSTROKE_DELAY


def check_system_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if required system dependencies are installed.

    Returns:
        Tuple of (all_present, missing_dependencies)
    """
    required_tools = {
        "xdotool": "Install with: sudo apt-get install xdotool",
        "notify-send": "Install with: sudo apt-get install libnotify-bin"
    }

    missing = []
    for tool, install_msg in required_tools.items():
        if not shutil.which(tool):
            missing.append(f"{tool} - {install_msg}")

    return len(missing) == 0, missing


def notify(title: str, message: str) -> None:
    """Show desktop notification."""
    subprocess.run(["notify-send", title, message], check=False)


def type_text(text: str) -> None:
    """Type text at cursor position using xdotool."""
    subprocess.run(
        ["xdotool", "type", "--delay", str(XDOTOOL_KEYSTROKE_DELAY), text],
        check=False
    )
