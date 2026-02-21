"""System utilities - dependency checks and desktop notifications."""

import shutil
import subprocess
import sys
from typing import List, Tuple


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


def check_dependencies_or_exit() -> None:
    """Check system dependencies and exit with actionable error if any are missing."""
    deps_ok, missing = check_system_dependencies()
    if not deps_ok:
        print("Error: Missing required system dependencies:", file=sys.stderr)
        for dep in missing:
            print(f"  - {dep}", file=sys.stderr)
        sys.exit(1)


def notify(title: str, message: str) -> None:
    """Show desktop notification."""
    subprocess.run(["notify-send", title, message], check=False)
