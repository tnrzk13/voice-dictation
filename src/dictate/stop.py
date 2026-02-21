"""Stop the dictation daemon."""

from .config import SOCKET_PATH
from .daemon_support import stop_daemon


def main() -> None:
    """Stop the dictation daemon if running."""
    stop_daemon(
        socket_path=SOCKET_PATH,
        pkill_patterns=["dictate.live.daemon", "dictate-daemon"],
        daemon_name="Dictation",
    )
