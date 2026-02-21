"""Stop all dictation daemons (batch and live)."""

from .config import SOCKET_PATH
from .daemon_support import stop_daemon
from .live.config import LIVE_SOCKET_PATH


def main() -> None:
    """Stop both batch and live daemons if running."""
    stop_daemon(
        socket_path=SOCKET_PATH,
        pkill_patterns=["dictate.batch.daemon", "dictate-daemon"],
        daemon_name="Dictation",
    )
    stop_daemon(
        socket_path=LIVE_SOCKET_PATH,
        pkill_patterns=["dictate.live.daemon", "dictate-live-daemon", "dictate-live"],
        daemon_name="Live Dictation",
    )
