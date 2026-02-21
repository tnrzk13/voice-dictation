"""Command-line interface for live streaming dictation.

Entry point for the `dictate-live` command. Uses Whisper for real-time
streaming transcription - words appear as you speak.
"""

import argparse
import sys

from dictate.daemon_support import stop_daemon as _stop_daemon
from dictate.system import check_dependencies_or_exit, notify

from .config import LIVE_SOCKET_PATH

from dictate.config import SAMPLE_RATE


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Live streaming voice dictation with real-time transcription",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the live dictation daemon",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        metavar="RATE",
        help=f"Audio sample rate in Hz (default: {SAMPLE_RATE})",
    )
    return parser.parse_args()


def stop_daemon() -> None:
    """Stop the live dictation daemon and clean up."""
    _stop_daemon(
        socket_path=LIVE_SOCKET_PATH,
        pkill_patterns=["dictate.live.daemon", "dictate-live-daemon"],
        daemon_name="Live Dictation",
    )


def main() -> None:
    """Main entry point for live dictation."""
    args = parse_args()

    if args.stop:
        stop_daemon()
        return

    check_dependencies_or_exit()
    _ensure_daemon_running()
    _run_dictation(args)


def _ensure_daemon_running() -> None:
    """Start the daemon if it's not already running."""
    from .client import LiveDaemonClient

    if not LiveDaemonClient.is_daemon_running():
        if not LiveDaemonClient.start_daemon():
            notify("Live Dictation Error", "Failed to start daemon")
            sys.exit(1)


def _run_dictation(args: argparse.Namespace) -> None:
    """Connect to daemon, record audio, and stream transcription."""
    from .client import LiveDaemonClient
    from .recorder import LiveRecorder

    notify("Live Dictation", "Recording started...")

    client = LiveDaemonClient()
    client.connect()

    recorder = LiveRecorder(
        client,
        sample_rate=args.sample_rate,
    )
    client.set_stop_event(recorder.stop_event)
    recorder.record()


if __name__ == "__main__":
    main()
