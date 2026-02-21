"""Command-line interface for voice dictation.

Entry point for the `dictate` command. Uses Whisper for real-time
streaming transcription - words appear as you speak.
"""

import argparse
import sys

from dictate.config import SAMPLE_RATE, SOCKET_PATH
from dictate.daemon_support import stop_daemon as _stop_daemon
from dictate.system import check_dependencies_or_exit, notify


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice dictation with real-time streaming transcription",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the dictation daemon",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming - text appears only after recording stops",
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
    """Stop the dictation daemon and clean up."""
    _stop_daemon(
        socket_path=SOCKET_PATH,
        pkill_patterns=["dictate.live.daemon", "dictate-daemon"],
        daemon_name="Dictation",
    )


def main() -> None:
    """Main entry point for dictation."""
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
            notify("Failed to start")
            sys.exit(1)


def _run_dictation(args: argparse.Namespace) -> None:
    """Connect to daemon, record audio, and stream transcription."""
    import threading

    from .client import LiveDaemonClient
    from .recorder import LiveRecorder

    stop_event = threading.Event()

    client = LiveDaemonClient(
        streaming=not args.no_stream,
        stop_event=stop_event,
    )
    client.connect()

    recorder = LiveRecorder(
        client,
        typer=client.typer,
        sample_rate=args.sample_rate,
        stop_event=stop_event,
    )
    client.set_stop_event(stop_event)
    recorder.record()


if __name__ == "__main__":
    main()
