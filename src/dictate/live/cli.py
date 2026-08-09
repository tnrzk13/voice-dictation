"""Command-line interface for voice dictation.

Entry point for the `dictate` command. Uses Whisper for real-time
streaming transcription - words appear as you speak.
"""

import argparse
import socket
import sys

from dictate.config import SAMPLE_RATE, SOCKET_PATH
from dictate.daemon_support import read_daemon_config
from dictate.model_loader import add_model_args, is_model_cached
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
    add_model_args(parser)
    return parser.parse_args()


def stop_daemon() -> None:
    """Stop the dictation daemon and clean up."""
    from dictate.stop import main as stop_main

    stop_main()


def main() -> None:
    """Main entry point for dictation."""
    args = parse_args()

    if args.stop:
        stop_daemon()
        return

    check_dependencies_or_exit()
    _ensure_daemon_running(args)
    _run_dictation(args)


def _ensure_daemon_running(args: argparse.Namespace) -> None:
    """Start the daemon if it's not already running with matching config."""
    from dictate.config import DAEMON_DOWNLOAD_TIMEOUT, DAEMON_STARTUP_TIMEOUT

    from .client import LiveDaemonClient

    if LiveDaemonClient.is_daemon_running() and _daemon_config_matches(args):
        return

    if LiveDaemonClient.is_daemon_running():
        print("Daemon is running with a different model config; restarting it.")
        stop_daemon()

    daemon_args = [
        "--model", args.model,
        "--device", args.device,
        "--compute-type", args.compute_type,
    ]
    if args.quiet:
        daemon_args.append("--quiet")
    needs_download = not is_model_cached(args.model)
    timeout = DAEMON_DOWNLOAD_TIMEOUT if needs_download else DAEMON_STARTUP_TIMEOUT
    if not LiveDaemonClient.start_daemon(
        extra_args=daemon_args, timeout=timeout
    ):
        notify("Failed to start")
        sys.exit(1)


def _daemon_config_matches(args: argparse.Namespace) -> bool:
    """Check whether the running daemon was started with the same model args."""
    config = read_daemon_config(SOCKET_PATH)
    if not config:
        return False
    return (
        config.get("model") == args.model
        and config.get("device") == args.device
        and config.get("compute_type") == args.compute_type
    )


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
    try:
        client.connect()
    except (socket.timeout, OSError) as e:
        print(f"Error: Could not connect to daemon: {e}", file=sys.stderr)
        sys.exit(1)

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
