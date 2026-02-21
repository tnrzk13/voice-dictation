"""Command-line interface for live streaming dictation.

Entry point for the `dictate-live` command. Uses Vosk for real-time
streaming transcription - words appear as you speak.
"""

import argparse
import os
import subprocess
import sys

from dictate.utils import check_system_dependencies, notify

from .config import (
    LIVE_SOCKET_PATH,
    SAMPLE_RATE,
    SILENCE_DURATION,
    SILENCE_THRESHOLD,
)


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
        "--silence-threshold",
        type=float,
        default=SILENCE_THRESHOLD,
        metavar="THRESHOLD",
        help=f"RMS energy threshold for silence detection (default: {SILENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--silence-duration",
        type=int,
        default=SILENCE_DURATION,
        metavar="SECONDS",
        help=f"Seconds of silence before stopping (default: {SILENCE_DURATION})",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        metavar="RATE",
        help=f"Audio sample rate in Hz (default: {SAMPLE_RATE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help="Vosk model name (default: vosk-model-en-us-0.22)",
    )
    return parser.parse_args()


def stop_daemon() -> None:
    """Stop the live dictation daemon and clean up."""
    daemon_stopped = False

    for pattern in ["dictate.live.daemon", "dictate-live-daemon"]:
        try:
            result = subprocess.run(
                ["pkill", "-f", pattern],
                check=False,
                capture_output=True,
            )
            if result.returncode == 0:
                daemon_stopped = True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    socket_existed = os.path.exists(LIVE_SOCKET_PATH)
    try:
        if socket_existed:
            os.remove(LIVE_SOCKET_PATH)
    except (OSError, PermissionError) as e:
        print(f"Warning: Could not remove socket file: {e}")

    notify("Live Dictation Daemon", "Stopped - memory freed")

    if daemon_stopped or socket_existed:
        print("Live dictation daemon stopped")
    else:
        print("No live daemon was running")


def main() -> None:
    """Main entry point for live dictation."""
    args = parse_args()

    if args.stop:
        stop_daemon()
        return

    _check_dependencies()
    _ensure_daemon_running()
    _run_dictation(args)


def _check_dependencies() -> None:
    """Verify system dependencies are available."""
    deps_ok, missing = check_system_dependencies()
    if not deps_ok:
        print("Error: Missing required system dependencies:", file=sys.stderr)
        for dep in missing:
            print(f"  - {dep}", file=sys.stderr)
        sys.exit(1)


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
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
    )
    recorder.record()


if __name__ == "__main__":
    main()
