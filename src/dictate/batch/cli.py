#!/usr/bin/env python3
"""Command-line interface for batch voice dictation."""

import sys
import time
import argparse

from dictate.daemon_support import stop_daemon as _stop_daemon
from dictate.system import notify, check_dependencies_or_exit
from dictate.config import (
    SOCKET_PATH,
    PRE_RECORDING_DELAY,
    CHUNK_DURATION,
    SAMPLE_RATE,
)

from .client import DaemonClient
from .recorder import StreamingAudioRecorder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice dictation - press Enter to stop recording",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming transcription (transcribe all at once after recording)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming transcription (transcribe in chunks while recording - default)"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the dictation daemon and free up memory"
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=CHUNK_DURATION,
        metavar="SECONDS",
        help=f"Seconds between transcriptions in streaming mode (default: {CHUNK_DURATION})"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        metavar="RATE",
        help=f"Audio sample rate in Hz (default: {SAMPLE_RATE})"
    )
    return parser.parse_args()


def stop_daemon() -> None:
    """Stop the dictation daemon process and clean up socket."""
    _stop_daemon(
        socket_path=SOCKET_PATH,
        pkill_patterns=["dictate.batch.daemon", "dictate-daemon"],
        daemon_name="Dictation",
    )


def _ensure_daemon_running() -> None:
    """Start the daemon if it's not already running, or exit on failure."""
    if DaemonClient.is_running():
        return
    if not DaemonClient.start():
        notify("Dictation Error", "Failed to start daemon")
        sys.exit(1)


def _run_dictation(args: argparse.Namespace) -> None:
    """Record audio and transcribe via daemon."""
    client = DaemonClient()
    recorder = StreamingAudioRecorder(
        client,
        chunk_duration=args.chunk_duration,
        streaming=not args.no_stream,
        sample_rate=args.sample_rate,
    )
    recorder.record()


def main() -> None:
    """Main entry point for dictation client."""
    args = parse_args()

    if args.stop:
        stop_daemon()
        return

    check_dependencies_or_exit()
    time.sleep(PRE_RECORDING_DELAY)
    _ensure_daemon_running()
    _run_dictation(args)


if __name__ == "__main__":
    main()
