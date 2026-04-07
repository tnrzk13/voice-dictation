"""Command-line interface for voice dictation.

Entry point for the `dictate` command. Uses Whisper for real-time
streaming transcription - words appear as you speak.
"""

import argparse
import sys

from dictate.config import (
    SAMPLE_RATE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)
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
    parser.add_argument(
        "--model",
        default=WHISPER_MODEL_SIZE,
        help=f"Whisper model size (default: {WHISPER_MODEL_SIZE})",
    )
    parser.add_argument(
        "--device",
        default=WHISPER_DEVICE,
        help=f"Compute device: cuda or cpu (default: {WHISPER_DEVICE})",
    )
    parser.add_argument(
        "--compute-type",
        default=WHISPER_COMPUTE_TYPE,
        help=f"Model precision: float16, int8, etc. (default: {WHISPER_COMPUTE_TYPE})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress download progress notifications",
    )
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
    """Start the daemon if it's not already running."""
    from dictate.config import DAEMON_DOWNLOAD_TIMEOUT, DAEMON_STARTUP_TIMEOUT

    from .client import LiveDaemonClient

    if not LiveDaemonClient.is_daemon_running():
        daemon_args = [
            "--model", args.model,
            "--device", args.device,
            "--compute-type", args.compute_type,
        ]
        if args.quiet:
            daemon_args.append("--quiet")
        needs_download = not _is_model_cached(args.model)
        timeout = DAEMON_DOWNLOAD_TIMEOUT if needs_download else DAEMON_STARTUP_TIMEOUT
        if not LiveDaemonClient.start_daemon(
            extra_args=daemon_args, timeout=timeout
        ):
            notify("Failed to start")
            sys.exit(1)


def _is_model_cached(model_size: str) -> bool:
    """Check if the Whisper model is already downloaded."""
    try:
        from faster_whisper.utils import download_model

        download_model(model_size, local_files_only=True)
        return True
    except Exception:
        return False


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
