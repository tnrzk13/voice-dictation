#!/usr/bin/env python3
"""
Command-line interface for voice dictation.

Dictation client - records audio with pause detection and sends to daemon for transcription.
Supports streaming transcription for long recording sessions.
"""
import sys
import time
import argparse

from dictate.daemon_support import stop_daemon as _stop_daemon
from dictate.system import notify, check_dependencies_or_exit
from dictate.config import (
    SOCKET_PATH,
    PRE_RECORDING_DELAY,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    CHUNK_DURATION,
    SAMPLE_RATE
)

from .client import DaemonClient
from .recorder import StreamingAudioRecorder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice dictation with automatic pause detection",
        epilog="""
Examples:
  %(prog)s                                    # Default streaming mode
  %(prog)s --stream                           # Explicit streaming mode
  %(prog)s --no-stream                        # Non-streaming mode
  %(prog)s --stop                             # Stop the daemon
  %(prog)s --silence-threshold 0.02           # Less sensitive (louder sounds required)
  %(prog)s --silence-duration 6               # Longer pause before stopping
  %(prog)s --chunk-duration 3                 # More responsive streaming
  %(prog)s --silence-threshold 0.005 --silence-duration 2  # Quick, sensitive mode

How it works:
  1. Run the script (starts recording after 2-second delay)
  2. Speak naturally into your microphone
  3. Pause for 4 seconds (default) to stop recording
  4. Text is typed at your cursor position

Streaming vs Non-Streaming:
  - Streaming (default): Transcribes and types text every 5 seconds while you speak.
    Great for long dictation sessions.

  - Non-streaming (--no-stream): Records everything, then transcribes all at once.
    Good for short commands or when you want all text to appear together.

Customization Tips:
  - If it stops too early: increase --silence-duration or decrease --silence-threshold
  - If it doesn't detect pauses: increase --silence-threshold
  - For faster feedback: decrease --chunk-duration (streaming mode)
        """
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
        "--silence-threshold",
        type=float,
        default=SILENCE_THRESHOLD,
        metavar="THRESHOLD",
        help=f"RMS energy threshold for silence detection (default: {SILENCE_THRESHOLD}). "
             "Lower values = more sensitive to sound. Range: 0.001-0.1"
    )
    parser.add_argument(
        "--silence-duration",
        type=int,
        default=SILENCE_DURATION,
        metavar="SECONDS",
        help=f"Seconds of continuous silence before stopping recording (default: {SILENCE_DURATION})"
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
        help=f"Audio sample rate in Hz (default: {SAMPLE_RATE}). Common values: 8000, 16000, 44100"
    )
    return parser.parse_args()


def stop_daemon() -> None:
    """Stop the dictation daemon process and clean up socket."""
    _stop_daemon(
        socket_path=SOCKET_PATH,
        pkill_patterns=["dictate.batch.daemon", "dictate-daemon"],
        daemon_name="Dictation",
    )


def main() -> None:
    """Main entry point for dictation client."""
    args = parse_args()

    # Handle --stop flag
    if args.stop:
        stop_daemon()
        return

    check_dependencies_or_exit()

    # Determine streaming mode (default is True)
    streaming = not args.no_stream

    # Give user time to switch windows
    time.sleep(PRE_RECORDING_DELAY)

    # Ensure daemon is running
    if not DaemonClient.is_running():
        if not DaemonClient.start():
            notify("Dictation Error", "Failed to start daemon")
            sys.exit(1)

    # Record and transcribe
    client = DaemonClient()
    recorder = StreamingAudioRecorder(
        client,
        chunk_duration=args.chunk_duration,
        streaming=streaming,
        sample_rate=args.sample_rate,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration
    )
    recorder.record()


if __name__ == "__main__":
    main()
