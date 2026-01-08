#!/usr/bin/env python3
"""
Command-line interface for voice dictation.

Dictation client - records audio with pause detection and sends to daemon for transcription.
Supports streaming transcription for long recording sessions.
"""
import sys
import time
import argparse

from .client import DaemonClient
from .recorder import StreamingAudioRecorder
from .utils import notify


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice dictation with automatic pause detection",
        epilog="""
Examples:
  %(prog)s                  # Default streaming mode
  %(prog)s --stream         # Explicit streaming mode
  %(prog)s --no-stream      # Non-streaming mode

How it works:
  1. Run the script (starts recording after 2-second delay)
  2. Speak naturally into your microphone
  3. Pause for 4 seconds to stop recording
  4. Text is typed at your cursor position

Streaming vs Non-Streaming:
  - Streaming (default): Transcribes and types text every 5 seconds while you speak.
    Great for long dictation sessions.

  - Non-streaming (--no-stream): Records everything, then transcribes all at once.
    Good for short commands or when you want all text to appear together.
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
    return parser.parse_args()


def main() -> None:
    """Main entry point for dictation client."""
    args = parse_args()

    # Determine streaming mode (default is True)
    streaming = not args.no_stream

    # Give user time to switch windows (2 second delay)
    time.sleep(2)

    # Ensure daemon is running
    if not DaemonClient.is_running():
        if not DaemonClient.start():
            notify("Dictation Error", "Failed to start daemon")
            sys.exit(1)

    # Record and transcribe
    client = DaemonClient()
    recorder = StreamingAudioRecorder(client, streaming=streaming)
    recorder.record()


if __name__ == "__main__":
    main()
