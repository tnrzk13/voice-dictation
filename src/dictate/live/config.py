"""Configuration constants for live streaming dictation."""

import os

# Whisper Model Configuration
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Transcription Timing
TRANSCRIBE_INTERVAL = 2  # seconds between transcription cycles
MAX_WINDOW_SECONDS = 20  # finalize segments when audio exceeds this length

# Daemon Configuration
LIVE_SOCKET_PATH = "/tmp/dictate-live-daemon.sock"
LIVE_DAEMON_LOG = os.path.expanduser("~/.local/share/voice-dictation/live-daemon.log")
LIVE_DAEMON_STARTUP_TIMEOUT = 10  # seconds - Whisper model load can take a moment
LIVE_DAEMON_POLL_INTERVAL = 0.1

__all__ = [
    "WHISPER_MODEL_SIZE",
    "WHISPER_DEVICE",
    "WHISPER_COMPUTE_TYPE",
    "TRANSCRIBE_INTERVAL",
    "MAX_WINDOW_SECONDS",
    "LIVE_SOCKET_PATH",
    "LIVE_DAEMON_LOG",
    "LIVE_DAEMON_STARTUP_TIMEOUT",
    "LIVE_DAEMON_POLL_INTERVAL",
]
