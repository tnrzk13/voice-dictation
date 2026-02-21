"""Configuration constants for live streaming dictation."""

import os

from dictate.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE  # noqa: F401

# Transcription Timing
TRANSCRIBE_INTERVAL = 2  # seconds between transcription cycles
MAX_WINDOW_SECONDS = 20  # finalize segments when audio exceeds this length

# Daemon
LIVE_SOCKET_PATH = "/tmp/dictate-live-daemon.sock"
LIVE_DAEMON_LOG = os.path.expanduser("~/.local/share/voice-dictation/live-daemon.log")
LIVE_DAEMON_STARTUP_TIMEOUT = 10  # Whisper model load can take a moment
LIVE_DAEMON_POLL_INTERVAL = 0.1
