"""Configuration constants for live streaming dictation."""

import os

from dictate.config import SAMPLE_RATE, SILENCE_THRESHOLD

# Live dictation uses a longer silence duration than batch mode since users
# pause more naturally during continuous dictation.
SILENCE_DURATION = 8

# Vosk Model Configuration
VOSK_MODEL_NAME = "vosk-model-en-us-0.22"
VOSK_MODEL_DIR = os.path.expanduser("~/.local/share/voice-dictation/models")

# Daemon Configuration
LIVE_SOCKET_PATH = "/tmp/dictate-live-daemon.sock"
LIVE_DAEMON_LOG = os.path.expanduser("~/.local/share/voice-dictation/live-daemon.log")
LIVE_DAEMON_STARTUP_TIMEOUT = 10  # seconds - Vosk model is large, allow more time
LIVE_DAEMON_POLL_INTERVAL = 0.1

# Re-export parent config values used by live modules
__all__ = [
    "SAMPLE_RATE",
    "SILENCE_THRESHOLD",
    "SILENCE_DURATION",
    "VOSK_MODEL_NAME",
    "VOSK_MODEL_DIR",
    "LIVE_SOCKET_PATH",
    "LIVE_DAEMON_LOG",
    "LIVE_DAEMON_STARTUP_TIMEOUT",
    "LIVE_DAEMON_POLL_INTERVAL",
]
