"""Configuration constants for voice dictation."""

import os

# Audio
SAMPLE_RATE = 16000  # Hz - Whisper expects 16kHz

# Whisper Model
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Transcription Timing
TRANSCRIBE_INTERVAL = 2  # seconds between transcription cycles
MAX_WINDOW_SECONDS = 20  # finalize segments when audio exceeds this length

# Daemon
SOCKET_PATH = "/tmp/dictate-live-daemon.sock"
DAEMON_LOG = os.path.expanduser("~/.local/share/voice-dictation/live-daemon.log")
DAEMON_STARTUP_TIMEOUT = 10  # Whisper model load can take a moment
DAEMON_POLL_INTERVAL = 0.1

# UI/UX
XDOTOOL_KEYSTROKE_DELAY = 5  # milliseconds between keystrokes
BACKSPACE_SETTLE_DELAY = 0.05  # seconds - let backspaces process before typing
