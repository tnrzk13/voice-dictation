"""Configuration constants for voice dictation."""

import os

# Audio
SAMPLE_RATE = 16000  # Hz - Whisper expects 16kHz
SILENCE_THRESHOLD = 0.01  # RMS energy below this = silence
SILENCE_DURATION = 2  # seconds of silence before auto-stop
CHUNK_DURATION = 5  # seconds between streaming transcriptions

# Whisper Model (shared by batch and live daemons)
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Daemon
SOCKET_PATH = "/tmp/dictate-daemon.sock"
DAEMON_LOG = os.path.expanduser("~/.local/share/voice-dictation/daemon.log")
DAEMON_STARTUP_TIMEOUT = 5  # seconds to wait for daemon to start
PROCESSING_TIMEOUT = 10  # seconds to wait for final processing

# Network
RECV_BUFFER_SIZE = 1024
DAEMON_RECV_BUFFER_SIZE = 4096
END_MARKER = "__END__"

# UI/UX
PRE_RECORDING_DELAY = 2  # seconds before recording starts (time to switch windows)
DAEMON_POLL_INTERVAL = 0.1
XDOTOOL_KEYSTROKE_DELAY = 5  # milliseconds between keystrokes
