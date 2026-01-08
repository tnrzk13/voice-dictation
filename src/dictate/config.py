"""Configuration constants for voice dictation."""

import os

# Audio Configuration
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01  # RMS energy threshold (lower = more sensitive to sound)
SILENCE_DURATION = 4  # seconds of continuous silence before stopping
CHUNK_DURATION = 5  # seconds - transcribe audio every N seconds

# Daemon Configuration
SOCKET_PATH = "/tmp/dictate-daemon.sock"
DAEMON_LOG = os.path.expanduser("~/.local/share/voice-dictation/daemon.log")
DAEMON_STARTUP_TIMEOUT = 5  # seconds to wait for daemon to start
PROCESSING_TIMEOUT = 10  # seconds to wait for final processing

# Network Configuration
RECV_BUFFER_SIZE = 1024
END_MARKER = "__END__"
