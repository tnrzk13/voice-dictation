"""Configuration constants for voice dictation."""

import os

# Audio Configuration
SAMPLE_RATE = 16000  # Hz - Audio sample rate. Higher = better quality but more data
                      # Whisper works best with 16000. Common values: 8000, 16000, 44100

SILENCE_THRESHOLD = 0.01  # RMS energy threshold for detecting silence (range: 0.001-0.1)
                           # Lower values = more sensitive to sound (picks up quieter sounds)
                           # Higher values = less sensitive (requires louder sounds)
                           # Adjust based on your microphone and room noise level

SILENCE_DURATION = 2  # seconds - Duration of continuous silence before stopping recording
                       # Increase if you need longer pauses while speaking
                       # Decrease for quicker stops after finishing dictation

CHUNK_DURATION = 5  # seconds - In streaming mode, transcribe audio every N seconds
                     # Lower = more responsive but may be less accurate for long sentences
                     # Higher = better context but less real-time feedback
                     # Recommended range: 3-10 seconds

# Daemon Configuration
SOCKET_PATH = "/tmp/dictate-daemon.sock"
DAEMON_LOG = os.path.expanduser("~/.local/share/voice-dictation/daemon.log")
DAEMON_STARTUP_TIMEOUT = 5  # seconds to wait for daemon to start
PROCESSING_TIMEOUT = 10  # seconds to wait for final processing

# Network Configuration
RECV_BUFFER_SIZE = 1024
DAEMON_RECV_BUFFER_SIZE = 4096  # Larger buffer for daemon to receive audio data
END_MARKER = "__END__"

# UI/UX Configuration
PRE_RECORDING_DELAY = 2  # seconds - delay before starting recording (time to switch windows)
DAEMON_POLL_INTERVAL = 0.1  # seconds - interval for checking daemon status
XDOTOOL_KEYSTROKE_DELAY = 5  # milliseconds - delay between keystrokes when typing
