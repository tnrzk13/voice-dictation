"""Configuration constants for voice dictation."""

import os
from pathlib import Path

# Audio
SAMPLE_RATE = 16000  # Hz - Whisper expects 16kHz
BYTES_PER_SAMPLE = 2  # int16
BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE

# Whisper Model
WHISPER_MODEL_SIZE = "large-v3-turbo"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# Transcription Timing
TRANSCRIBE_INTERVAL = 2  # seconds between transcription cycles
MAX_WINDOW_SECONDS = 20  # finalize segments when audio exceeds this length

# Vocabulary hints - bias Whisper toward domain-specific terms it often mishears
# Loaded from hotwords.txt in the project root (one word/phrase per line)
_HOTWORDS_PATH = Path(__file__).resolve().parents[2] / "hotwords.txt"
WHISPER_HOTWORDS = " ".join(
    line.strip()
    for line in _HOTWORDS_PATH.read_text().splitlines()
    if line.strip() and not line.startswith("#")
) if _HOTWORDS_PATH.exists() else ""

# Daemon
SOCKET_PATH = "/tmp/dictate-live-daemon.sock"
DAEMON_LOG = os.path.expanduser("~/.local/share/voice-dictation/live-daemon.log")
DAEMON_STARTUP_TIMEOUT = 10  # Whisper model load can take a moment
DAEMON_POLL_INTERVAL = 0.1

# Stability - auto-commit words consistent across consecutive partials
STABILITY_THRESHOLD = 2       # consecutive agreeing partials before auto-commit
KEEP_TAIL_WORDS = 2           # always keep last N words uncommitted for corrections
MAX_PREFIX_MISMATCHES = 3     # word substitutions tolerated during committed prefix stripping (floor)
MAX_PREFIX_MISMATCH_FRACTION = 0.3  # also tolerate up to 30% mismatches for long committed text

# UI/UX
XDOTOOL_KEYSTROKE_DELAY = 12  # milliseconds between keystrokes
BACKSPACE_SETTLE_DELAY = 0.05  # seconds - let backspaces process before typing
