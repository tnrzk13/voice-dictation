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
WHISPER_COMPUTE_TYPE = "int8_float16"

# Transcription
WHISPER_BEAM_SIZE = 1
WHISPER_TEMPERATURE = 0.0
WHISPER_VAD_FILTER = True
WHISPER_VAD_MIN_SILENCE_MS = 500
WHISPER_NO_REPEAT_NGRAM_SIZE = 3
WHISPER_REPETITION_PENALTY = 1.2

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
DAEMON_STARTUP_TIMEOUT = 10  # seconds - Whisper model load (no download)
DAEMON_DOWNLOAD_TIMEOUT = 300  # seconds - first-time model download + load
DAEMON_POLL_INTERVAL = 0.1
DAEMON_FINISH_TIMEOUT = 30  # seconds - wait for daemon final results after EOF

# Socket
SOCKET_TIMEOUT = 5  # seconds - connect and daemon recv timeouts so hangs fail cleanly
AUDIO_RECV_BUFFER_BYTES = 8000  # bytes read per socket recv in the daemon receiver
MESSAGE_RECV_BUFFER_BYTES = 4096  # bytes read per socket recv for JSON messages

# Buffering
MAX_BUFFER_SECONDS = 60  # hard cap on audio buffer growth when model is slow

# Stability - auto-commit words consistent across consecutive partials
STABILITY_THRESHOLD = 2       # consecutive agreeing partials before auto-commit
KEEP_TAIL_WORDS = 2           # always keep last N words uncommitted for corrections
MAX_PREFIX_MISMATCHES = 3     # word substitutions tolerated during committed prefix stripping (floor)
MAX_PREFIX_MISMATCH_FRACTION = 0.3  # also tolerate up to 30% mismatches for long committed text

# UI/UX
XDOTOOL_KEYSTROKE_DELAY = 12  # milliseconds between keystrokes
BACKSPACE_SETTLE_DELAY = 0.05  # seconds - let backspaces process before typing
