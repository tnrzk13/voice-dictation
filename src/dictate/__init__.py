"""Voice dictation with streaming transcription using Whisper."""

__version__ = "1.0.0"

from .client import DaemonClient
from .recorder import StreamingAudioRecorder
from .silence import SilenceDetector
from .config import (
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    CHUNK_DURATION,
    SOCKET_PATH,
)
from .utils import notify, type_text

__all__ = [
    "DaemonClient",
    "StreamingAudioRecorder",
    "SilenceDetector",
    "SAMPLE_RATE",
    "SILENCE_THRESHOLD",
    "SILENCE_DURATION",
    "CHUNK_DURATION",
    "SOCKET_PATH",
    "notify",
    "type_text",
    "__version__",
]
