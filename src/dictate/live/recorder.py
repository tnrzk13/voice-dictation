"""Live audio recorder - streams every frame directly to the daemon.

Unlike the chunked recorder, this sends each sounddevice callback frame
immediately to the daemon via the LiveDaemonClient. Silence detection
ends the session.
"""

import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd

from dictate.silence import SilenceDetector

from .client import LiveDaemonClient
from .config import (
    LIVE_DAEMON_POLL_INTERVAL,
    SAMPLE_RATE,
    SILENCE_DURATION,
    SILENCE_THRESHOLD,
)


class LiveRecorder:
    """Records audio and streams each frame to the live daemon."""

    def __init__(
        self,
        client: LiveDaemonClient,
        sample_rate: int = SAMPLE_RATE,
        silence_threshold: float = SILENCE_THRESHOLD,
        silence_duration: float = SILENCE_DURATION,
    ) -> None:
        self._client = client
        self._sample_rate = sample_rate
        self._detector = SilenceDetector(
            threshold=silence_threshold,
            silence_duration=silence_duration,
        )
        self._recording = True

    def record(self) -> None:
        """Record audio until silence detected, streaming frames to daemon."""
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype=np.int16,
            callback=self._audio_callback,
        ):
            while self._recording:
                time.sleep(LIVE_DAEMON_POLL_INTERVAL)

        self._client.finish()

    def _audio_callback(
        self,
        indata: NDArray[Any],
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Send each audio frame to the daemon and check for silence."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        self._client.send_audio(indata.tobytes())

        # SilenceDetector expects float32 for RMS calculation
        float_data = indata.astype(np.float32) / 32768.0
        chunk_duration = frames / self._sample_rate
        if self._detector.update(float_data, chunk_duration):
            self._recording = False
