"""Live audio recorder - streams every frame directly to the daemon.

Unlike the chunked recorder, this sends each sounddevice callback frame
immediately to the daemon via the LiveDaemonClient. Recording continues
until the user presses Enter.
"""

import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd

from .client import LiveDaemonClient
from .config import SAMPLE_RATE


class LiveRecorder:
    """Records audio and streams each frame to the live daemon."""

    def __init__(
        self,
        client: LiveDaemonClient,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._client = client
        self._sample_rate = sample_rate

    def record(self) -> None:
        """Record audio until Enter is pressed, streaming frames to daemon."""
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype=np.int16,
            callback=self._audio_callback,
        ):
            input("Recording... press Enter to stop.\n")

        self._client.finish()

    def _audio_callback(
        self,
        indata: NDArray[Any],
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Send each audio frame to the daemon."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        self._client.send_audio(indata.tobytes())
