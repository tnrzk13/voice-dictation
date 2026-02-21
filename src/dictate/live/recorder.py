"""Live audio recorder - streams every frame directly to the daemon.

Unlike the chunked recorder, this sends each sounddevice callback frame
immediately to the daemon via the LiveDaemonClient. Recording continues
until the user presses a key or the process is killed (via dictate-stop /
Shift+Super+D).
"""

import sys
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd

from dictate.config import SAMPLE_RATE

from .client import LiveDaemonClient
from .keyboard_monitor import KeyboardMonitor


class LiveRecorder:
    """Records audio and streams each frame to the live daemon."""

    def __init__(
        self,
        client: LiveDaemonClient,
        typer=None,
        sample_rate: int = SAMPLE_RATE,
        stop_event: threading.Event = None,
    ) -> None:
        self._client = client
        self._typer = typer
        self._sample_rate = sample_rate
        self._stop = stop_event or threading.Event()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop

    def record(self) -> None:
        """Record audio until user presses a key or process is killed."""
        monitor = self._start_keyboard_monitor()
        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype=np.int16,
                callback=self._audio_callback,
            ):
                self._stop.wait()
        finally:
            if monitor is not None:
                monitor.stop()

        self._client.finish()

    def _start_keyboard_monitor(self):
        if self._typer is None:
            return None
        monitor = KeyboardMonitor(self._stop, self._typer)
        monitor.start()
        return monitor

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
