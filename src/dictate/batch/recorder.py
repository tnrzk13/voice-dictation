"""Audio recording with streaming transcription support."""

import sys
import queue
import threading
from typing import Any
import numpy as np
from numpy.typing import NDArray
import sounddevice as sd

from dictate.config import (
    SAMPLE_RATE,
    CHUNK_DURATION,
    PROCESSING_TIMEOUT,
    DAEMON_POLL_INTERVAL,
)
from dictate.system import notify

from .client import DaemonClient


class StreamingAudioRecorder:
    """Records audio and optionally streams chunks for transcription in real-time."""

    def __init__(
        self,
        daemon_client: DaemonClient,
        chunk_duration: float = CHUNK_DURATION,
        streaming: bool = True,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.audio_queue: queue.Queue[NDArray[Any]] = queue.Queue()
        self.recording = True
        self.daemon_client = daemon_client
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.accumulated_audio: list[NDArray[Any]] = []
        self.accumulated_samples = 0
        self.streaming = streaming

    def audio_callback(
        self,
        indata: NDArray[Any],
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags
    ) -> None:
        """Callback for audio stream - queues audio chunks."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        self.audio_queue.put(indata.copy())

    def _send_chunk_for_transcription(self, audio_chunk: NDArray[Any]) -> None:
        """Send an audio chunk to daemon for transcription in background."""
        threading.Thread(
            target=self.daemon_client.transcribe,
            args=(audio_chunk,),
            daemon=True
        ).start()

    def process_audio_chunks(self) -> None:
        """Process audio queue and send chunks for transcription periodically."""
        while self.recording or not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get(timeout=DAEMON_POLL_INTERVAL)
                self.accumulated_audio.append(chunk)
                self.accumulated_samples += len(chunk)

                # When we have enough audio, send for transcription (only in streaming mode)
                if self.streaming and self.accumulated_samples >= self.chunk_samples:
                    audio_to_transcribe = np.concatenate(self.accumulated_audio, axis=0)
                    self._send_chunk_for_transcription(audio_to_transcribe)

                    # Reset accumulator
                    self.accumulated_audio = []
                    self.accumulated_samples = 0

            except queue.Empty:
                continue

        # Send any remaining audio (or all audio if not streaming)
        if self.accumulated_audio:
            audio_to_transcribe = np.concatenate(self.accumulated_audio, axis=0)
            self.daemon_client.transcribe(audio_to_transcribe)

    def record(self) -> None:
        """Record audio until Enter is pressed, transcribing in chunks."""
        notify("Dictation", "Recording started...")

        # Start processing thread
        process_thread = threading.Thread(
            target=self.process_audio_chunks,
            daemon=True
        )
        process_thread.start()

        # Record audio
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self.audio_callback
        ):
            input("Recording... press Enter to stop.\n")

        self.recording = False

        # Wait for processing to complete
        process_thread.join(timeout=PROCESSING_TIMEOUT)
