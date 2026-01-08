"""Silence detection for audio streams."""

from typing import Any
import numpy as np
from numpy.typing import NDArray

from .config import SILENCE_THRESHOLD, SILENCE_DURATION


class SilenceDetector:
    """Detects silence in audio stream using RMS energy."""

    def __init__(
        self,
        threshold: float = SILENCE_THRESHOLD,
        silence_duration: float = SILENCE_DURATION
    ):
        self.threshold = threshold
        self.required_silence = silence_duration
        self.current_silence = 0.0

    def calculate_rms(self, audio_chunk: NDArray[Any]) -> float:
        """Calculate RMS (Root Mean Square) energy of audio chunk."""
        return float(np.sqrt(np.mean(audio_chunk**2)))

    def is_silent(self, audio_chunk: NDArray[Any]) -> bool:
        """Check if audio chunk is silent based on RMS energy."""
        rms = self.calculate_rms(audio_chunk)
        return rms < self.threshold

    def update(self, audio_chunk: NDArray[Any], chunk_duration: float) -> bool:
        """
        Update silence timer and return True if silence threshold reached.

        Returns:
            bool: True if continuous silence exceeds required duration
        """
        if self.is_silent(audio_chunk):
            self.current_silence += chunk_duration
            return self.current_silence >= self.required_silence
        else:
            self.current_silence = 0.0
            return False

    def reset(self) -> None:
        """Reset the silence timer."""
        self.current_silence = 0.0
