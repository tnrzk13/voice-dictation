"""Tests for the live audio recorder."""

import numpy as np
from unittest.mock import MagicMock, patch

from dictate.live.recorder import LiveRecorder


class TestAudioCallback:
    def _make_recorder(self, silence_threshold=0.01, silence_duration=4):
        client = MagicMock()
        recorder = LiveRecorder(
            client,
            sample_rate=16000,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration,
        )
        return recorder, client

    def test_sends_audio_bytes_to_client(self):
        recorder, client = self._make_recorder()
        # Loud audio so silence detector doesn't trigger
        audio = (np.random.randn(1600, 1) * 5000).astype(np.int16)

        recorder._audio_callback(audio, 1600, None, False)

        client.send_audio.assert_called_once()
        sent = client.send_audio.call_args[0][0]
        assert isinstance(sent, bytes)
        assert len(sent) == 1600 * 2  # int16 = 2 bytes per sample

    def test_silence_stops_recording(self):
        recorder, _ = self._make_recorder(silence_threshold=0.01, silence_duration=0.1)
        silent = np.zeros((1600, 1), dtype=np.int16)

        # Feed enough silence to exceed duration (1600 samples / 16000 Hz = 0.1s)
        recorder._audio_callback(silent, 1600, None, False)

        assert recorder._recording is False

    def test_loud_audio_does_not_stop_recording(self):
        recorder, _ = self._make_recorder()
        loud = (np.ones((1600, 1)) * 10000).astype(np.int16)

        recorder._audio_callback(loud, 1600, None, False)

        assert recorder._recording is True

    def test_silence_resets_on_loud_audio(self):
        recorder, _ = self._make_recorder(silence_threshold=0.01, silence_duration=0.2)
        silent = np.zeros((1600, 1), dtype=np.int16)
        loud = (np.ones((1600, 1)) * 10000).astype(np.int16)

        # Almost enough silence
        recorder._audio_callback(silent, 1600, None, False)
        assert recorder._recording is True

        # Reset with loud audio
        recorder._audio_callback(loud, 1600, None, False)
        assert recorder._recording is True

        # Not enough silence again after reset
        recorder._audio_callback(silent, 1600, None, False)
        assert recorder._recording is True
