"""Tests for the live audio recorder."""

import threading

import numpy as np
from unittest.mock import MagicMock

from dictate.live.recorder import LiveRecorder


class TestAudioCallback:
    def _make_recorder(self, stop_event=None):
        client = MagicMock()
        recorder = LiveRecorder(client, sample_rate=16000, stop_event=stop_event)
        return recorder, client

    def test_sends_audio_bytes_to_client(self):
        recorder, client = self._make_recorder()
        audio = (np.random.randn(1600, 1) * 5000).astype(np.int16)

        recorder._audio_callback(audio, 1600, None, False)

        client.send_audio.assert_called_once()
        sent = client.send_audio.call_args[0][0]
        assert isinstance(sent, bytes)
        assert len(sent) == 1600 * 2  # int16 = 2 bytes per sample

    def test_drops_audio_after_stop_event(self):
        """Audio callback stops sending frames once stop_event is set."""
        stop = threading.Event()
        recorder, client = self._make_recorder(stop_event=stop)
        audio = (np.random.randn(1600, 1) * 5000).astype(np.int16)

        stop.set()
        recorder._audio_callback(audio, 1600, None, False)

        client.send_audio.assert_not_called()
