"""Tests for the live audio recorder."""

import numpy as np
from unittest.mock import MagicMock

from dictate.live.recorder import LiveRecorder


class TestAudioCallback:
    def _make_recorder(self):
        client = MagicMock()
        recorder = LiveRecorder(client, sample_rate=16000)
        return recorder, client

    def test_sends_audio_bytes_to_client(self):
        recorder, client = self._make_recorder()
        audio = (np.random.randn(1600, 1) * 5000).astype(np.int16)

        recorder._audio_callback(audio, 1600, None, False)

        client.send_audio.assert_called_once()
        sent = client.send_audio.call_args[0][0]
        assert isinstance(sent, bytes)
        assert len(sent) == 1600 * 2  # int16 = 2 bytes per sample
