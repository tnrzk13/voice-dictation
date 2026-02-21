"""Tests for the live dictation daemon message protocol."""

import json
import time
from unittest.mock import MagicMock, patch

from dictate.live.daemon import _send_message, handle_client


class TestSendMessage:
    def test_sends_partial_as_json(self):
        conn = MagicMock()
        _send_message(conn, "partial", "hello")
        sent = conn.sendall.call_args[0][0]
        msg = json.loads(sent.decode("utf-8").strip())
        assert msg == {"type": "partial", "text": "hello"}

    def test_sends_final_as_json(self):
        conn = MagicMock()
        _send_message(conn, "final", "hello world")
        sent = conn.sendall.call_args[0][0]
        msg = json.loads(sent.decode("utf-8").strip())
        assert msg == {"type": "final", "text": "hello world"}

    def test_sends_end_marker(self):
        conn = MagicMock()
        _send_message(conn, "end", "")
        sent = conn.sendall.call_args[0][0]
        msg = json.loads(sent.decode("utf-8").strip())
        assert msg == {"type": "end", "text": ""}

    def test_messages_are_newline_terminated(self):
        conn = MagicMock()
        _send_message(conn, "partial", "test")
        sent = conn.sendall.call_args[0][0]
        assert sent.endswith(b"\n")


def _make_segment(text=" Hello world.", start=0.0, end=1.0):
    """Create a mock Whisper segment with .text, .start, .end attributes."""
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    return seg


def _make_whisper_model(segments=None):
    """Create a mock Whisper model returning fresh segment iterators per call.

    Args:
        segments: List of mock segments to return from transcribe().
                  Defaults to a single segment with " Hello world."
    """
    if segments is None:
        segments = [_make_segment()]
    model = MagicMock()
    model.transcribe.side_effect = lambda *a, **kw: (iter(list(segments)), None)
    return model


def _parse_sent_messages(conn):
    """Extract all JSON messages sent via sendall on a mock connection."""
    messages = []
    for call in conn.sendall.call_args_list:
        raw = call[0][0].decode("utf-8").strip()
        messages.append(json.loads(raw))
    return messages


class TestHandleClient:
    @patch("dictate.live.daemon.TRANSCRIBE_INTERVAL", 0.01)
    def test_sends_partial_results(self):
        """Transcription during the session produces partial messages."""
        model = _make_whisper_model([_make_segment(" Hello world.")])

        conn = MagicMock()
        audio = b"\x00" * 8000
        calls = []

        def recv_with_delay(size):
            calls.append(1)
            if len(calls) == 1:
                return audio
            # Block receiver so transcriber can run at least one cycle
            time.sleep(0.1)
            return b""

        conn.recv.side_effect = recv_with_delay

        handle_client(conn, model)

        messages = _parse_sent_messages(conn)
        partials = [m for m in messages if m["type"] == "partial"]
        assert len(partials) >= 1
        assert "Hello world." in partials[0]["text"]

    def test_sends_final_on_eof(self):
        """Final transcription is sent when the client finishes."""
        model = _make_whisper_model([_make_segment(" Final words.")])

        conn = MagicMock()
        conn.recv.side_effect = [b"\x00" * 8000, b""]

        handle_client(conn, model)

        messages = _parse_sent_messages(conn)
        finals = [m for m in messages if m["type"] == "final"]
        assert any("Final words." in m["text"] for m in finals)

    def test_sends_end_on_completion(self):
        """End marker is always the last message sent."""
        model = _make_whisper_model()

        conn = MagicMock()
        conn.recv.side_effect = [b"\x00" * 8000, b""]

        handle_client(conn, model)

        messages = _parse_sent_messages(conn)
        assert messages[-1]["type"] == "end"

    def test_skips_empty_transcription(self):
        """No partial/final messages are sent when Whisper returns nothing."""
        model = _make_whisper_model([])

        conn = MagicMock()
        conn.recv.side_effect = [b"\x00" * 8000, b""]

        handle_client(conn, model)

        messages = _parse_sent_messages(conn)
        assert all(m["type"] == "end" for m in messages)

    def test_handles_client_disconnect(self):
        """Daemon handles abrupt client disconnection without crashing."""
        model = _make_whisper_model()

        conn = MagicMock()
        conn.recv.side_effect = ConnectionResetError("client gone")

        # Should not raise
        handle_client(conn, model)
