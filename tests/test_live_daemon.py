"""Tests for the live dictation daemon message protocol."""

import json
from unittest.mock import MagicMock

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


def _make_recognizer(accept_results=None, partial_text="", final_text=""):
    """Create a mock recognizer with configurable behavior.

    Args:
        accept_results: List of bools for AcceptWaveform return values.
        partial_text: Text returned by PartialResult.
        final_text: Text returned by FinalResult (end-of-stream flush).
    """
    recognizer = MagicMock()
    if accept_results is not None:
        recognizer.AcceptWaveform.side_effect = accept_results
    else:
        recognizer.AcceptWaveform.return_value = False
    recognizer.PartialResult.return_value = json.dumps({"partial": partial_text})
    recognizer.FinalResult.return_value = json.dumps({"text": final_text})
    return recognizer


def _parse_sent_messages(conn):
    """Extract all JSON messages sent via sendall on a mock connection."""
    messages = []
    for call in conn.sendall.call_args_list:
        raw = call[0][0].decode("utf-8").strip()
        messages.append(json.loads(raw))
    return messages


class TestHandleClient:
    def test_sends_final_on_accepted_waveform(self):
        recognizer = _make_recognizer(accept_results=[True, False])
        recognizer.Result.return_value = json.dumps({"text": "hello world"})
        recognizer.PartialResult.return_value = json.dumps({"partial": ""})

        conn = MagicMock()
        conn.recv.side_effect = [b"\x00" * 8000, b"\x00" * 8000, b""]

        handle_client(conn, model=None, recognizer=recognizer)

        messages = _parse_sent_messages(conn)
        types = [m["type"] for m in messages]
        assert "final" in types
        assert "end" in types

    def test_sends_partial_results(self):
        recognizer = _make_recognizer(partial_text="hel")

        conn = MagicMock()
        conn.recv.side_effect = [b"\x00" * 8000, b""]

        handle_client(conn, model=None, recognizer=recognizer)

        messages = _parse_sent_messages(conn)
        partials = [m for m in messages if m["type"] == "partial"]
        assert len(partials) >= 1
        assert partials[0]["text"] == "hel"

    def test_sends_end_marker_on_completion(self):
        recognizer = _make_recognizer()

        conn = MagicMock()
        conn.recv.return_value = b""

        handle_client(conn, model=None, recognizer=recognizer)

        messages = _parse_sent_messages(conn)
        assert messages[-1]["type"] == "end"

    def test_flushes_remaining_audio_on_eof(self):
        recognizer = _make_recognizer(final_text="final words")

        conn = MagicMock()
        conn.recv.side_effect = [b"\x00" * 8000, b""]

        handle_client(conn, model=None, recognizer=recognizer)

        messages = _parse_sent_messages(conn)
        finals = [m for m in messages if m["type"] == "final"]
        assert any(m["text"] == "final words" for m in finals)

    def test_skips_empty_text(self):
        recognizer = _make_recognizer(accept_results=[True])
        recognizer.Result.return_value = json.dumps({"text": ""})

        conn = MagicMock()
        conn.recv.side_effect = [b"\x00" * 8000, b""]

        handle_client(conn, model=None, recognizer=recognizer)

        messages = _parse_sent_messages(conn)
        # Only end marker should appear - no empty finals/partials
        assert all(m["type"] == "end" or m["text"] for m in messages)

    def test_handles_client_disconnect(self):
        recognizer = MagicMock()

        conn = MagicMock()
        conn.recv.side_effect = ConnectionResetError("client gone")

        # Should not raise
        handle_client(conn, model=None, recognizer=recognizer)
