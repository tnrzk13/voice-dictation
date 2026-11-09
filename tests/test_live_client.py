"""Tests for the live dictation streaming client."""

import json
import logging
import threading
from unittest.mock import MagicMock, call

from dictate.live.client import LiveDaemonClient
from dictate.live.typer import ProgressiveTyper


def _encode_lines(*messages: dict) -> bytes:
    """Encode messages as the newline-delimited JSON the daemon sends."""
    return b"".join(json.dumps(message).encode() + b"\n" for message in messages)


class TestHandleMessage:
    def _make_client(self):
        typer = MagicMock(spec=ProgressiveTyper)
        client = LiveDaemonClient(typer=typer)
        return client, typer

    def test_routes_partial_to_typer(self):
        client, typer = self._make_client()
        client._handle_message(json.dumps({"type": "partial", "text": "hel"}))
        typer.apply_partial.assert_called_once_with("hel", "")

    def test_routes_finalized_prefix_to_typer(self):
        client, typer = self._make_client()
        client._handle_message(
            json.dumps({"type": "partial", "text": "hello world", "finalized": "hello"})
        )
        typer.apply_partial.assert_called_once_with("hello world", "hello")

    def test_routes_final_to_typer(self):
        client, typer = self._make_client()
        client._handle_message(json.dumps({"type": "final", "text": "hello"}))
        typer.apply_final.assert_called_once_with("hello")

    def test_end_signals_done(self):
        client, _ = self._make_client()
        assert not client._done.is_set()
        client._handle_message(json.dumps({"type": "end", "text": ""}))
        assert client._done.is_set()

    def test_ignores_invalid_json(self):
        client, typer = self._make_client()
        client._handle_message("not json at all")
        typer.apply_partial.assert_not_called()
        typer.apply_final.assert_not_called()

    def test_all_messages_suppressed_after_stop_event(self):
        """Partials and finals are suppressed after stop - text is already on screen."""
        client, typer = self._make_client()
        stop = threading.Event()
        stop.set()
        client._stop_event = stop
        client._handle_message(json.dumps({"type": "partial", "text": "hello"}))
        client._handle_message(json.dumps({"type": "final", "text": "hello"}))
        typer.apply_partial.assert_not_called()
        typer.apply_final.assert_not_called()


class TestProcessBuffer:
    def _make_client(self):
        typer = MagicMock(spec=ProgressiveTyper)
        client = LiveDaemonClient(typer=typer)
        return client, typer

    def test_parses_complete_lines(self):
        client, typer = self._make_client()
        buffer = json.dumps({"type": "final", "text": "hi"}).encode() + b"\n"
        remaining = client._process_buffer(buffer)
        assert remaining == b""
        typer.apply_final.assert_called_once_with("hi")

    def test_preserves_incomplete_line(self):
        client, typer = self._make_client()
        complete = json.dumps({"type": "final", "text": "a"}).encode() + b"\n"
        incomplete = b'{"type": "partial"'
        remaining = client._process_buffer(complete + incomplete)
        assert remaining == incomplete
        typer.apply_final.assert_called_once_with("a")

    def test_handles_multiple_finals(self):
        client, typer = self._make_client()
        line1 = json.dumps({"type": "final", "text": "a"}).encode() + b"\n"
        line2 = json.dumps({"type": "final", "text": "ab"}).encode() + b"\n"
        client._process_buffer(line1 + line2)
        assert typer.apply_final.call_count == 2

    def test_coalesces_consecutive_partials_to_last(self):
        client, typer = self._make_client()
        buffer = _encode_lines(
            {"type": "partial", "text": "hel", "finalized": ""},
            {"type": "partial", "text": "hello world", "finalized": "hello"},
        )
        client._process_buffer(buffer)
        typer.apply_partial.assert_called_once_with("hello world", "hello")

    def test_partial_then_final_applies_both_in_order(self):
        client, typer = self._make_client()
        buffer = _encode_lines(
            {"type": "partial", "text": "hello world", "finalized": "hello"},
            {"type": "final", "text": "hello world."},
        )
        client._process_buffer(buffer)
        assert typer.mock_calls == [
            call.apply_partial("hello world", "hello"),
            call.apply_final("hello world."),
        ]

    def test_partials_then_end_delivers_last_partial_and_end(self):
        client, typer = self._make_client()
        buffer = _encode_lines(
            {"type": "partial", "text": "hel", "finalized": ""},
            {"type": "partial", "text": "hello", "finalized": ""},
            {"type": "end", "text": ""},
        )
        client._process_buffer(buffer)
        typer.apply_partial.assert_called_once_with("hello", "")
        assert client._done.is_set()

    def test_partials_separated_by_final_are_not_coalesced(self):
        client, typer = self._make_client()
        buffer = _encode_lines(
            {"type": "partial", "text": "a", "finalized": ""},
            {"type": "final", "text": "a "},
            {"type": "partial", "text": "ab", "finalized": ""},
        )
        client._process_buffer(buffer)
        typed_partials = [c.args[0] for c in typer.apply_partial.call_args_list]
        assert typed_partials == ["a", "ab"]
        typer.apply_final.assert_called_once_with("a ")

    def test_logs_and_skips_malformed_line(self, caplog):
        client, typer = self._make_client()
        buffer = b"not json at all\n" + _encode_lines({"type": "final", "text": "hi"})
        with caplog.at_level(logging.WARNING):
            client._process_buffer(buffer)
        typer.apply_final.assert_called_once_with("hi")
        assert "Invalid JSON from daemon" in caplog.text

    def test_single_partial_unchanged(self):
        client, typer = self._make_client()
        client._process_buffer(_encode_lines({"type": "partial", "text": "hel"}))
        typer.apply_partial.assert_called_once_with("hel", "")


class TestSendAudio:
    def test_sends_bytes_to_socket(self):
        client = LiveDaemonClient()
        client._sock = MagicMock()
        frame = b"\x00" * 3200
        client.send_audio(frame)
        client._sock.sendall.assert_called_once_with(frame)

    def test_no_error_when_not_connected(self):
        client = LiveDaemonClient()
        client.send_audio(b"\x00" * 100)  # Should not raise

    def test_drops_audio_after_stop_event(self):
        """send_audio stops writing to socket once stop_event is set."""
        stop = threading.Event()
        client = LiveDaemonClient(stop_event=stop)
        client._sock = MagicMock()

        stop.set()
        client.send_audio(b"\x00" * 3200)

        client._sock.sendall.assert_not_called()


class TestFinish:
    def test_shuts_down_write_side(self):
        import socket as sock_mod

        client = LiveDaemonClient()
        mock_sock = MagicMock()
        client._sock = mock_sock
        client._done = MagicMock()
        client._done.wait.return_value = True

        client.finish()

        mock_sock.shutdown.assert_called_once_with(sock_mod.SHUT_WR)

    def test_no_error_when_not_connected(self):
        client = LiveDaemonClient()
        client.finish()  # Should not raise
