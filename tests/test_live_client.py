"""Tests for the live dictation streaming client."""

import json
from unittest.mock import MagicMock, patch

from dictate.live.client import LiveDaemonClient
from dictate.live.typer import ProgressiveTyper


class TestHandleMessage:
    def _make_client(self):
        typer = MagicMock(spec=ProgressiveTyper)
        client = LiveDaemonClient(typer=typer)
        return client, typer

    def test_routes_partial_to_typer(self):
        client, typer = self._make_client()
        client._handle_message(json.dumps({"type": "partial", "text": "hel"}))
        typer.apply_partial.assert_called_once_with("hel")

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


class TestProcessBuffer:
    def _make_client(self):
        typer = MagicMock(spec=ProgressiveTyper)
        client = LiveDaemonClient(typer=typer)
        return client, typer

    def test_parses_complete_lines(self):
        client, typer = self._make_client()
        buffer = json.dumps({"type": "partial", "text": "hi"}).encode() + b"\n"
        remaining = client._process_buffer(buffer)
        assert remaining == b""
        typer.apply_partial.assert_called_once_with("hi")

    def test_preserves_incomplete_line(self):
        client, typer = self._make_client()
        complete = json.dumps({"type": "partial", "text": "a"}).encode() + b"\n"
        incomplete = b'{"type": "partial"'
        remaining = client._process_buffer(complete + incomplete)
        assert remaining == incomplete
        typer.apply_partial.assert_called_once_with("a")

    def test_handles_multiple_lines(self):
        client, typer = self._make_client()
        line1 = json.dumps({"type": "partial", "text": "a"}).encode() + b"\n"
        line2 = json.dumps({"type": "final", "text": "ab"}).encode() + b"\n"
        client._process_buffer(line1 + line2)
        typer.apply_partial.assert_called_once_with("a")
        typer.apply_final.assert_called_once_with("ab")


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
