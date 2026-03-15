"""Streaming client for the dictation daemon.

Maintains a persistent socket connection for an entire dictation session.
Audio frames are sent continuously; partial/final JSON results are received
on a background thread and routed to the ProgressiveTyper.
"""

import json
import logging
import socket
import threading
from typing import Optional

from dictate.config import DAEMON_POLL_INTERVAL, DAEMON_STARTUP_TIMEOUT, SOCKET_PATH
from dictate.daemon_support import is_daemon_running, start_daemon_process

from .typer import ProgressiveTyper

logger = logging.getLogger(__name__)


class LiveDaemonClient:
    """Persistent streaming client for the dictation daemon."""

    def __init__(
        self,
        typer: Optional[ProgressiveTyper] = None,
        streaming: bool = True,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._typer = typer or ProgressiveTyper()
        self._streaming = streaming
        self._stop_event = stop_event
        self._sock: Optional[socket.socket] = None
        self._receiver_thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._on_disconnect: Optional[threading.Event] = None

    @property
    def typer(self) -> ProgressiveTyper:
        return self._typer

    def set_stop_event(self, event: threading.Event) -> None:
        """Register an external event to set when the daemon disconnects."""
        self._on_disconnect = event

    def connect(self) -> None:
        """Connect to the daemon socket."""
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(SOCKET_PATH)
        self._done.clear()
        self._receiver_thread = threading.Thread(
            target=self._receive_messages, daemon=True
        )
        self._receiver_thread.start()

    def send_audio(self, frame: bytes) -> None:
        """Send a raw PCM int16 audio frame to the daemon."""
        if self._sock is None:
            return
        if self._stop_event is not None and self._stop_event.is_set():
            return
        try:
            self._sock.sendall(frame)
        except (BrokenPipeError, OSError) as e:
            logger.warning(f"Failed to send audio: {e}")

    def finish(self) -> None:
        """Signal end of audio and wait for final results."""
        if self._sock is None:
            return
        try:
            self._sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        self._done.wait(timeout=5.0)
        self._close()

    def _close(self) -> None:
        """Close the socket connection."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _receive_messages(self) -> None:
        """Background thread: read newline-delimited JSON from daemon."""
        buffer = b""
        try:
            while True:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                buffer = self._process_buffer(buffer)
        except OSError:
            pass
        finally:
            self._done.set()
            if self._on_disconnect is not None:
                self._on_disconnect.set()

    def _process_buffer(self, buffer: bytes) -> bytes:
        """Parse complete JSON lines from buffer, route to typer."""
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            self._handle_message(line.decode("utf-8"))
        return buffer

    def _handle_message(self, raw: str) -> None:
        """Route a single JSON message to the appropriate typer method."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from daemon: {raw}")
            return

        msg_type = msg.get("type")
        text = msg.get("text", "")

        if msg_type == "end":
            self._done.set()
        elif self._stop_event is not None and self._stop_event.is_set():
            return
        elif msg_type == "partial" and self._streaming:
            self._typer.apply_partial(text)
        elif msg_type == "final":
            self._typer.apply_final(text)

    @staticmethod
    def is_daemon_running() -> bool:
        """Check if the daemon socket exists."""
        return is_daemon_running(SOCKET_PATH)

    @staticmethod
    def start_daemon() -> bool:
        """Start the daemon in the background."""
        from dictate.system import notify

        notify("Loading...")

        return start_daemon_process(
            entry_point="dictate-daemon",
            module_path="dictate.live.daemon",
            socket_path=SOCKET_PATH,
            timeout=DAEMON_STARTUP_TIMEOUT,
            poll_interval=DAEMON_POLL_INTERVAL,
        )
