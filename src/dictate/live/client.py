"""Streaming client for the live dictation daemon.

Maintains a persistent socket connection for an entire dictation session.
Audio frames are sent continuously; partial/final JSON results are received
on a background thread and routed to the ProgressiveTyper.
"""

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

from .config import (
    LIVE_DAEMON_POLL_INTERVAL,
    LIVE_DAEMON_STARTUP_TIMEOUT,
    LIVE_SOCKET_PATH,
)
from .typer import ProgressiveTyper

logger = logging.getLogger(__name__)


PARTIAL_DEBOUNCE_SECONDS = 0.1


class LiveDaemonClient:
    """Persistent streaming client for the live dictation daemon."""

    def __init__(self, typer: Optional[ProgressiveTyper] = None) -> None:
        self._typer = typer or ProgressiveTyper()
        self._sock: Optional[socket.socket] = None
        self._receiver_thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._typer_lock = threading.Lock()
        self._partial_timer: Optional[threading.Timer] = None

    @property
    def typer(self) -> ProgressiveTyper:
        return self._typer

    def connect(self) -> None:
        """Connect to the live daemon socket."""
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(LIVE_SOCKET_PATH)
        self._done.clear()
        self._receiver_thread = threading.Thread(
            target=self._receive_messages, daemon=True
        )
        self._receiver_thread.start()

    def send_audio(self, frame: bytes) -> None:
        """Send a raw PCM int16 audio frame to the daemon."""
        if self._sock is None:
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

        if msg_type == "partial":
            self._debounce_partial(text)
        elif msg_type == "final":
            self._flush_partial_and_apply_final(text)
        elif msg_type == "end":
            self._cancel_pending_partial()
            self._done.set()

    def _debounce_partial(self, text: str) -> None:
        """Schedule a partial update, cancelling any pending one."""
        self._cancel_pending_partial()
        self._partial_timer = threading.Timer(
            PARTIAL_DEBOUNCE_SECONDS, self._apply_partial, args=[text]
        )
        self._partial_timer.start()

    def _apply_partial(self, text: str) -> None:
        """Apply a partial result under lock."""
        with self._typer_lock:
            self._typer.apply_partial(text)

    def _flush_partial_and_apply_final(self, text: str) -> None:
        """Cancel any pending partial and apply the final result."""
        self._cancel_pending_partial()
        with self._typer_lock:
            self._typer.apply_final(text)

    def _cancel_pending_partial(self) -> None:
        """Cancel a scheduled partial if one exists."""
        if self._partial_timer is not None:
            self._partial_timer.cancel()
            self._partial_timer = None

    @staticmethod
    def is_daemon_running() -> bool:
        """Check if the live daemon socket exists."""
        return os.path.exists(LIVE_SOCKET_PATH)

    @staticmethod
    def start_daemon() -> bool:
        """Start the live daemon in the background."""
        from dictate.utils import notify

        notify("Live Dictation", "Starting daemon (loading Vosk model)...")

        try:
            subprocess.Popen(
                ["dictate-live-daemon"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
        except FileNotFoundError:
            subprocess.Popen(
                [sys.executable, "-m", "dictate.live.daemon"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )

        return _wait_for_socket(LIVE_SOCKET_PATH, LIVE_DAEMON_STARTUP_TIMEOUT)


def _wait_for_socket(path: str, timeout: float) -> bool:
    """Poll until socket file appears or timeout expires."""
    iterations = int(timeout / LIVE_DAEMON_POLL_INTERVAL)
    for _ in range(iterations):
        time.sleep(LIVE_DAEMON_POLL_INTERVAL)
        if os.path.exists(path):
            return True
    return False
