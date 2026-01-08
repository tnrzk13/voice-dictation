"""Daemon client for transcription requests."""

import os
import sys
import time
import socket
import pickle
import subprocess
from typing import Callable, Optional, Any
from numpy.typing import NDArray

from .config import (
    SOCKET_PATH,
    DAEMON_STARTUP_TIMEOUT,
    RECV_BUFFER_SIZE,
    END_MARKER
)
from .utils import notify, type_text


class DaemonClient:
    """Client for communicating with the dictation daemon."""

    def __init__(self, transcription_handler: Optional[Callable[[str], None]] = None):
        """
        Initialize daemon client.

        Args:
            transcription_handler: Callable that receives transcribed text segments
        """
        self.transcription_handler = transcription_handler or type_text

    @staticmethod
    def is_running() -> bool:
        """Check if daemon is running."""
        return os.path.exists(SOCKET_PATH)

    @staticmethod
    def start() -> bool:
        """Start the daemon in the background."""
        notify("Dictation", "Starting daemon (first use)...")

        python_path = sys.executable

        # Use the installed dictate-daemon command if available, otherwise fall back to module
        try:
            subprocess.Popen(
                ["dictate-daemon"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy()
            )
        except FileNotFoundError:
            # Fall back to python module execution
            subprocess.Popen(
                [python_path, "-m", "dictate.daemon"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy()
            )

        # Wait for daemon to be ready
        timeout_iterations = int(DAEMON_STARTUP_TIMEOUT / 0.1)
        for _ in range(timeout_iterations):
            time.sleep(0.1)
            if os.path.exists(SOCKET_PATH):
                return True

        return False

    def _send_audio(self, sock: socket.socket, audio: NDArray[Any]) -> None:
        """Send audio data to daemon via socket."""
        data = pickle.dumps(audio)
        sock.sendall(data)
        sock.shutdown(socket.SHUT_WR)

    def _receive_transcription(self, sock: socket.socket) -> None:
        """Receive and process transcription from daemon."""
        buffer = b""
        while True:
            chunk = sock.recv(RECV_BUFFER_SIZE)
            if not chunk:
                break

            buffer += chunk
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                text = line.decode('utf-8')

                if text == END_MARKER:
                    return

                # Send to handler (type_text by default)
                if text:
                    self.transcription_handler(text + " ")

    def transcribe(self, audio: NDArray[Any]) -> None:
        """Send audio to daemon for transcription and stream results."""
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(SOCKET_PATH)
                self._send_audio(sock, audio)
                self._receive_transcription(sock)
        except (ConnectionRefusedError, FileNotFoundError) as e:
            # Expected errors when daemon isn't running
            print(f"Daemon connection error: {e}", file=sys.stderr)
        except Exception as e:
            # Log unexpected errors for debugging
            print(f"Unexpected transcription error: {e}", file=sys.stderr)
