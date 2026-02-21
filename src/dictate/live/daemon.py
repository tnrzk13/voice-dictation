"""Live dictation daemon - keeps Whisper model loaded for streaming transcription.

Receives raw PCM int16 audio frames over a persistent socket connection,
accumulates them into an audio buffer, and periodically transcribes with
faster-whisper. Streams back partial/final results as newline-delimited JSON.

Architecture:
  - Receiver thread: reads raw PCM bytes from socket, appends to shared buffer
  - Transcriber thread: every ~2s, transcribes accumulated audio with Whisper

Protocol:
  Client sends: raw PCM int16 bytes (continuous stream)
  Client sends: EOF (shutdown write side) to signal end
  Daemon sends: {"type": "partial", "text": "..."}\n
  Daemon sends: {"type": "final", "text": "..."}\n
  Daemon sends: {"type": "end"}\n
"""

import json
import logging
import socket
import sys
import threading

import numpy as np

from dictate.config import SAMPLE_RATE
from dictate.daemon_support import (
    cleanup_socket,
    create_daemon_socket,
    setup_daemon_logging,
)

from .config import (
    LIVE_DAEMON_LOG,
    LIVE_SOCKET_PATH,
    MAX_WINDOW_SECONDS,
    TRANSCRIBE_INTERVAL,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)


def _load_whisper_model():
    """Load faster-whisper model, failing fast if unavailable."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logging.error(
            "faster-whisper package not installed. "
            "Install with: pip install faster-whisper>=0.10.0"
        )
        sys.exit(1)

    logging.info(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    logging.info("Whisper model loaded.")
    return model


def handle_client(connection: socket.socket, model) -> None:
    """Process a single client's streaming audio session.

    Spawns a receiver thread to collect audio and runs the transcription
    loop on the current thread.
    """
    audio_buffer = bytearray()
    buffer_lock = threading.Lock()
    client_done = threading.Event()

    receiver = threading.Thread(
        target=_receive_audio,
        args=(connection, audio_buffer, buffer_lock, client_done),
        daemon=True,
    )
    receiver.start()

    try:
        _transcribe_loop(connection, model, audio_buffer, buffer_lock, client_done)
    except (ConnectionResetError, BrokenPipeError) as e:
        logging.warning(f"Client disconnected: {e}")
    except Exception as e:
        logging.error(f"Error handling client: {e}", exc_info=True)


def _receive_audio(
    connection: socket.socket,
    audio_buffer: bytearray,
    buffer_lock: threading.Lock,
    client_done: threading.Event,
) -> None:
    """Receiver thread: read raw PCM bytes from socket into shared buffer."""
    try:
        while True:
            data = connection.recv(8000)
            if not data:
                break
            with buffer_lock:
                audio_buffer.extend(data)
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        client_done.set()


def _transcribe_loop(
    connection: socket.socket,
    model,
    audio_buffer: bytearray,
    buffer_lock: threading.Lock,
    client_done: threading.Event,
) -> None:
    """Transcription loop: periodically transcribe accumulated audio.

    When the audio window exceeds MAX_WINDOW_SECONDS, finalizes completed
    segments and trims the buffer to keep transcription fast.
    """
    finalized_text = ""
    last_partial_text = ""
    bytes_per_sample = 2  # int16
    bytes_per_second = SAMPLE_RATE * bytes_per_sample

    while not client_done.is_set():
        client_done.wait(timeout=TRANSCRIBE_INTERVAL)

        with buffer_lock:
            if not audio_buffer:
                continue
            snapshot = bytes(audio_buffer)

        full_text = _transcribe_audio(model, snapshot)
        if not full_text:
            continue

        display_text = (finalized_text + full_text).strip()
        if display_text == last_partial_text:
            continue
        last_partial_text = display_text
        _send_message(connection, "partial", display_text)

        window_seconds = len(snapshot) / bytes_per_second
        if window_seconds > MAX_WINDOW_SECONDS:
            finalized_text, bytes_trimmed = _finalize_segments(
                model, snapshot, finalized_text, bytes_per_second
            )
            if bytes_trimmed > 0:
                with buffer_lock:
                    del audio_buffer[:bytes_trimmed]

    # Final transcription of remaining audio
    with buffer_lock:
        snapshot = bytes(audio_buffer)

    if snapshot:
        final_text = _transcribe_audio(model, snapshot)
        if final_text:
            text = (finalized_text + final_text).strip()
            _send_message(connection, "final", text)

    _send_message(connection, "end", "")


def _finalize_segments(model, snapshot, finalized_text, bytes_per_second):
    """Finalize completed segments and return trim info.

    Transcribes with segment timestamps, finalizes all but the last segment
    (which may be incomplete), and returns how many bytes to trim from the
    front of the audio buffer.
    """
    segments = _transcribe_segments(model, snapshot)
    if len(segments) <= 1:
        return finalized_text, 0

    for seg in segments[:-1]:
        finalized_text += seg["text"]

    last_start = segments[-1]["start"]
    bytes_trimmed = int(last_start * bytes_per_second)
    bytes_trimmed -= bytes_trimmed % 2  # align to int16 boundary
    return finalized_text, bytes_trimmed


def _pcm_to_float32(audio_bytes: bytes) -> np.ndarray:
    """Convert raw PCM int16 bytes to float32 array normalized to [-1, 1]."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribe_audio(model, audio_bytes: bytes) -> str:
    """Transcribe raw PCM int16 bytes, returning the full text."""
    audio = _pcm_to_float32(audio_bytes)
    segments, _ = model.transcribe(audio, language="en", beam_size=1)
    return "".join(seg.text for seg in segments)


def _transcribe_segments(model, audio_bytes: bytes) -> list:
    """Transcribe and return segment dicts with text, start, end times."""
    audio = _pcm_to_float32(audio_bytes)
    segments, _ = model.transcribe(audio, language="en", beam_size=1)
    return [{"text": seg.text, "start": seg.start, "end": seg.end} for seg in segments]


def _send_message(connection: socket.socket, msg_type: str, text: str) -> None:
    """Send a newline-delimited JSON message to the client."""
    msg = {"type": msg_type, "text": text}
    connection.sendall(json.dumps(msg).encode("utf-8") + b"\n")


def main() -> None:
    """Main daemon entry point - load model and listen for connections."""
    setup_daemon_logging(LIVE_DAEMON_LOG)
    model = _load_whisper_model()

    sock = create_daemon_socket(LIVE_SOCKET_PATH)
    logging.info(f"Live daemon ready on {LIVE_SOCKET_PATH}")

    try:
        _accept_connections(sock, model)
    except KeyboardInterrupt:
        logging.info("Daemon shutting down.")
    finally:
        sock.close()
        cleanup_socket(LIVE_SOCKET_PATH)


def _accept_connections(sock: socket.socket, model) -> None:
    """Accept client connections in a loop."""
    while True:
        connection = None
        try:
            connection, _ = sock.accept()
            logging.info("Client connected.")
            handle_client(connection, model)
            logging.info("Client session ended.")
        except OSError as e:
            logging.error(f"Socket error: {e}")
        finally:
            if connection:
                try:
                    connection.close()
                except OSError:
                    pass


if __name__ == "__main__":
    main()
