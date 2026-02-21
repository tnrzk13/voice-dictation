"""Batch dictation daemon - keeps Whisper model loaded for fast transcription."""

import logging
import pickle
import sys

import numpy as np

from dictate.config import (
    DAEMON_LOG,
    DAEMON_RECV_BUFFER_SIZE,
    END_MARKER,
    SAMPLE_RATE,
    SOCKET_PATH,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)
from dictate.daemon_support import setup_daemon_logging, create_daemon_socket


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
    logging.info("Model loaded.")
    return model


def _accept_connections(sock, model) -> None:
    """Accept client connections in a loop."""
    while True:
        connection = None
        try:
            connection, _ = sock.accept()
            _handle_client(connection, model)
        except (ConnectionResetError, BrokenPipeError) as e:
            logging.warning(f"Client disconnected unexpectedly: {e}")
        except pickle.UnpicklingError as e:
            logging.error(f"Failed to deserialize audio data: {e}")
        except OSError as e:
            logging.error(f"Socket error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error processing transcription: {e}", exc_info=True)
        finally:
            if connection:
                try:
                    connection.close()
                except OSError:
                    pass


def _handle_client(connection, model) -> None:
    """Receive audio, transcribe, and stream results back to client."""
    data = b""
    while True:
        chunk = connection.recv(DAEMON_RECV_BUFFER_SIZE)
        if not chunk:
            break
        data += chunk

    audio = pickle.loads(data)

    segments, _ = model.transcribe(audio.flatten(), language="en")
    for segment in segments:
        text = segment.text.strip()
        if text:
            connection.sendall(text.encode('utf-8') + b'\n')

    connection.sendall((END_MARKER + '\n').encode('utf-8'))


def main() -> None:
    """Main daemon entry point - load model and listen for connections."""
    setup_daemon_logging(DAEMON_LOG)
    model = _load_whisper_model()

    sock = create_daemon_socket(SOCKET_PATH)
    logging.info(f"Ready for dictation requests on {SOCKET_PATH}")

    _accept_connections(sock, model)


if __name__ == "__main__":
    main()
