"""
Live dictation daemon - keeps Vosk model loaded for streaming transcription.

Receives raw PCM int16 audio frames over a persistent socket connection,
feeds them to Vosk's KaldiRecognizer, and streams back partial/final results
as newline-delimited JSON.

Protocol:
  Client sends: raw PCM int16 bytes (continuous stream)
  Client sends: EOF (shutdown write side) to signal end
  Daemon sends: {"type": "partial", "text": "..."}\n
  Daemon sends: {"type": "final", "text": "..."}\n
  Daemon sends: {"type": "end"}\n
"""

import json
import logging
import os
import socket
import sys

from dictate.daemon_support import (
    cleanup_socket,
    create_daemon_socket,
    setup_daemon_logging,
)

from dictate.punctuation import try_load_punctuation

from .config import (
    LIVE_DAEMON_LOG,
    LIVE_SOCKET_PATH,
    RECASEPUNC_MODEL_DIR,
    SAMPLE_RATE,
    VOSK_MODEL_DIR,
    VOSK_MODEL_NAME,
)


def load_vosk_model():
    """Load Vosk model, failing fast with download instructions if missing."""
    try:
        from vosk import Model, SetLogLevel
    except ImportError:
        logging.error(
            "vosk package not installed. Install with: pip install vosk>=0.3.45"
        )
        sys.exit(1)

    SetLogLevel(-1)  # Suppress Vosk's noisy internal logging

    model_path = os.path.join(VOSK_MODEL_DIR, VOSK_MODEL_NAME)
    if not os.path.isdir(model_path):
        logging.error(
            f"Vosk model not found at {model_path}\n"
            f"Download it with: scripts/download-model.sh\n"
            f"Or manually:\n"
            f"  mkdir -p {VOSK_MODEL_DIR}\n"
            f"  cd {VOSK_MODEL_DIR}\n"
            f"  wget https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip\n"
            f"  unzip {VOSK_MODEL_NAME}.zip"
        )
        sys.exit(1)

    logging.info(f"Loading Vosk model from {model_path}...")
    model = Model(model_path)
    logging.info("Vosk model loaded.")
    return model


def _create_recognizer(model):
    """Create a KaldiRecognizer from the loaded Vosk model."""
    from vosk import KaldiRecognizer

    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(False)
    return recognizer


def handle_client(connection: socket.socket, model, punctuator=None, recognizer=None) -> None:
    """Process a single client's streaming audio session."""
    if recognizer is None:
        recognizer = _create_recognizer(model)

    try:
        while True:
            data = connection.recv(8000)  # ~250ms of int16 audio at 16kHz
            if not data:
                break

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    text = _apply_punctuation(punctuator, text)
                    _send_message(connection, "final", text)
            else:
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "").strip()
                if text:
                    _send_message(connection, "partial", text)

        # Flush any remaining audio
        result = json.loads(recognizer.FinalResult())
        text = result.get("text", "").strip()
        if text:
            text = _apply_punctuation(punctuator, text)
            _send_message(connection, "final", text)

        _send_message(connection, "end", "")

    except (ConnectionResetError, BrokenPipeError) as e:
        logging.warning(f"Client disconnected: {e}")
    except Exception as e:
        logging.error(f"Error handling client: {e}", exc_info=True)


def _apply_punctuation(punctuator, text: str) -> str:
    """Apply punctuation restoration if available, falling back to raw text."""
    if punctuator is None:
        return text
    try:
        return punctuator.restore(text)
    except Exception as e:
        logging.warning(f"Punctuation restoration failed: {e}")
        return text


def _send_message(connection: socket.socket, msg_type: str, text: str) -> None:
    """Send a newline-delimited JSON message to the client."""
    msg = {"type": msg_type, "text": text}
    connection.sendall(json.dumps(msg).encode("utf-8") + b"\n")


def main() -> None:
    """Main daemon entry point - load model and listen for connections."""
    setup_daemon_logging(LIVE_DAEMON_LOG)
    model = load_vosk_model()
    punctuator = try_load_punctuation(RECASEPUNC_MODEL_DIR)

    sock = create_daemon_socket(LIVE_SOCKET_PATH)
    logging.info(f"Live daemon ready on {LIVE_SOCKET_PATH}")

    try:
        _accept_connections(sock, model, punctuator)
    except KeyboardInterrupt:
        logging.info("Daemon shutting down.")
    finally:
        sock.close()
        cleanup_socket(LIVE_SOCKET_PATH)


def _accept_connections(sock: socket.socket, model, punctuator=None) -> None:
    """Accept client connections in a loop."""
    while True:
        connection = None
        try:
            connection, _ = sock.accept()
            logging.info("Client connected.")
            handle_client(connection, model, punctuator)
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
