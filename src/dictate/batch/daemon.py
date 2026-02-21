"""
Dictation daemon - keeps Whisper model loaded in memory for fast transcription.
Run this once on system startup to keep model ready.
"""
import pickle
import logging

import numpy as np
from faster_whisper import WhisperModel

from dictate.config import SOCKET_PATH, SAMPLE_RATE, DAEMON_LOG, END_MARKER, DAEMON_RECV_BUFFER_SIZE
from dictate.daemon_support import setup_daemon_logging, create_daemon_socket


def main() -> None:
    """Main daemon entry point."""
    setup_daemon_logging(DAEMON_LOG)

    logging.info("Loading Whisper model...")
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )
    logging.info(f"Model loaded! Listening on {SOCKET_PATH}")

    sock = create_daemon_socket(SOCKET_PATH)
    logging.info("Ready for dictation requests!")

    while True:
        connection = None
        try:
            connection, _ = sock.accept()

            # Receive audio data
            data = b""
            while True:
                chunk = connection.recv(DAEMON_RECV_BUFFER_SIZE)
                if not chunk:
                    break
                data += chunk

            # Deserialize audio
            audio = pickle.loads(data)

            # Transcribe and stream each segment as it's ready
            segments, _ = model.transcribe(audio.flatten(), language="en")
            for segment in segments:
                text = segment.text.strip()
                if text:
                    # Send segment with delimiter
                    connection.sendall(text.encode('utf-8') + b'\n')

            # Send end marker
            connection.sendall((END_MARKER + '\n').encode('utf-8'))

        except pickle.UnpicklingError as e:
            logging.error(f"Failed to deserialize audio data: {e}")
        except (ConnectionResetError, BrokenPipeError) as e:
            logging.warning(f"Client disconnected unexpectedly: {e}")
        except OSError as e:
            logging.error(f"Socket error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error processing transcription: {e}", exc_info=True)
        finally:
            if connection:
                try:
                    connection.close()
                except OSError:
                    pass  # Socket already closed


if __name__ == "__main__":
    main()
