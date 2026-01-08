"""
Dictation daemon - keeps Whisper model loaded in memory for fast transcription.
Run this once on system startup to keep model ready.
"""
import os
import sys
import socket
import pickle
import logging
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel

from .config import SOCKET_PATH, SAMPLE_RATE, DAEMON_LOG


def setup_logging() -> None:
    """Configure logging for daemon."""
    # Ensure log directory exists
    log_path = Path(DAEMON_LOG)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(DAEMON_LOG),
            logging.StreamHandler(sys.stderr)
        ]
    )


def main() -> None:
    """Main daemon entry point."""
    setup_logging()

    logging.info("Loading Whisper model...")
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )
    logging.info(f"Model loaded! Listening on {SOCKET_PATH}")

    # Remove existing socket if present
    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass

    # Create Unix domain socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCKET_PATH)
    sock.listen(1)

    logging.info("Ready for dictation requests!")

    while True:
        try:
            connection, _ = sock.accept()

            # Receive audio data
            data = b""
            while True:
                chunk = connection.recv(4096)
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
            connection.sendall(b'__END__\n')
            connection.close()

        except Exception as e:
            logging.error(f"Error processing transcription: {e}")
            continue


if __name__ == "__main__":
    main()
