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

import argparse
import json
import logging
import os
import socket
import sys
import threading

import numpy as np

from dictate.config import (
    BYTES_PER_SAMPLE,
    BYTES_PER_SECOND,
    DAEMON_LOG,
    MAX_WINDOW_SECONDS,
    SOCKET_PATH,
    TRANSCRIBE_INTERVAL,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_HOTWORDS,
    WHISPER_MODEL_SIZE,
    WHISPER_TEMPERATURE,
    WHISPER_VAD_FILTER,
    WHISPER_VAD_MIN_SILENCE_MS,
)
from dictate.daemon_support import (
    cleanup_socket,
    create_daemon_socket,
    setup_daemon_logging,
)


def _configure_cuda_paths():
    """Preload pip-installed NVIDIA libraries so CTranslate2 can find them."""
    try:
        import ctypes

        import nvidia.cublas
        import nvidia.cudnn

        cudnn_dir = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib")
        cublas_dir = os.path.join(os.path.dirname(nvidia.cublas.__file__), "lib")

        for lib_dir, pattern in [
            (cublas_dir, "libcublas.so"),
            (cudnn_dir, "libcudnn_ops.so"),
            (cudnn_dir, "libcudnn_cnn.so"),
            (cudnn_dir, "libcudnn.so"),
        ]:
            for f in sorted(os.listdir(lib_dir)):
                if f.startswith(pattern.replace(".so", "")) and ".so" in f:
                    ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
                    break
    except (ImportError, OSError) as e:
        logging.debug(f"CUDA library preload skipped: {e}")


def _is_model_cached(model_size):
    """Check if the model is already downloaded."""
    try:
        from faster_whisper.utils import download_model

        download_model(model_size, local_files_only=True)
        return True
    except Exception:
        return False


def _download_model_with_progress(model_size, quiet):
    """Download the model with optional desktop notification progress."""
    from dictate.system import notify
    from faster_whisper.utils import _MODELS

    import huggingface_hub
    from tqdm import tqdm

    repo_id = _MODELS.get(model_size, model_size)
    allow_patterns = [
        "config.json",
        "preprocessor_config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]

    notify_fn = notify if not quiet else lambda msg: None
    last_milestone = [0]

    class ProgressTqdm(tqdm):
        def update(self, n=1):
            super().update(n)
            if not self.total or "Fetching" in (self.desc or ""):
                return
            percent = int(self.n / self.total * 100)
            milestone = percent // 10 * 10
            if milestone > last_milestone[0]:
                last_milestone[0] = milestone
                notify_fn(f"Downloading {model_size}: {milestone}%")

    logging.info(f"Downloading model {model_size} ({repo_id})...")
    notify_fn(f"Downloading {model_size}...")

    huggingface_hub.snapshot_download(
        repo_id,
        allow_patterns=allow_patterns,
        tqdm_class=ProgressTqdm,
    )

    logging.info("Download complete.")
    notify_fn(f"Download complete - loading {model_size}")


def _load_whisper_model(model_size, device, compute_type, quiet=False):
    """Load faster-whisper model, downloading with progress if needed."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logging.error(
            "faster-whisper package not installed. "
            "Install with: pip install faster-whisper>=0.10.0"
        )
        sys.exit(1)

    if not _is_model_cached(model_size):
        _download_model_with_progress(model_size, quiet)

    _configure_cuda_paths()

    logging.info(f"Loading Whisper model ({model_size})...")
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
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

    while not client_done.is_set():
        client_done.wait(timeout=TRANSCRIBE_INTERVAL)

        with buffer_lock:
            if not audio_buffer:
                continue
            snapshot = bytes(audio_buffer)

        segments = _transcribe(model, snapshot)
        full_text = "".join(seg["text"] for seg in segments)
        if not full_text:
            continue

        display_text = _concat_transcriptions(finalized_text, full_text)
        if display_text == last_partial_text:
            continue
        last_partial_text = display_text
        _send_message(connection, "partial", display_text)

        window_seconds = len(snapshot) / BYTES_PER_SECOND
        if window_seconds > MAX_WINDOW_SECONDS:
            finalized_text, bytes_trimmed = _finalize_completed_segments(
                segments, finalized_text, len(snapshot)
            )
            with buffer_lock:
                del audio_buffer[:bytes_trimmed]

    # Use last partial as the final when available - avoids re-running
    # Whisper inference which adds 2-5s latency on CPU. Fall back to
    # re-transcription only for sessions too short to produce a partial.
    if last_partial_text:
        _send_message(connection, "final", last_partial_text)
    else:
        with buffer_lock:
            snapshot = bytes(audio_buffer)
        if snapshot:
            segments = _transcribe(model, snapshot)
            final_text = "".join(seg["text"] for seg in segments)
            if final_text:
                text = _concat_transcriptions(finalized_text, final_text)
                _send_message(connection, "final", text)

    _send_message(connection, "end", "")


def _finalize_completed_segments(segments, finalized_text, snapshot_bytes):
    """Finalize segments and return how many bytes to trim from the buffer.

    Multi-segment: finalizes all but the last (in-progress) segment, trims
    the completed portion. Single segment: force-finalizes everything to
    cap buffer growth - audio arriving next cycle provides natural context.
    """
    if len(segments) <= 1:
        for seg in segments:
            finalized_text = _concat_transcriptions(finalized_text, seg["text"])
        trim_bytes = snapshot_bytes - (snapshot_bytes % BYTES_PER_SAMPLE)
        logging.info("Force-trimmed buffer (single segment)")
        return finalized_text, trim_bytes

    for seg in segments[:-1]:
        finalized_text = _concat_transcriptions(finalized_text, seg["text"])

    last_start = segments[-1]["start"]
    trim_bytes = int(last_start * BYTES_PER_SECOND)
    trim_bytes -= trim_bytes % BYTES_PER_SAMPLE
    return finalized_text, trim_bytes


def _concat_transcriptions(finalized: str, new: str) -> str:
    """Join finalized and new transcription text, ensuring word separation.

    Whisper's first segment in a transcription has no leading space, so after
    buffer trimming and re-transcription, the new text may lack a separator.
    """
    finalized = finalized.strip()
    new = new.strip()
    if not finalized:
        return new
    if not new:
        return finalized
    return finalized + " " + new


def _pcm_to_float32(audio_bytes: bytes) -> np.ndarray:
    """Convert raw PCM int16 bytes to float32 array normalized to [-1, 1]."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribe(model, audio_bytes: bytes) -> list:
    """Transcribe raw PCM int16 bytes, returning segment dicts.

    Each segment has 'text', 'start', and 'end' keys. Callers that only
    need the full text can join segment texts.
    """
    audio = _pcm_to_float32(audio_bytes)
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=WHISPER_BEAM_SIZE,
        temperature=WHISPER_TEMPERATURE,
        vad_filter=WHISPER_VAD_FILTER,
        vad_parameters=dict(min_silence_duration_ms=WHISPER_VAD_MIN_SILENCE_MS),
        hotwords=WHISPER_HOTWORDS,
    )
    return [{"text": seg.text, "start": seg.start, "end": seg.end} for seg in segments]


def _send_message(connection: socket.socket, msg_type: str, text: str) -> None:
    """Send a newline-delimited JSON message to the client."""
    msg = {"type": msg_type, "text": text}
    connection.sendall(json.dumps(msg).encode("utf-8") + b"\n")


def _parse_args() -> argparse.Namespace:
    """Parse daemon command-line arguments."""
    parser = argparse.ArgumentParser(description="Voice dictation daemon")
    parser.add_argument(
        "--model",
        default=WHISPER_MODEL_SIZE,
        help=f"Whisper model size (default: {WHISPER_MODEL_SIZE})",
    )
    parser.add_argument(
        "--device",
        default=WHISPER_DEVICE,
        help=f"Compute device: cuda or cpu (default: {WHISPER_DEVICE})",
    )
    parser.add_argument(
        "--compute-type",
        default=WHISPER_COMPUTE_TYPE,
        help=f"Model precision: float16, int8, etc. (default: {WHISPER_COMPUTE_TYPE})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress download progress notifications",
    )
    return parser.parse_args()


def main() -> None:
    """Main daemon entry point - load model and listen for connections."""
    setup_daemon_logging(DAEMON_LOG)
    args = _parse_args()
    model = _load_whisper_model(args.model, args.device, args.compute_type, args.quiet)

    sock = create_daemon_socket(SOCKET_PATH)
    logging.info(f"Daemon ready on {SOCKET_PATH}")

    try:
        _accept_connections(sock, model)
    except KeyboardInterrupt:
        logging.info("Daemon shutting down.")
    finally:
        sock.close()
        cleanup_socket(SOCKET_PATH)


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
