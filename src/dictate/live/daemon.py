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
import signal
import socket
import threading

import numpy as np

from dictate.config import (
    AUDIO_RECV_BUFFER_BYTES,
    BYTES_PER_SAMPLE,
    BYTES_PER_SECOND,
    DAEMON_LOG,
    MAX_BUFFER_SECONDS,
    MAX_WINDOW_SECONDS,
    SOCKET_PATH,
    SOCKET_TIMEOUT,
    TRANSCRIBE_INTERVAL,
    WHISPER_BEAM_SIZE,
    WHISPER_HOTWORDS,
    WHISPER_NO_REPEAT_NGRAM_SIZE,
    WHISPER_REPETITION_PENALTY,
    WHISPER_TEMPERATURE,
    WHISPER_VAD_FILTER,
    WHISPER_VAD_MIN_SILENCE_MS,
)
from dictate.daemon_support import (
    cleanup_socket,
    create_daemon_socket,
    setup_daemon_logging,
    write_daemon_config,
)
from dictate.model_loader import add_model_args, load_whisper_model


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
    """Receiver thread: read raw PCM bytes from socket into shared buffer.

    Drops the oldest audio if the buffer grows beyond MAX_BUFFER_SECONDS,
    preventing unbounded memory use when the model is slower than real-time.
    """
    connection.settimeout(SOCKET_TIMEOUT)
    max_buffer_bytes = MAX_BUFFER_SECONDS * BYTES_PER_SECOND

    try:
        while True:
            try:
                data = connection.recv(AUDIO_RECV_BUFFER_BYTES)
            except socket.timeout:
                continue
            if not data:
                break
            with buffer_lock:
                audio_buffer.extend(data)
                if len(audio_buffer) > max_buffer_bytes:
                    _trim_oldest_audio(audio_buffer, max_buffer_bytes)
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        client_done.set()


def _trim_oldest_audio(audio_buffer: bytearray, max_bytes: int) -> None:
    """Trim audio from the front of the buffer to keep it under max_bytes."""
    excess = len(audio_buffer) - max_bytes
    if excess <= 0:
        return
    trim = excess - excess % BYTES_PER_SAMPLE
    del audio_buffer[:trim]
    logging.warning(f"Audio buffer overflow: dropped {trim} bytes of old audio")


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
        repetition_penalty=WHISPER_REPETITION_PENALTY,
        no_repeat_ngram_size=WHISPER_NO_REPEAT_NGRAM_SIZE,
    )
    return [{"text": seg.text, "start": seg.start, "end": seg.end} for seg in segments]


def _send_message(connection: socket.socket, msg_type: str, text: str) -> None:
    """Send a newline-delimited JSON message to the client."""
    msg = {"type": msg_type, "text": text}
    connection.sendall(json.dumps(msg).encode("utf-8") + b"\n")


def _parse_args() -> argparse.Namespace:
    """Parse daemon command-line arguments."""
    parser = argparse.ArgumentParser(description="Voice dictation daemon")
    add_model_args(parser)
    return parser.parse_args()


def _install_signal_handlers(sock_ref: list) -> None:
    """Close the listening socket on SIGTERM/SIGINT so accept() exits cleanly."""
    def _on_signal(signum: int, _frame) -> None:
        logging.info(f"Received signal {signum}, shutting down.")
        sock = sock_ref[0]
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)


def main() -> None:
    """Main daemon entry point - load model and listen for connections."""
    setup_daemon_logging(DAEMON_LOG)
    args = _parse_args()
    write_daemon_config(
        SOCKET_PATH,
        {
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "quiet": args.quiet,
        },
    )
    model = load_whisper_model(args.model, args.device, args.compute_type, args.quiet)

    sock = create_daemon_socket(SOCKET_PATH)
    logging.info(f"Daemon ready on {SOCKET_PATH}")

    sock_ref = [sock]
    _install_signal_handlers(sock_ref)

    try:
        _accept_connections(sock, model)
    finally:
        sock.close()
        cleanup_socket(SOCKET_PATH)


def _accept_connections(sock: socket.socket, model) -> None:
    """Accept client connections and handle each session in its own thread."""
    while True:
        try:
            connection, _ = sock.accept()
        except OSError:
            break
        threading.Thread(
            target=_handle_client_session,
            args=(connection, model),
            daemon=True,
        ).start()


def _handle_client_session(connection: socket.socket, model) -> None:
    """Run a single client session and ensure the connection is closed."""
    try:
        logging.info("Client connected.")
        handle_client(connection, model)
        logging.info("Client session ended.")
    except Exception as e:
        logging.error(f"Error handling client: {e}", exc_info=True)
    finally:
        try:
            connection.close()
        except OSError:
            pass


if __name__ == "__main__":
    main()
