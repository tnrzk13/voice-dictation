"""Live dictation daemon - keeps Whisper model loaded for streaming transcription.

Receives raw PCM int16 audio frames over a persistent socket connection,
accumulates them into an audio buffer, and periodically transcribes with
faster-whisper. Streams back partial/final results as newline-delimited JSON.

Architecture:
  - Receiver thread: reads raw PCM bytes from socket, appends to shared buffer
  - Transcriber thread: transcribes once enough new audio accumulates, with a
    timer floor so sparse audio still gets cycles

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
    KEEP_TAIL_SECONDS,
    MAX_BUFFER_SECONDS,
    SOCKET_PATH,
    SOCKET_TIMEOUT,
    TRANSCRIBE_INTERVAL,
    TRANSCRIBE_MIN_AUDIO_SECONDS,
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


class _AudioBuffer:
    """Shared PCM buffer coordinating the receiver and transcriber threads.

    Counts newly received audio separately from the retained context tail so
    the transcriber wakes on un-transcribed audio rather than raw buffer
    length, which always includes context kept for re-decoding.
    """

    def __init__(self) -> None:
        self._data = bytearray()
        self._condition = threading.Condition()
        self._pending_bytes = 0
        self._finished = False

    def append(self, data: bytes, max_bytes: int) -> None:
        with self._condition:
            self._data.extend(data)
            self._pending_bytes += len(data)
            if len(self._data) > max_bytes:
                _trim_oldest_audio(self._data, max_bytes)
            self._condition.notify()

    def finish(self) -> None:
        with self._condition:
            self._finished = True
            self._condition.notify_all()

    def wait_for_audio(self, watermark_bytes: int, timeout: float) -> bool:
        """Block until enough new audio arrives, the client finishes, or timeout."""
        with self._condition:
            return self._condition.wait_for(
                lambda: self._finished or self._pending_bytes >= watermark_bytes,
                timeout=timeout,
            )

    def take_snapshot(self) -> bytes:
        """Return buffered audio, resetting the new-audio counter."""
        with self._condition:
            self._pending_bytes = 0
            return bytes(self._data)

    def is_finished(self) -> bool:
        with self._condition:
            return self._finished

    def trim(self, num_bytes: int) -> None:
        if num_bytes <= 0:
            return
        with self._condition:
            del self._data[:num_bytes]


def handle_client(connection: socket.socket, model) -> None:
    """Process a single client's streaming audio session.

    Spawns a receiver thread to collect audio and runs the transcription
    loop on the current thread.
    """
    audio = _AudioBuffer()

    receiver = threading.Thread(
        target=_receive_audio,
        args=(connection, audio),
        daemon=True,
    )
    receiver.start()

    try:
        _transcribe_loop(connection, model, audio)
    except (ConnectionResetError, BrokenPipeError) as e:
        logging.warning(f"Client disconnected: {e}")
    except Exception as e:
        logging.error(f"Error handling client: {e}", exc_info=True)


def _receive_audio(connection: socket.socket, audio: _AudioBuffer) -> None:
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
            audio.append(data, max_buffer_bytes)
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        audio.finish()


def _trim_oldest_audio(audio_buffer: bytearray, max_bytes: int) -> None:
    """Trim audio from the front of the buffer to keep it under max_bytes."""
    excess = len(audio_buffer) - max_bytes
    if excess <= 0:
        return
    trim = excess - excess % BYTES_PER_SAMPLE
    del audio_buffer[:trim]
    logging.warning(f"Audio buffer overflow: dropped {trim} bytes of old audio")


def _transcribe_loop(connection: socket.socket, model, audio: _AudioBuffer) -> None:
    """Transcription loop: transcribe as soon as enough audio accumulates.

    Wakes on TRANSCRIBE_MIN_AUDIO_SECONDS of new audio, falling back to
    TRANSCRIBE_INTERVAL so sparse or silent audio still gets a cycle. Each
    cycle finalizes completed segments (and the stable prefix of a
    continuous segment), trimming the buffer so only the in-progress tail is
    re-transcribed with full context.
    """
    finalized_text = ""
    last_partial_text = ""
    watermark_bytes = int(TRANSCRIBE_MIN_AUDIO_SECONDS * BYTES_PER_SECOND)

    while not audio.is_finished():
        audio.wait_for_audio(watermark_bytes, TRANSCRIBE_INTERVAL)

        snapshot = audio.take_snapshot()
        if not snapshot:
            continue

        segments = _transcribe(model, snapshot, initial_prompt=finalized_text)
        full_text = "".join(seg["text"] for seg in segments)
        if not full_text:
            continue

        display_text = _concat_transcriptions(finalized_text, full_text)
        if display_text != last_partial_text:
            last_partial_text = display_text
            _send_message(connection, "partial", display_text, finalized=finalized_text)

        finalized_text, bytes_trimmed = _finalize_completed_segments(
            segments, finalized_text
        )
        audio.trim(bytes_trimmed)

    # Use last partial as the final when available - avoids re-running
    # Whisper inference which adds 2-5s latency on CPU. Fall back to
    # re-transcription only for sessions too short to produce a partial.
    if last_partial_text:
        _send_message(connection, "final", last_partial_text)
    else:
        snapshot = audio.take_snapshot()
        if snapshot:
            segments = _transcribe(model, snapshot, initial_prompt=finalized_text)
            final_text = "".join(seg["text"] for seg in segments)
            if final_text:
                text = _concat_transcriptions(finalized_text, final_text)
                _send_message(connection, "final", text)

    _send_message(connection, "end", "")


def _finalize_completed_segments(segments, finalized_text):
    """Finalize segments and return how many bytes to trim from the buffer.

    Multi-segment: finalizes all but the last (in-progress) segment, trims
    the completed portion. Single segment (continuous speech): finalizes all
    but a KEEP_TAIL_SECONDS tail and trims to that word boundary, so the
    next cycle re-transcribes with leading context instead of from a cold
    start mid-sentence.
    """
    if len(segments) <= 1:
        return _finalize_single_segment(segments, finalized_text)

    last_start = segments[-1]["start"]
    trim_bytes = int(last_start * BYTES_PER_SECOND)
    trim_bytes -= trim_bytes % BYTES_PER_SAMPLE
    if trim_bytes <= 0:
        return finalized_text, 0

    for seg in segments[:-1]:
        finalized_text = _concat_transcriptions(finalized_text, seg["text"])

    return finalized_text, trim_bytes


def _finalize_single_segment(segments, finalized_text):
    """Finalize a single continuous segment while keeping a context tail.

    Continuous speech yields one segment with no silence boundary to finalize
    against. Finalize all but the last KEEP_TAIL_SECONDS of speech and trim the
    buffer to the start of the first kept word. Word timestamps give the real
    boundary, so pauses and speaking rate cannot skew the split. Defers when
    timings are unavailable or misaligned (e.g. after repetition collapse).
    Returns (finalized_text, trim_bytes).
    """
    if not segments:
        return finalized_text, 0

    seg = segments[0]
    words = seg["text"].split()
    word_spans = seg.get("words")
    if len(words) < 2 or not word_spans or len(word_spans) != len(words):
        return finalized_text, 0

    finalized_count = _count_finalizable_words(word_spans, seg["end"])
    if finalized_count <= 0:
        return finalized_text, 0

    trim_seconds = word_spans[finalized_count]["start"]
    trim_bytes = int(trim_seconds * BYTES_PER_SECOND)
    trim_bytes -= trim_bytes % BYTES_PER_SAMPLE
    if trim_bytes <= 0:
        return finalized_text, 0

    finalized_text = _concat_transcriptions(
        finalized_text, " ".join(words[:finalized_count])
    )
    logging.info(
        f"Finalized {finalized_count}/{len(words)} words of single segment "
        f"(kept {len(words) - finalized_count} words for context)"
    )
    return finalized_text, trim_bytes


def _count_finalizable_words(word_spans, seg_end):
    """Count leading words removable while keeping KEEP_TAIL_SECONDS of speech.

    Words before the returned index are finalizable; the word at that index and
    after stay in the buffer as context. Zero means the segment fits in the tail.
    """
    finalized_count = 0
    for i, span in enumerate(word_spans):
        if seg_end - span["start"] >= KEEP_TAIL_SECONDS:
            finalized_count = i
        else:
            break
    return finalized_count


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


def _collapse_repetitions(text: str) -> str:
    """Collapse immediately repeated phrases Whisper occasionally hallucinates.

    Only 3+ word phrases collapse - shorter repeats like "no no" are real
    speech. Preserves a leading space because segment texts are joined
    without separators.
    """
    leading = " " if text.startswith(" ") else ""
    words = text.split()
    n = len(words)
    for length in range(n // 2, 2, -1):
        for i in range(n - 2 * length + 1):
            first = words[i : i + length]
            second = words[i + length : i + 2 * length]
            if [w.lower() for w in first] == [w.lower() for w in second]:
                collapsed = words[: i + length] + words[i + 2 * length :]
                return _collapse_repetitions(leading + " ".join(collapsed))
    return leading + " ".join(words)


def _transcribe(model, audio_bytes: bytes, initial_prompt: str = "") -> list:
    """Transcribe raw PCM int16 bytes, returning segment dicts.

    ``initial_prompt`` primes the decoder with text finalized before the
    trimmed buffer, so re-decoding a mid-sentence tail keeps its context
    instead of inventing or dropping boundary words.

    Each segment has 'text', 'start', 'end', and 'words' (per-word start/end
    spans) keys. Callers that only need the full text can join segment texts.
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
        word_timestamps=True,
        initial_prompt=initial_prompt or None,
    )
    return [
        {
            "text": _collapse_repetitions(seg.text),
            "start": seg.start,
            "end": seg.end,
            "words": [
                {"start": word.start, "end": word.end} for word in (seg.words or [])
            ],
        }
        for seg in segments
    ]


def _send_message(
    connection: socket.socket, msg_type: str, text: str, finalized: str = None
) -> None:
    """Send a newline-delimited JSON message to the client.

    ``finalized`` is set on partials so the client knows which prefix the
    daemon has committed and must not revise.
    """
    msg = {"type": msg_type, "text": text}
    if finalized is not None:
        msg["finalized"] = finalized
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
