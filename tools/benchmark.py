#!/usr/bin/env python3
"""Measure the live dictation pipeline on the repo's audio fixtures.

Each fixture's PCM is streamed through the real daemon path: a socketpair, the
daemon's own handle_client, the 0.3s new-audio gate, and real-time pacing. The
harness instruments that path to record decode time, partial latency,
correction behaviour, and GPU memory. Product code is not modified.

Run from the repo root:

    python3 tools/benchmark.py --runs 2

Results are written to tools/benchmark-results.json by default, plus a summary
table on stdout.
"""

import argparse
import contextlib
import csv
import json
import math
import platform
import re
import socket
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for path in (SRC_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dictate.config import (  # noqa: E402
    BYTES_PER_SAMPLE,
    BYTES_PER_SECOND,
    KEEP_TAIL_SECONDS,
    MAX_SESSION_SECONDS,
    SAMPLE_RATE,
    TRANSCRIBE_INTERVAL,
    TRANSCRIBE_MIN_AUDIO_SECONDS,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_HOTWORD_TERMS,
    WHISPER_MODEL_SIZE,
    WHISPER_NO_REPEAT_NGRAM_SIZE,
    WHISPER_REPETITION_PENALTY,
    WHISPER_TEMPERATURE,
    WHISPER_VAD_FILTER,
    WHISPER_VAD_MIN_SILENCE_MS,
)
from dictate.live import daemon as live_daemon  # noqa: E402
from dictate.live import typer as live_typer  # noqa: E402
from dictate.live.typer import ProgressiveTyper  # noqa: E402
from dictate.model_loader import load_whisper_model  # noqa: E402
from tools.fixture_definitions import FIXTURES as SCRIPT_DEFINITIONS  # noqa: E402

SCHEMA_VERSION = 1
FIXTURES_DIR = REPO_ROOT / "tests" / "audio_fixtures"
LOCAL_FIXTURES_DIR = REPO_ROOT / "tests" / "audio_fixtures_local"
DEFAULT_OUTPUT = REPO_ROOT / "tools" / "benchmark-results.json"

# 1024 frames at 16 kHz, the usual PortAudio callback size.
FEED_FRAME_SECONDS = 0.064
FEED_FRAME_BYTES = int(FEED_FRAME_SECONDS * BYTES_PER_SECOND)
VRAM_SAMPLE_INTERVAL_SECONDS = 0.1
VRAM_BASELINE_SECONDS = 0.5
VRAM_SETTLE_SECONDS = 0.3
REPLAY_TIMEOUT_SECONDS = 180.0

# Committed fixture directories predate the shared fixture definitions.
COMMITTED_SCRIPT_NAMES = {
    "hello_world": "simple_sentence",
    "formatting_commands": "formatting_commands",
    "pause_and_continue": "mid_sentence_pause",
}

# The stutter fixture's spoken text includes words absent from its script, so
# its script is not valid ground truth for WER.
SCRIPT_WER_EXCLUSIONS = {"repetition_recovery"}

TOKEN_PATTERN = re.compile(r"[^\w']+", re.UNICODE)
VRAM_QUERY = [
    "nvidia-smi",
    "--query-gpu=memory.used",
    "--format=csv,noheader,nounits",
]
GPU_INFO_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,driver_version,memory.total",
    "--format=csv,noheader,nounits",
]


@dataclass
class Fixture:
    """One audio file and the ground-truth material available for it."""

    name: str
    source: str
    wav_path: Path
    reference_path: Optional[Path]
    script: Optional[str]
    chunks_path: Optional[Path]
    audio_bytes: bytes = b""
    audio_seconds: float = 0.0
    wav_bytes: int = 0
    golden_message_count: Optional[int] = None


class SessionRecorder:
    """Collect timing events from the daemon's receiver and transcriber threads."""

    def __init__(self, watermark_bytes: int) -> None:
        self._watermark_bytes = watermark_bytes
        self._lock = threading.Lock()
        self._pending_trigger_at: Optional[float] = None
        self._dropped_bytes = 0
        self.cycles: List[dict] = []
        self.messages: List[dict] = []
        self.decodes: List[dict] = []
        self.commit_groups: List[dict] = []
        self.last_snapshot_start_seconds = 0.0
        self.floor_wakes = 0

    def note_append(self, pending_bytes: int, at: float) -> None:
        with self._lock:
            if self._pending_trigger_at is None and pending_bytes >= self._watermark_bytes:
                self._pending_trigger_at = at

    def note_trim(self, num_bytes: int) -> None:
        with self._lock:
            self._dropped_bytes += max(0, num_bytes)

    def note_snapshot(self, at: float) -> None:
        with self._lock:
            self.last_snapshot_start_seconds = self._dropped_bytes / BYTES_PER_SECOND
            self.cycles.append(
                {"trigger_at": self._pending_trigger_at, "snapshot_at": at, "partial_at": None}
            )
            self._pending_trigger_at = None

    def note_commit_group(self, words: Sequence[dict], at: float) -> None:
        with self._lock:
            self.commit_groups.append(
                {
                    "at": at,
                    "end_times": [word["end"] for word in words],
                    "buffer_start_seconds": self.last_snapshot_start_seconds,
                    "send_at": None,
                }
            )

    def note_floor_wake(self) -> None:
        with self._lock:
            self.floor_wakes += 1

    def note_message(
        self, msg_type: str, text: str, finalized: Optional[str], at: float
    ) -> None:
        with self._lock:
            entry = {"type": msg_type, "text": text, "at": at}
            if finalized is not None:
                entry["finalized"] = finalized
            self.messages.append(entry)
            if msg_type != "partial":
                return
            for cycle in reversed(self.cycles):
                if cycle["partial_at"] is None:
                    cycle["partial_at"] = at
                    break
            for group in self.commit_groups:
                if group["send_at"] is None:
                    group["send_at"] = at
                    break

    def note_decode(
        self,
        started_at: float,
        finished_at: float,
        audio_seconds: float,
        kind: str,
        word_timestamps: bool,
        vad_filter: bool,
    ) -> None:
        with self._lock:
            self.decodes.append(
                {
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "audio_seconds": audio_seconds,
                    "kind": kind,
                    "word_timestamps": word_timestamps,
                    "vad_filter": vad_filter,
                }
            )


class TimedModel:
    """Proxy around WhisperModel that times the real transcribe generator."""

    def __init__(self, model, recorder: SessionRecorder) -> None:
        self._model = model
        self._recorder = recorder

    def transcribe(self, audio, **kwargs):
        started_at = time.perf_counter()
        audio_seconds = len(audio) / SAMPLE_RATE
        segments, info = self._model.transcribe(audio, **kwargs)
        word_timestamps = bool(kwargs.get("word_timestamps", True))
        vad_filter = bool(kwargs.get("vad_filter", True))
        kind = "final" if not word_timestamps and not vad_filter else "streaming"

        def timed_segments():
            try:
                for segment in segments:
                    yield segment
            finally:
                self._recorder.note_decode(
                    started_at,
                    time.perf_counter(),
                    audio_seconds,
                    kind,
                    word_timestamps,
                    vad_filter,
                )

        return timed_segments(), info


def _flat_word_spans(segments: Sequence[dict]) -> List[dict]:
    spans: List[dict] = []
    for segment in segments:
        spans.extend(segment.get("words") or [])
    return spans


class _CommitInstrumentation:
    def __init__(self, recorder: SessionRecorder) -> None:
        self._recorder = recorder
        self._patched: List[Tuple[str, object]] = []

    def patch(self) -> None:
        self._patch_finalize()

    def unpatch(self) -> None:
        for name, original in self._patched:
            setattr(live_daemon, name, original)
        self._patched.clear()

    def _patch_finalize(self) -> None:
        original = getattr(live_daemon, "_finalize_completed_segments", None)
        if original is None:
            return

        def finalize(segments, finalized_text):
            result_text, trim_bytes = original(segments, finalized_text)
            committed = len(result_text.split()) - len(finalized_text.split())
            spans = _flat_word_spans(segments)
            if committed > 0 and len(spans) >= committed:
                self._recorder.note_commit_group(
                    spans[:committed], time.perf_counter()
                )
            return result_text, trim_bytes

        self._replace("_finalize_completed_segments", original, finalize)

    def _replace(self, name: str, original, replacement) -> None:
        self._patched.append((name, original))
        setattr(live_daemon, name, replacement)


def _commit_instrumentation(recorder: SessionRecorder) -> _CommitInstrumentation:
    return _CommitInstrumentation(recorder)


@contextlib.contextmanager
def instrument_daemon(recorder: SessionRecorder):
    """Wrap daemon buffer and send methods to record timing without editing them."""
    original_append = live_daemon._AudioBuffer.append
    original_snapshot = live_daemon._AudioBuffer.take_snapshot
    original_wait = live_daemon._AudioBuffer.wait_for_audio
    original_trim = live_daemon._AudioBuffer.trim
    original_overflow = live_daemon._trim_oldest_audio
    original_send = live_daemon._send_message
    commit_hook = _commit_instrumentation(recorder)

    def append(self, data, max_bytes, max_session_bytes=None):
        original_append(self, data, max_bytes, max_session_bytes)
        recorder.note_append(self._pending_bytes, time.perf_counter())

    def take_snapshot(self):
        recorder.note_snapshot(time.perf_counter())
        return original_snapshot(self)

    def trim(self, num_bytes):
        original_trim(self, num_bytes)
        recorder.note_trim(num_bytes)

    def trim_oldest(audio_buffer, max_bytes):
        before = len(audio_buffer)
        original_overflow(audio_buffer, max_bytes)
        recorder.note_trim(before - len(audio_buffer))

    def wait_for_audio(self, watermark_bytes, timeout):
        woke = original_wait(self, watermark_bytes, timeout)
        if not woke:
            recorder.note_floor_wake()
        return woke

    def send_message(connection, msg_type, text, finalized=None):
        original_send(connection, msg_type, text, finalized)
        recorder.note_message(msg_type, text, finalized, time.perf_counter())

    live_daemon._AudioBuffer.append = append
    live_daemon._AudioBuffer.take_snapshot = take_snapshot
    live_daemon._AudioBuffer.trim = trim
    live_daemon._AudioBuffer.wait_for_audio = wait_for_audio
    live_daemon._trim_oldest_audio = trim_oldest
    live_daemon._send_message = send_message
    commit_hook.patch()
    try:
        yield
    finally:
        commit_hook.unpatch()
        live_daemon._AudioBuffer.append = original_append
        live_daemon._AudioBuffer.take_snapshot = original_snapshot
        live_daemon._AudioBuffer.trim = original_trim
        live_daemon._AudioBuffer.wait_for_audio = original_wait
        live_daemon._trim_oldest_audio = original_overflow
        live_daemon._send_message = original_send


class VramSampler:
    """Poll whole-GPU memory usage in a background thread."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.samples: List[Tuple[float, int]] = []
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._query()
            if value is None:
                self.error = "nvidia-smi memory query failed"
                return
            self.samples.append((time.perf_counter(), value))
            self._stop.wait(self.interval_seconds)

    def _query(self) -> Optional[int]:
        try:
            result = subprocess.run(
                VRAM_QUERY, capture_output=True, text=True, timeout=5.0, check=False
            )
            if result.returncode != 0:
                return None
            return int(result.stdout.strip().splitlines()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None


def _stream_audio(
    sock: socket.socket, audio_bytes: bytes, frame_bytes: int
) -> Tuple[float, float, float]:
    """Send PCM at wall-clock audio speed. Returns (first send, last send, start)."""
    started_at = time.perf_counter()
    first_at: Optional[float] = None
    offset = 0
    while offset < len(audio_bytes):
        target = started_at + offset / BYTES_PER_SECOND
        delay = target - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        frame = audio_bytes[offset : offset + frame_bytes]
        sock.sendall(frame)
        if first_at is None:
            first_at = time.perf_counter()
        offset += len(frame)
    return first_at or started_at, time.perf_counter(), started_at


def _drain_messages(sock: socket.socket, done: threading.Event) -> None:
    buffer = b""
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    message = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if message.get("type") == "end":
                    done.set()
                    return
    except OSError:
        return
    finally:
        done.set()


def replay_fixture(model, fixture: Fixture) -> Tuple[SessionRecorder, float, float]:
    """Run one fixture through the daemon and return its recorded events."""
    recorder = SessionRecorder(int(TRANSCRIBE_MIN_AUDIO_SECONDS * BYTES_PER_SECOND))
    client_sock, daemon_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    done = threading.Event()
    reader = threading.Thread(
        target=_drain_messages, args=(client_sock, done), daemon=True
    )
    reader.start()

    with instrument_daemon(recorder):
        daemon_thread = threading.Thread(
            target=live_daemon.handle_client,
            args=(daemon_sock, TimedModel(model, recorder)),
            daemon=True,
        )
        daemon_thread.start()
        _, _, feed_started = _stream_audio(
            client_sock, fixture.audio_bytes, FEED_FRAME_BYTES
        )
        client_sock.shutdown(socket.SHUT_WR)
        daemon_thread.join(timeout=REPLAY_TIMEOUT_SECONDS)

    if daemon_thread.is_alive():
        client_sock.close()
        daemon_sock.close()
        raise RuntimeError(
            f"{fixture.name}: daemon did not finish within {REPLAY_TIMEOUT_SECONDS:.0f}s"
        )
    if not done.wait(timeout=REPLAY_TIMEOUT_SECONDS):
        raise RuntimeError(f"{fixture.name}: no end message reached the client")

    messages = list(recorder.messages)
    client_sock.close()
    daemon_sock.close()

    if not any(message["type"] == "end" for message in messages):
        raise RuntimeError(f"{fixture.name}: daemon sent no end message")
    end_at = messages[-1]["at"]
    return recorder, feed_started, end_at - feed_started


def _load_wav(path: Path) -> Tuple[bytes, float]:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != BYTES_PER_SAMPLE
            or handle.getframerate() != SAMPLE_RATE
        ):
            raise ValueError(
                f"{path}: expected 16 kHz mono int16, got "
                f"{handle.getframerate()} Hz, {handle.getnchannels()} channel(s), "
                f"{handle.getsampwidth()} bytes/sample"
            )
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    return raw, frames / SAMPLE_RATE


def _count_lines(path: Optional[Path]) -> Optional[int]:
    if path is None or not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _discover_fixtures(include_local: bool, filters: Sequence[str]) -> List[Fixture]:
    fixtures: List[Fixture] = []

    for wav_path in sorted(FIXTURES_DIR.glob("*.wav")):
        name = wav_path.stem
        fixture_dir = FIXTURES_DIR / name
        script_name = COMMITTED_SCRIPT_NAMES.get(name)
        script = SCRIPT_DEFINITIONS.get(script_name, {}).get("script") if script_name else None
        fixtures.append(
            Fixture(
                name=name,
                source="committed",
                wav_path=wav_path,
                reference_path=fixture_dir / "reference.txt",
                script=script,
                chunks_path=fixture_dir / "chunks.jsonl",
            )
        )

    if include_local:
        for wav_path in sorted(LOCAL_FIXTURES_DIR.glob("*/takes/*/audio.wav")):
            fixture_name = wav_path.parents[2].name
            take_name = wav_path.parent.name
            script = SCRIPT_DEFINITIONS.get(fixture_name, {}).get("script")
            fixtures.append(
                Fixture(
                    name=f"{fixture_name}/{take_name}",
                    source="local",
                    wav_path=wav_path,
                    reference_path=wav_path.parent / "reference.txt",
                    script=script,
                    chunks_path=wav_path.parent / "chunks.jsonl",
                )
            )

    if filters:
        fixtures = [
            fixture
            for fixture in fixtures
            if any(pattern in fixture.name for pattern in filters)
        ]
    return fixtures


def _load_fixture_audio(fixture: Fixture) -> None:
    fixture.audio_bytes, fixture.audio_seconds = _load_wav(fixture.wav_path)
    fixture.wav_bytes = fixture.wav_path.stat().st_size
    fixture.golden_message_count = _count_lines(fixture.chunks_path)


def normalize_tokens(text: str) -> List[str]:
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    return TOKEN_PATTERN.sub(" ", text.lower()).split()


def _edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> Dict[str, int]:
    """Levenshtein alignment counts. Ties prefer substitution, then deletion."""
    rows, columns = len(reference), len(hypothesis)
    distance = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        distance[i][0] = i
    for j in range(columns + 1):
        distance[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                distance[i][j] = 1 + min(
                    distance[i - 1][j - 1],
                    distance[i - 1][j],
                    distance[i][j - 1],
                )

    substitutions = deletions = insertions = 0
    i, j = rows, columns
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": substitutions + deletions + insertions,
    }


def _wer_block(reference: Optional[str], hypothesis: str) -> Optional[dict]:
    if reference is None:
        return None
    reference_tokens = normalize_tokens(reference)
    hypothesis_tokens = normalize_tokens(hypothesis)
    counts = _edit_counts(reference_tokens, hypothesis_tokens)
    rate = counts["errors"] / len(reference_tokens) if reference_tokens else None
    return {
        "reference_words": len(reference_tokens),
        "hypothesis_words": len(hypothesis_tokens),
        "substitutions": counts["substitutions"],
        "deletions": counts["deletions"],
        "insertions": counts["insertions"],
        "errors": counts["errors"],
        "wer": rate,
    }


def _percentile(values: Sequence[float], percent: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percent / 100.0 * len(ordered)) - 1)
    return ordered[index]


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return _percentile(values, 50)


def _latest(messages: Sequence[dict], msg_type: str) -> Optional[dict]:
    for message in reversed(messages):
        if message["type"] == msg_type:
            return message
    return None


def _decode_metrics(recorder: SessionRecorder, audio_seconds: float, session_seconds: float) -> dict:
    decodes = recorder.decodes
    streaming = [decode for decode in decodes if decode["kind"] == "streaming"]
    final = [decode for decode in decodes if decode["kind"] == "final"]
    durations = [decode["finished_at"] - decode["started_at"] for decode in decodes]
    per_call_rtf = [
        duration / decode["audio_seconds"]
        for duration, decode in zip(durations, decodes)
        if decode["audio_seconds"] > 0
    ]
    total_wall = sum(durations)
    total_decoded_audio = sum(decode["audio_seconds"] for decode in decodes)
    return {
        "decode_calls": len(decodes),
        "streaming_decode_calls": len(streaming),
        "final_decode_calls": len(final),
        "decode_wall_seconds": total_wall,
        "streaming_decode_wall_seconds": sum(
            decode["finished_at"] - decode["started_at"] for decode in streaming
        ),
        "final_decode_wall_seconds": sum(
            decode["finished_at"] - decode["started_at"] for decode in final
        ),
        "decoded_audio_seconds": total_decoded_audio,
        "decode_audio_amplification": (
            total_decoded_audio / audio_seconds if audio_seconds else None
        ),
        "rtf_total": total_wall / audio_seconds if audio_seconds else None,
        "rtf_per_decode_p50": _percentile(per_call_rtf, 50),
        "rtf_per_decode_p95": _percentile(per_call_rtf, 95),
        "rtf_per_decode_max": max(per_call_rtf) if per_call_rtf else None,
        "rtf_session": total_wall / session_seconds if session_seconds > 0 else None,
        "per_decode_rtf": per_call_rtf,
        "per_decode_seconds": durations,
        "per_decode_audio_seconds": [decode["audio_seconds"] for decode in decodes],
    }


def _latency_metrics(
    recorder: SessionRecorder, messages: Sequence[dict], session_started: float
) -> dict:
    gate_latencies = [
        cycle["partial_at"] - cycle["trigger_at"]
        for cycle in recorder.cycles
        if cycle["partial_at"] is not None and cycle["trigger_at"] is not None
    ]
    first_partial = next(
        (message for message in messages if message["type"] == "partial"), None
    )
    return {
        "gate_triggers": len(gate_latencies),
        "latency_samples": len(gate_latencies),
        "p50_seconds": _percentile(gate_latencies, 50),
        "p95_seconds": _percentile(gate_latencies, 95),
        "max_seconds": max(gate_latencies) if gate_latencies else None,
        "min_seconds": min(gate_latencies) if gate_latencies else None,
        "floor_wakes": recorder.floor_wakes,
        "time_to_first_partial_seconds": (
            first_partial["at"] - session_started if first_partial else None
        ),
        "samples_seconds": gate_latencies,
    }


def _word_commit_metrics(
    recorder: SessionRecorder, audio_started_at: float
) -> dict:
    delays: List[float] = []
    for group in recorder.commit_groups:
        if group["send_at"] is None:
            continue
        word_start = audio_started_at + group["buffer_start_seconds"]
        delays.extend(
            group["send_at"] - (word_start + end) for end in group["end_times"]
        )
    return {
        "samples": len(delays),
        "samples_seconds": delays,
        "p50_seconds": _percentile(delays, 50),
        "p95_seconds": _percentile(delays, 95),
        "max_seconds": max(delays) if delays else None,
    }


def _correction_metrics(messages: Sequence[dict], final_text: str) -> dict:
    counts = {"backspaces": 0, "backspace_calls": 0, "type_calls": 0, "typed_chars": 0}
    typer = ProgressiveTyper()
    previous_partial: Optional[str] = None
    previous_finalized: Optional[str] = None
    previous_committed = ""
    revisions = 0
    regressions: List[dict] = []
    committed_regressions = 0
    final_backspaces = 0
    final_changed = False
    visible_text = ""
    visible_backspaces = 0

    def fake_backspaces(count: int) -> None:
        counts["backspace_calls"] += 1
        counts["backspaces"] += int(count)

    def fake_type(text: str) -> None:
        counts["type_calls"] += 1
        counts["typed_chars"] += len(text)

    with mock.patch.object(live_typer, "_send_backspaces", fake_backspaces), mock.patch.object(
        live_typer, "_type_text", fake_type
    ):
        for message in messages:
            if message["type"] == "partial":
                finalized = message.get("finalized", "")
                typer.apply_partial(message["text"], finalized)
                if previous_partial is not None and message["text"] != previous_partial:
                    revisions += 1
                previous_partial = message["text"]
                if (
                    previous_finalized
                    and finalized
                    and not finalized.startswith(previous_finalized)
                ):
                    regressions.append(
                        {"previous": previous_finalized, "current": finalized}
                    )
                previous_finalized = finalized
                if not typer.committed.startswith(previous_committed):
                    committed_regressions += 1
                previous_committed = typer.committed
                visible_text = typer.displayed_text
                visible_backspaces = counts["backspaces"]
            elif message["type"] == "final":
                before = counts["backspaces"]
                typer.apply_final(message["text"])
                final_backspaces = counts["backspaces"] - before
                final_changed = normalize_tokens(message["text"]) != (
                    normalize_tokens(previous_partial) if previous_partial else []
                )
        displayed_text = typer.displayed_text
        committed_text = typer.committed

    word_count = len(normalize_tokens(final_text))
    return {
        "partial_messages": sum(1 for message in messages if message["type"] == "partial"),
        "partial_revisions": revisions,
        "backspaces_total": counts["backspaces"],
        "visible_backspaces": visible_backspaces,
        "backspace_calls": counts["backspace_calls"],
        "typed_characters": counts["typed_chars"],
        "final_backspaces": final_backspaces,
        "final_differs_from_last_partial": final_changed,
        "words_in_final_transcript": word_count,
        "backspaces_per_100_words": (
            counts["backspaces"] / word_count * 100.0 if word_count else None
        ),
        "visible_backspaces_per_100_words": (
            visible_backspaces / word_count * 100.0 if word_count else None
        ),
        "revisions_per_word": revisions / word_count if word_count else None,
        "finalized_prefix_regressions": len(regressions),
        "finalized_prefix_regression_examples": regressions[:3],
        "committed_prefix_regressions": committed_regressions,
        "visible_text": visible_text.strip(),
        "displayed_text": displayed_text.strip(),
        "committed_text": committed_text.strip(),
    }


def benchmark_fixture(model, fixture: Fixture) -> Tuple[dict, List[Tuple[float, float]]]:
    recorder, session_started, session_seconds = replay_fixture(model, fixture)
    messages = recorder.messages

    final_message = _latest(messages, "final")
    final_text = final_message["text"] if final_message else ""
    final_source = "final" if final_message else "none"
    if not final_text:
        last_partial = _latest(messages, "partial")
        if last_partial:
            final_text = last_partial["text"]
            final_source = "last_partial"

    reference_text = None
    if fixture.reference_path and fixture.reference_path.exists():
        reference_text = fixture.reference_path.read_text(encoding="utf-8").strip()

    script = fixture.script
    script_wer_included = script is not None and (
        fixture.name.split("/")[0] not in SCRIPT_WER_EXCLUSIONS
    )

    decode = _decode_metrics(recorder, fixture.audio_seconds, session_seconds)
    latency = _latency_metrics(recorder, messages, session_started)
    corrections = _correction_metrics(messages, final_text)
    word_commit = _word_commit_metrics(recorder, session_started)
    visible_text = corrections["visible_text"]

    intervals = [
        (decode_call["started_at"], decode_call["finished_at"])
        for decode_call in recorder.decodes
    ]

    result = {
        "name": fixture.name,
        "source": fixture.source,
        "wav_path": str(fixture.wav_path.relative_to(REPO_ROOT)),
        "wav_bytes": fixture.wav_bytes,
        "audio_seconds": fixture.audio_seconds,
        "golden_message_count": fixture.golden_message_count,
        "reference_present": reference_text is not None,
        "final_source": final_source,
        "final_text": final_text,
        "visible_text": visible_text,
        "reference_text": reference_text,
        "script_text": script,
        "script_wer_included": script_wer_included,
        "session_seconds": session_seconds,
        "message_counts": {
            "partial": sum(1 for message in messages if message["type"] == "partial"),
            "final": sum(1 for message in messages if message["type"] == "final"),
        },
        "wer_reference": _wer_block(reference_text, final_text),
        "wer_script": _wer_block(script, final_text),
        "visible_wer_reference": _wer_block(reference_text, visible_text),
        "visible_wer_script": _wer_block(script, visible_text),
        "decode": decode,
        "latency": latency,
        "word_commit_delay": word_commit,
        "corrections": corrections,
    }
    return result, intervals


def _sum_wer(blocks: Sequence[Optional[dict]]) -> Optional[dict]:
    usable = [block for block in blocks if block is not None]
    if not usable:
        return None
    reference_words = sum(block["reference_words"] for block in usable)
    errors = sum(block["errors"] for block in usable)
    return {
        "fixtures": len(usable),
        "reference_words": reference_words,
        "substitutions": sum(block["substitutions"] for block in usable),
        "deletions": sum(block["deletions"] for block in usable),
        "insertions": sum(block["insertions"] for block in usable),
        "errors": errors,
        "wer": errors / reference_words if reference_words else None,
    }


def _aggregate(results: Sequence[dict]) -> dict:
    total_audio = sum(result["audio_seconds"] for result in results)
    decode_wall = sum(result["decode"]["decode_wall_seconds"] for result in results)
    decoded_audio = sum(result["decode"]["decoded_audio_seconds"] for result in results)
    session_wall = sum(result["session_seconds"] for result in results)

    per_decode_rtf: List[float] = []
    for result in results:
        per_decode_rtf.extend(result["decode"]["per_decode_rtf"])

    words = sum(result["corrections"]["words_in_final_transcript"] for result in results)
    backspaces = sum(result["corrections"]["backspaces_total"] for result in results)

    return {
        "fixtures": len(results),
        "audio_seconds": total_audio,
        "session_seconds": session_wall,
        "decode": {
            "decode_wall_seconds": decode_wall,
            "decoded_audio_seconds": decoded_audio,
            "decode_audio_amplification": decoded_audio / total_audio if total_audio else None,
            "rtf_total": decode_wall / total_audio if total_audio else None,
            "rtf_session": decode_wall / session_wall if session_wall else None,
            "rtf_per_decode_p50": _percentile(per_decode_rtf, 50),
            "rtf_per_decode_p95": _percentile(per_decode_rtf, 95),
            "rtf_per_decode_max": max(per_decode_rtf) if per_decode_rtf else None,
        },
        "wer_reference": _sum_wer(
            [result["wer_reference"] for result in results if result["wer_reference"]]
        ),
        "wer_script": _sum_wer(
            [
                result["wer_script"]
                for result in results
                if result["script_wer_included"] and result["wer_script"]
            ]
        ),
        "visible_wer_reference": _sum_wer(
            [
                result["visible_wer_reference"]
                for result in results
                if result["visible_wer_reference"]
            ]
        ),
        "visible_wer_script": _sum_wer(
            [
                result["visible_wer_script"]
                for result in results
                if result["script_wer_included"] and result["visible_wer_script"]
            ]
        ),
        "latency": _pool_latency(results),
        "word_commit_delay": _pool_word_commit_delay(results),
        "corrections": {
            "partial_messages": sum(
                result["corrections"]["partial_messages"] for result in results
            ),
            "partial_revisions": sum(
                result["corrections"]["partial_revisions"] for result in results
            ),
            "backspaces_total": backspaces,
            "visible_backspaces": sum(
                result["corrections"]["visible_backspaces"] for result in results
            ),
            "words_in_final_transcript": words,
            "backspaces_per_100_words": backspaces / words * 100.0 if words else None,
            "visible_backspaces_per_100_words": (
                sum(result["corrections"]["visible_backspaces"] for result in results)
                / words
                * 100.0
                if words
                else None
            ),
            "revisions_per_word": (
                sum(result["corrections"]["partial_revisions"] for result in results)
                / words
                if words
                else None
            ),
            "finalized_prefix_regressions": sum(
                result["corrections"]["finalized_prefix_regressions"]
                for result in results
            ),
            "committed_prefix_regressions": sum(
                result["corrections"]["committed_prefix_regressions"]
                for result in results
            ),
            "final_backspaces": sum(
                result["corrections"]["final_backspaces"] for result in results
            ),
        },
    }


def _pool_latency(results: Sequence[dict]) -> dict:
    """Pool raw latency samples so aggregate percentiles are exact."""
    samples: List[float] = []
    for result in results:
        samples.extend(result["latency"]["samples_seconds"])
    return {
        "latency_samples": len(samples),
        "p50_seconds": _percentile(samples, 50),
        "p95_seconds": _percentile(samples, 95),
        "max_seconds": max(samples) if samples else None,
        "min_seconds": min(samples) if samples else None,
        "time_to_first_partial_seconds": _median(
            [
                result["latency"]["time_to_first_partial_seconds"]
                for result in results
                if result["latency"]["time_to_first_partial_seconds"] is not None
            ]
        ),
    }


def _pool_word_commit_delay(results: Sequence[dict]) -> dict:
    samples: List[float] = []
    for result in results:
        samples.extend(result["word_commit_delay"]["samples_seconds"])
    return {
        "samples": len(samples),
        "p50_seconds": _percentile(samples, 50),
        "p95_seconds": _percentile(samples, 95),
        "max_seconds": max(samples) if samples else None,
    }


def _vram_metrics(
    samples: Sequence[Tuple[float, int]],
    intervals: Sequence[Tuple[float, float]],
    window_start: float,
    window_end: float,
    baseline_mib: Optional[float],
) -> dict:
    in_window = [(t, mib) for t, mib in samples if window_start <= t <= window_end]
    during = [
        mib
        for t, mib in in_window
        if any(start <= t <= finish for start, finish in intervals)
    ]
    all_values = [mib for _, mib in in_window]
    peak = max(during) if during else None
    return {
        "baseline_mib": baseline_mib,
        "peak_during_decode_mib": peak,
        "median_during_decode_mib": _median(during),
        "peak_in_window_mib": max(all_values) if all_values else None,
        "delta_over_baseline_mib": (
            peak - baseline_mib if peak is not None and baseline_mib is not None else None
        ),
        "samples_in_window": len(in_window),
        "samples_during_decode": len(during),
    }


def _gpu_info() -> Optional[dict]:
    try:
        result = subprocess.run(
            GPU_INFO_QUERY, capture_output=True, text=True, timeout=5.0, check=False
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        fields = next(csv.reader([result.stdout.strip().splitlines()[0]]))
        header = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5.0, check=False
        )
        match = re.search(r"CUDA Version:\s*([\d.]+)", header.stdout)
        return {
            "name": fields[0].strip(),
            "driver_version": fields[1].strip(),
            "memory_total_mib": int(fields[2].strip()),
            "cuda_version": match.group(1) if match else None,
        }
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _package_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def _model_info(model_size: str) -> dict:
    from faster_whisper.utils import _MODELS, download_model

    info = {
        "name": model_size,
        "repo_id": _MODELS.get(model_size, model_size),
        "cached": False,
        "files_bytes": None,
        "model_bin_bytes": None,
    }
    try:
        path = Path(download_model(model_size, local_files_only=True))
    except Exception:
        return info

    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    info.update(
        {
            "cached": True,
            "files_bytes": sum(candidate.stat().st_size for candidate in files),
            "model_bin_bytes": (
                (path / "model.bin").stat().st_size
                if (path / "model.bin").exists()
                else None
            ),
        }
    )
    return info


def _environment(model_size: str, device: str, compute_type: str) -> dict:
    import ctranslate2

    gpu = _gpu_info()
    return {
        "gpu": gpu,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "faster_whisper": _package_version("faster_whisper"),
            "ctranslate2": _package_version("ctranslate2"),
            "numpy": _package_version("numpy"),
        },
        "ctranslate2_cuda_device_count": ctranslate2.get_cuda_device_count(),
        "device": device,
        "compute_type": compute_type,
        "model": _model_info(model_size),
        "nvidia_smi_available": gpu is not None,
    }


def _config_fingerprint(args: argparse.Namespace) -> dict:
    return {
        "sample_rate": SAMPLE_RATE,
        "bytes_per_sample": BYTES_PER_SAMPLE,
        "transcribe_interval_seconds": TRANSCRIBE_INTERVAL,
        "transcribe_min_audio_seconds": TRANSCRIBE_MIN_AUDIO_SECONDS,
        "keep_tail_seconds": KEEP_TAIL_SECONDS,
        "max_session_seconds": MAX_SESSION_SECONDS,
        "beam_size": WHISPER_BEAM_SIZE,
        "temperature": WHISPER_TEMPERATURE,
        "vad_filter": WHISPER_VAD_FILTER,
        "vad_min_silence_ms": WHISPER_VAD_MIN_SILENCE_MS,
        "no_repeat_ngram_size": WHISPER_NO_REPEAT_NGRAM_SIZE,
        "repetition_penalty": WHISPER_REPETITION_PENALTY,
        "hotwords_present": bool(WHISPER_HOTWORD_TERMS),
        "hotwords_count": len(WHISPER_HOTWORD_TERMS),
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "feed_frame_seconds": FEED_FRAME_SECONDS,
        "vram_sample_interval_seconds": args.vram_interval,
        "pacing": "real_time",
    }


def _format_latency(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds * 1000:.0f}ms"


def _format_metric(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _print_run(run: dict, fixture_results: Sequence[dict], vram: dict) -> None:
    header = (
        f"{'fixture':<28} {'audio_s':>8} {'WER_ref':>8} {'WER_scr':>8} "
        f"{'RTF':>6} {'cRTF50':>7} {'cRTF95':>7} {'lat50':>7} {'lat95':>7} "
        f"{'latmax':>7} {'rev':>4} {'bs/100w':>8} {'reg':>4}"
    )
    print(header)
    print("-" * len(header))
    for result in fixture_results:
        wer_reference = result["wer_reference"]["wer"] if result["wer_reference"] else None
        wer_script = result["wer_script"]["wer"] if result["wer_script"] else None
        latency = result["latency"]
        corrections = result["corrections"]
        row = (
            f"{result['name'][:28]:<28} {result['audio_seconds']:>8.2f} "
            f"{_format_metric(wer_reference):>8} {_format_metric(wer_script):>8} "
            f"{_format_metric(result['decode']['rtf_total']):>6} "
            f"{_format_metric(result['decode']['rtf_per_decode_p50']):>7} "
            f"{_format_metric(result['decode']['rtf_per_decode_p95']):>7} "
            f"{_format_latency(latency['p50_seconds']):>7} "
            f"{_format_latency(latency['p95_seconds']):>7} "
            f"{_format_latency(latency['max_seconds']):>7} "
            f"{corrections['partial_revisions']:>4} "
            f"{_format_metric(corrections['backspaces_per_100_words'], 1):>8} "
            f"{corrections['finalized_prefix_regressions']:>4}"
        )
        print(row)

    aggregate = run["aggregate"]
    wer_reference = aggregate["wer_reference"]["wer"] if aggregate["wer_reference"] else None
    wer_script = aggregate["wer_script"]["wer"] if aggregate["wer_script"] else None
    latency = aggregate["latency"]
    corrections = aggregate["corrections"]
    row = (
        f"{'AGGREGATE':<28} {aggregate['audio_seconds']:>8.2f} "
        f"{_format_metric(wer_reference):>8} {_format_metric(wer_script):>8} "
        f"{_format_metric(aggregate['decode']['rtf_total']):>6} "
        f"{_format_metric(aggregate['decode']['rtf_per_decode_p50']):>7} "
        f"{_format_metric(aggregate['decode']['rtf_per_decode_p95']):>7} "
        f"{_format_latency(latency['p50_seconds']):>7} "
        f"{_format_latency(latency['p95_seconds']):>7} "
        f"{_format_latency(latency['max_seconds']):>7} "
        f"{corrections['partial_revisions']:>4} "
        f"{_format_metric(corrections['backspaces_per_100_words'], 1):>8} "
        f"{corrections['finalized_prefix_regressions']:>4}"
    )
    print(row)
    print(
        f"  First partial after audio start (median): "
        f"{_format_latency(latency['time_to_first_partial_seconds'])}"
    )
    word_commit = aggregate["word_commit_delay"]
    print(
        f"  Word commit delay p50/p95/max: "
        f"{_format_latency(word_commit['p50_seconds'])} / "
        f"{_format_latency(word_commit['p95_seconds'])} / "
        f"{_format_latency(word_commit['max_seconds'])} "
        f"({word_commit['samples']} words); revisions/word "
        f"{_format_metric(corrections['revisions_per_word'])}"
    )
    visible_reference = aggregate.get("visible_wer_reference")
    visible_script = aggregate.get("visible_wer_script")
    print(
        f"  Visible screen before final: WER_ref "
        f"{_format_metric(visible_reference['wer'] if visible_reference else None)} "
        f"WER_scr "
        f"{_format_metric(visible_script['wer'] if visible_script else None)}; "
        f"visible backspaces {corrections['visible_backspaces']} "
        f"({_format_metric(corrections['visible_backspaces_per_100_words'], 1)}/100w)"
    )
    print(
        f"  VRAM: baseline {_format_metric(vram['baseline_mib'], 0)} MiB, "
        f"peak during decode {_format_metric(vram['peak_during_decode_mib'], 0)} MiB, "
        f"median during decode {_format_metric(vram['median_during_decode_mib'], 0)} MiB, "
        f"delta {_format_metric(vram['delta_over_baseline_mib'], 0)} MiB "
        f"({vram['samples_during_decode']} samples)"
    )
    print(
        f"  Decode wall {aggregate['decode']['decode_wall_seconds']:.2f}s for "
        f"{aggregate['audio_seconds']:.2f}s audio "
        f"(amplification {_format_metric(aggregate['decode']['decode_audio_amplification'], 2)}x)"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the live dictation pipeline")
    parser.add_argument("--model", default=WHISPER_MODEL_SIZE)
    parser.add_argument("--device", default=WHISPER_DEVICE)
    parser.add_argument("--compute-type", default=WHISPER_COMPUTE_TYPE)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Also replay gitignored tests/audio_fixtures_local takes",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        help="Only run fixtures whose name contains this substring (repeatable)",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--vram-interval", type=float, default=VRAM_SAMPLE_INTERVAL_SECONDS
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    fixtures = _discover_fixtures(args.include_local, args.fixture)
    if not fixtures:
        print("No fixtures matched", file=sys.stderr)
        return 2
    for fixture in fixtures:
        _load_fixture_audio(fixture)

    environment = _environment(args.model, args.device, args.compute_type)
    print(
        f"GPU: {environment['gpu']['name'] if environment['gpu'] else 'unavailable'}, "
        f"Python {environment['python_version']}, "
        f"faster-whisper {environment['packages']['faster_whisper']}, "
        f"CTranslate2 {environment['packages']['ctranslate2']}"
    )
    print(
        f"Model {args.model} on {args.device} ({args.compute_type}), "
        f"cached={environment['model']['cached']}, "
        f"files={environment['model']['files_bytes']}"
    )
    print(f"Fixtures: {len(fixtures)}")

    sampler = VramSampler(args.vram_interval)
    sampler.start()
    time.sleep(VRAM_BASELINE_SECONDS)
    baseline_mib = _median([mib for _, mib in sampler.samples])
    baseline_sample_count = len(sampler.samples)

    print("Loading model...", flush=True)
    load_started = time.perf_counter()
    model = load_whisper_model(args.model, args.device, args.compute_type, quiet=True)
    model_load_seconds = time.perf_counter() - load_started
    load_finished = time.perf_counter()
    time.sleep(VRAM_SETTLE_SECONDS)

    runs = []
    run_windows: List[Tuple[float, float]] = []
    run_intervals: List[List[Tuple[float, float]]] = []
    for run_index in range(1, args.runs + 1):
        run_started = time.perf_counter()
        fixture_results = []
        current_intervals: List[Tuple[float, float]] = []
        for fixture in fixtures:
            result, intervals = benchmark_fixture(model, fixture)
            fixture_results.append(result)
            current_intervals.extend(intervals)
            print(
                f"  run {run_index} {fixture.name}: "
                f"WER_ref={_format_metric(result['wer_reference']['wer'] if result['wer_reference'] else None)} "
                f"RTF={_format_metric(result['decode']['rtf_total'])} "
                f"lat95={_format_latency(result['latency']['p95_seconds'])}",
                flush=True,
            )
        run_ended = time.perf_counter()

        aggregate = _aggregate(fixture_results)
        run = {
            "run": run_index,
            "started_at_offset_seconds": run_started - load_finished,
            "wall_seconds": run_ended - run_started,
            "fixtures": fixture_results,
            "aggregate": aggregate,
            "aggregate_by_source": {
                source: _aggregate(
                    [result for result in fixture_results if result["source"] == source]
                )
                for source in sorted({result["source"] for result in fixture_results})
            },
        }
        runs.append(run)
        run_windows.append((run_started, run_ended))
        run_intervals.append(current_intervals)

    sampler.stop()
    if sampler.error:
        print(f"VRAM sampling stopped: {sampler.error}", file=sys.stderr)

    for run, window, intervals in zip(runs, run_windows, run_intervals):
        run["vram"] = _vram_metrics(
            sampler.samples, intervals, window[0], window[1], baseline_mib
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment,
        "config": _config_fingerprint(args),
        "model_load_seconds": model_load_seconds,
        "baseline_vram_mib": baseline_mib,
        "baseline_vram_samples": baseline_sample_count,
        "vram_sampling_error": sampler.error,
        "fixtures": [
            {
                "name": fixture.name,
                "source": fixture.source,
                "wav_path": str(fixture.wav_path.relative_to(REPO_ROOT)),
                "wav_bytes": fixture.wav_bytes,
                "audio_seconds": fixture.audio_seconds,
                "golden_message_count": fixture.golden_message_count,
                "reference_present": fixture.reference_path is not None
                and fixture.reference_path.exists(),
                "script_present": fixture.script is not None,
            }
            for fixture in fixtures
        ],
        "runs": runs,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for run in runs:
        print()
        print(f"=== Run {run['run']} ===")
        _print_run(run, run["fixtures"], run["vram"])
    print()
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
