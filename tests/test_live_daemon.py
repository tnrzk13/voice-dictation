"""Tests for the live dictation daemon message protocol."""

import json
import time
from unittest.mock import MagicMock, patch

from dictate.config import BYTES_PER_SAMPLE, BYTES_PER_SECOND
from dictate.live.daemon import (
    _collapse_repetitions,
    _concat_transcriptions,
    _finalize_completed_segments,
    _send_message,
    _trim_oldest_audio,
    handle_client,
)


class TestSendMessage:
    def test_sends_partial_as_json(self):
        conn = MagicMock()
        _send_message(conn, "partial", "hello")
        sent = conn.sendall.call_args[0][0]
        msg = json.loads(sent.decode("utf-8").strip())
        assert msg == {"type": "partial", "text": "hello"}

    def test_sends_final_as_json(self):
        conn = MagicMock()
        _send_message(conn, "final", "hello world")
        sent = conn.sendall.call_args[0][0]
        msg = json.loads(sent.decode("utf-8").strip())
        assert msg == {"type": "final", "text": "hello world"}

    def test_sends_end_marker(self):
        conn = MagicMock()
        _send_message(conn, "end", "")
        sent = conn.sendall.call_args[0][0]
        msg = json.loads(sent.decode("utf-8").strip())
        assert msg == {"type": "end", "text": ""}

    def test_messages_are_newline_terminated(self):
        conn = MagicMock()
        _send_message(conn, "partial", "test")
        sent = conn.sendall.call_args[0][0]
        assert sent.endswith(b"\n")


def _make_segment(text=" Hello world.", start=0.0, end=1.0):
    """Create a mock Whisper segment with .text, .start, .end attributes."""
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    return seg


def _timed_segment(words, starts, end):
    """Build a segment dict with per-word start/end spans."""
    return {
        "text": " ".join(words),
        "start": float(starts[0]) if starts else 0.0,
        "end": end,
        "words": [{"start": float(s), "end": float(s) + 0.5} for s in starts],
    }


def _make_whisper_model(segments=None):
    """Create a mock Whisper model returning fresh segment iterators per call.

    Args:
        segments: List of mock segments to return from transcribe().
                  Defaults to a single segment with " Hello world."
    """
    if segments is None:
        segments = [_make_segment()]
    model = MagicMock()
    model.transcribe.side_effect = lambda *a, **kw: (iter(list(segments)), None)
    return model


def _parse_sent_messages(conn):
    """Extract all JSON messages sent via sendall on a mock connection."""
    messages = []
    for call in conn.sendall.call_args_list:
        raw = call[0][0].decode("utf-8").strip()
        messages.append(json.loads(raw))
    return messages


class TestConcatTranscriptions:
    def test_joins_with_space(self):
        assert _concat_transcriptions("hello", "world") == "hello world"

    def test_strips_leading_space_from_whisper_segment(self):
        assert _concat_transcriptions("hello", " world") == "hello world"

    def test_empty_finalized(self):
        assert _concat_transcriptions("", " Hello world.") == "Hello world."

    def test_empty_new(self):
        assert _concat_transcriptions("hello", "") == "hello"

    def test_both_empty(self):
        assert _concat_transcriptions("", "") == ""

    def test_strips_trailing_space_from_finalized(self):
        assert _concat_transcriptions("hello ", " world") == "hello world"


class TestFinalizeCompletedSegments:
    def test_adds_space_between_finalized_and_segment(self):
        """Segments without leading spaces get proper word separation."""
        segments = [
            {"text": "we can", "start": 0.0, "end": 5.0},
            {"text": "do it", "start": 5.0, "end": 10.0},
        ]
        finalized, _ = _finalize_completed_segments(segments, "")
        assert finalized == "we can"

    def test_accumulates_across_multiple_trims(self):
        """Multiple buffer trims maintain spacing in finalized_text."""
        segments1 = [
            {"text": "we can", "start": 0.0, "end": 5.0},
            {"text": "do", "start": 5.0, "end": 7.0},
        ]
        finalized, _ = _finalize_completed_segments(segments1, "")
        assert finalized == "we can"

        segments2 = [
            {"text": "do it", "start": 0.0, "end": 3.0},
            {"text": "now", "start": 3.0, "end": 5.0},
        ]
        finalized, _ = _finalize_completed_segments(segments2, finalized)
        assert finalized == "we can do it"

    def test_single_segment_keeps_tail_of_speech(self):
        """A long single segment finalizes all but the last KEEP_TAIL_SECONDS of speech."""
        words = [f"word{i}" for i in range(10)]
        segments = [_timed_segment(words, starts=range(10), end=10.0)]
        finalized, bytes_trimmed = _finalize_completed_segments(segments, "")
        # Word starts 0..9s; keeping >=3s makes word at 7.0s the first kept.
        assert finalized == " ".join(words[:7])
        assert bytes_trimmed == 7 * BYTES_PER_SECOND

    def test_single_segment_uses_word_timestamps_across_pause(self):
        """A pause inflates duration, so the split must come from word timings."""
        words = [f"word{i}" for i in range(8)]
        starts = [0.0, 0.6, 1.2, 1.7, 6.0, 6.6, 7.1, 7.6]
        segments = [_timed_segment(words, starts=starts, end=8.0)]
        finalized, bytes_trimmed = _finalize_completed_segments(segments, "")
        # Speech resumes at 6.0s, so keeping >=3s starts the tail at word 3 (1.7s).
        assert finalized == "word0 word1 word2"
        assert bytes_trimmed == int(1.7 * BYTES_PER_SECOND)

    def test_single_segment_defers_when_word_timings_misaligned(self):
        """Repetition collapse can desync text from timings - defer, don't guess."""
        segments = [
            {
                "text": "one two three",
                "start": 0.0,
                "end": 10.0,
                "words": [{"start": 0.0, "end": 0.5}],
            }
        ]
        finalized, bytes_trimmed = _finalize_completed_segments(segments, "prior")
        assert finalized == "prior"
        assert bytes_trimmed == 0

    def test_single_segment_shorter_than_tail_keeps_all(self):
        """A segment shorter than KEEP_TAIL_SECONDS stays in the buffer."""
        segments = [_timed_segment(["hello", "world"], starts=[0.0, 1.0], end=2.0)]
        finalized, bytes_trimmed = _finalize_completed_segments(segments, "")
        assert finalized == ""
        assert bytes_trimmed == 0

    def test_single_segment_single_word_keeps_all(self):
        """A one-word segment cannot be split - finalize nothing."""
        segments = [_timed_segment(["hello"], starts=[0.0], end=5.0)]
        finalized, bytes_trimmed = _finalize_completed_segments(segments, "")
        assert finalized == ""
        assert bytes_trimmed == 0

    def test_bytes_trimmed_aligned_to_int16(self):
        """Trimmed bytes are aligned to 2-byte int16 boundary."""
        segments = [
            {"text": "hello", "start": 0.0, "end": 1.0},
            {"text": "world", "start": 1.5, "end": 3.0},
        ]
        _, bytes_trimmed = _finalize_completed_segments(segments, "")
        assert bytes_trimmed % 2 == 0
        assert bytes_trimmed == 48000  # 1.5 * 32000 = 48000


class TestHandleClient:
    @patch("dictate.live.daemon.TRANSCRIBE_INTERVAL", 0.01)
    def test_sends_partial_results(self):
        """Transcription during the session produces partial messages."""
        model = _make_whisper_model([_make_segment(" Hello world.")])

        conn = MagicMock()
        audio = b"\x00" * 8000
        calls = []

        def recv_with_delay(size):
            calls.append(1)
            if len(calls) == 1:
                return audio
            # Block receiver so transcriber can run at least one cycle
            time.sleep(0.1)
            return b""

        conn.recv.side_effect = recv_with_delay

        handle_client(conn, model)

        messages = _parse_sent_messages(conn)
        partials = [m for m in messages if m["type"] == "partial"]
        assert len(partials) >= 1
        assert "Hello world." in partials[0]["text"]

    def test_sends_final_on_eof(self):
        """Final transcription is sent when the client finishes."""
        model = _make_whisper_model([_make_segment(" Final words.")])

        conn = MagicMock()
        conn.recv.side_effect = [b"\x00" * 8000, b""]

        handle_client(conn, model)

        messages = _parse_sent_messages(conn)
        finals = [m for m in messages if m["type"] == "final"]
        assert any("Final words." in m["text"] for m in finals)

    def test_sends_end_on_completion(self):
        """End marker is always the last message sent."""
        model = _make_whisper_model()

        conn = MagicMock()
        conn.recv.side_effect = [b"\x00" * 8000, b""]

        handle_client(conn, model)

        messages = _parse_sent_messages(conn)
        assert messages[-1]["type"] == "end"

    def test_skips_empty_transcription(self):
        """No partial/final messages are sent when Whisper returns nothing."""
        model = _make_whisper_model([])

        conn = MagicMock()
        conn.recv.side_effect = [b"\x00" * 8000, b""]

        handle_client(conn, model)

        messages = _parse_sent_messages(conn)
        assert all(m["type"] == "end" for m in messages)

    def test_handles_client_disconnect(self):
        """Daemon handles abrupt client disconnection without crashing."""
        model = _make_whisper_model()

        conn = MagicMock()
        conn.recv.side_effect = ConnectionResetError("client gone")

        # Should not raise
        handle_client(conn, model)


class TestCommitPolicy:
    @patch("dictate.live.daemon.TRANSCRIBE_INTERVAL", 0.01)
    @patch("dictate.live.daemon._finalize_completed_segments")
    def test_finalizes_segments_every_cycle_below_20_seconds(self, mock_finalize):
        """Completed segments are finalized each cycle, not only after 20s."""
        mock_finalize.return_value = ("", 0)
        model = _make_whisper_model(
            [
                _make_segment(" Hello", 0.0, 0.5),
                _make_segment(" world", 0.5, 1.0),
            ]
        )

        conn = MagicMock()
        audio = b"\x00" * 8000
        calls = []

        def recv_with_delay(size):
            calls.append(1)
            if len(calls) == 1:
                return audio
            time.sleep(0.1)
            return b""

        conn.recv.side_effect = recv_with_delay

        handle_client(conn, model)

        # 8000 bytes is 0.25s of audio, far below the old 20s window
        assert mock_finalize.call_count >= 1


class TestTrimOldestAudio:
    def test_trims_to_max_bytes_aligned_to_int16(self):
        audio = bytearray(b"\x00" * 100)
        _trim_oldest_audio(audio, max_bytes=50)
        assert len(audio) <= 50
        assert len(audio) % BYTES_PER_SAMPLE == 0

    def test_no_change_when_under_max(self):
        audio = bytearray(b"\x00" * 40)
        _trim_oldest_audio(audio, max_bytes=50)
        assert len(audio) == 40


class TestCollapseRepetitions:
    def test_collapses_repeated_phrase(self):
        text = "shift super d to shift super d to kill the daemon"
        assert _collapse_repetitions(text) == "shift super d to kill the daemon"

    def test_collapses_case_insensitively(self):
        text = "Shift Super D to shift super d to"
        assert _collapse_repetitions(text) == "Shift Super D to"

    def test_no_collapse_for_short_repeat(self):
        assert _collapse_repetitions("no no") == "no no"

    def test_no_collapse_when_no_repetition(self):
        text = "shift super d to kill the daemon"
        assert _collapse_repetitions(text) == text

    def test_preserves_leading_space(self):
        text = " shift super d to shift super d to"
        assert _collapse_repetitions(text) == " shift super d to"

    def test_collapses_three_word_phrase(self):
        text = "one two three one two three four"
        assert _collapse_repetitions(text) == "one two three four"
