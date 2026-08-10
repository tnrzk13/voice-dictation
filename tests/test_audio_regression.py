"""Audio regression tests - replay captured chunk sequences from real audio.

These tests do not load the Whisper model. They read golden chunk sequences
produced by tools/capture_chunks.py and replay them through the live client
message routing to verify the end-to-end typed output.

To regenerate the fixtures after changing the model or adding new audio:

    python tools/generate_audio_fixtures.py
    python tools/capture_chunks.py tests/audio_fixtures/*.wav
"""

import json
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from dictate.live.client import LiveDaemonClient
from dictate.live.typer import ProgressiveTyper


FIXTURES_DIR = Path(__file__).parent / "audio_fixtures"
LOCAL_FIXTURES_DIR = Path(__file__).parent / "audio_fixtures_local"


@pytest.fixture
def typer() -> ProgressiveTyper:
    return ProgressiveTyper()


def _load_chunks(fixture_dir: Path) -> List[dict]:
    """Load captured JSON messages for a fixture."""
    path = fixture_dir / "chunks.jsonl"
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunks.append(json.loads(line))
    return chunks


def _load_reference(fixture_dir: Path) -> str:
    """Load the raw reference transcript produced by the model."""
    path = fixture_dir / "reference.txt"
    return path.read_text(encoding="utf-8").strip()


def _replay_chunks(typer: ProgressiveTyper, chunks: List[dict]) -> None:
    """Replay captured daemon messages through the client message router."""
    client = LiveDaemonClient(typer=typer, streaming=True)
    for chunk in chunks:
        client._handle_message(json.dumps(chunk))


def _discover_local_fixtures() -> List[tuple]:
    """Return (name, dir) pairs for any local fixtures with captured chunks."""
    fixtures = []
    if LOCAL_FIXTURES_DIR.exists():
        for fixture_dir in sorted(LOCAL_FIXTURES_DIR.iterdir()):
            if (fixture_dir / "chunks.jsonl").exists():
                fixtures.append((fixture_dir.name, fixture_dir))
    return fixtures


def _normalize(text: str) -> str:
    """Normalize text for tolerant comparison: lowercase and strip whitespace."""
    return text.strip().lower()


def _has_duplicate_segments(text: str) -> bool:
    """Return True if the text contains the same two-word sequence twice in a row."""
    words = text.strip().lower().split()
    if len(words) < 4:
        return False
    for i in range(len(words) - 3):
        if words[i] == words[i + 2] and words[i + 1] == words[i + 3]:
            return True
    return False


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestAudioRegression:
    def test_hello_world(self, mock_type, mock_bs, typer: ProgressiveTyper) -> None:
        """A simple sentence is typed correctly and fully committed."""
        chunks = _load_chunks(FIXTURES_DIR / "hello_world")
        _replay_chunks(typer, chunks)

        reference = _load_reference(FIXTURES_DIR / "hello_world")
        assert _normalize(typer.displayed_text) == _normalize(reference)
        assert typer.committed == "Hello world, this is a test. "
        assert typer.pending == ""

    def test_formatting_commands(self, mock_type, mock_bs, typer: ProgressiveTyper) -> None:
        """Spoken formatting commands are converted to symbols during playback."""
        chunks = _load_chunks(FIXTURES_DIR / "formatting_commands")
        _replay_chunks(typer, chunks)

        assert typer.displayed_text == "Tony/pictures from the beach. "
        assert "/" in typer.displayed_text
        assert "." in typer.displayed_text
        assert "slash" not in _normalize(typer.displayed_text)
        assert "period" not in _normalize(typer.displayed_text)

    def test_pause_and_continue_no_duplicates(self, mock_type, mock_bs, typer: ProgressiveTyper) -> None:
        """Audio with a pause does not cause duplicated text to be typed."""
        chunks = _load_chunks(FIXTURES_DIR / "pause_and_continue")
        _replay_chunks(typer, chunks)

        assert not _has_duplicate_segments(typer.displayed_text)
        assert _normalize(typer.displayed_text).startswith("this is the first part")
        assert "second part" in _normalize(typer.displayed_text)


LOCAL_FIXTURES = _discover_local_fixtures()


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
@pytest.mark.skipif(not LOCAL_FIXTURES, reason="No local audio fixtures found")
class TestLocalAudioRegression:
    @pytest.mark.parametrize("fixture_name, fixture_dir", LOCAL_FIXTURES)
    def test_local_fixture_does_not_duplicate_or_crash(
        self, mock_type, mock_bs, typer: ProgressiveTyper, fixture_name: str, fixture_dir: Path
    ) -> None:
        """Replays a user-recorded fixture and verifies the typer behaves sanely."""
        chunks = _load_chunks(fixture_dir)
        _replay_chunks(typer, chunks)

        assert typer.displayed_text.strip(), f"{fixture_name}: produced no text"
        assert not _has_duplicate_segments(typer.displayed_text), f"{fixture_name}: duplicated text"

    def test_numbers_and_symbols_preserves_formatted_values(
        self, mock_type, mock_bs, typer: ProgressiveTyper
    ) -> None:
        """The numbers_and_symbols fixture should keep the formatted numbers and symbols."""
        fixture_dir = LOCAL_FIXTURES_DIR / "numbers_and_symbols"
        if not fixture_dir.exists():
            pytest.skip("numbers_and_symbols fixture not recorded")
        chunks = _load_chunks(fixture_dir)
        _replay_chunks(typer, chunks)

        assert "$42.50" in typer.displayed_text
        assert "1%" in typer.displayed_text
        assert "8472" in typer.displayed_text
