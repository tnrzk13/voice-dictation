"""Audio regression tests - replay captured chunk sequences from real audio.

These tests do not load the Whisper model. They read golden chunk sequences
produced by tools/capture_chunks.py and replay them through the live client
message routing to verify the end-to-end typed output.

To regenerate the fixtures after changing the model or adding new audio:

    python tools/generate_audio_fixtures.py
    python tools/capture_chunks.py tests/audio_fixtures/*.wav
"""

import difflib
import json
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from dictate.live.client import LiveDaemonClient
from dictate.live.formatting import apply_formatting_commands
from dictate.live.typer import ProgressiveTyper
from tools.fixture_definitions import FIXTURES
from tools.fixture_store import (
    CHUNKS_FILENAME,
    LOCAL_FIXTURES_DIR,
    TAKES_DIRNAME,
)


FIXTURES_DIR = Path(__file__).parent / "audio_fixtures"


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


def _discover_local_takes() -> List[tuple]:
    """Return (name, dir) pairs for local takes that have captured chunks."""
    takes = []
    if LOCAL_FIXTURES_DIR.exists():
        for fixture_dir in sorted(LOCAL_FIXTURES_DIR.iterdir()):
            takes_dir = fixture_dir / TAKES_DIRNAME
            if not takes_dir.is_dir():
                continue
            for take_dir in sorted(takes_dir.iterdir()):
                if (take_dir / CHUNKS_FILENAME).exists():
                    takes.append((f"{fixture_dir.name}/{take_dir.name}", take_dir))
    return takes


REFERENCE_SIMILARITY_THRESHOLD = 0.99
SCRIPT_SIMILARITY_THRESHOLD = 0.95


def _normalize(text: str) -> str:
    """Normalize text for tolerant comparison: lowercase and collapse whitespace."""
    return " ".join(text.strip().lower().split())


def _similarity(a: str, b: str) -> float:
    """Return a ratio in [0, 1] describing how closely two texts match."""
    return difflib.SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


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


LOCAL_TAKES = _discover_local_takes()


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
@pytest.mark.skipif(not LOCAL_TAKES, reason="No local audio takes found")
class TestLocalAudioRegression:
    @pytest.mark.parametrize(
        "take_name, take_dir", LOCAL_TAKES, ids=[name for name, _ in LOCAL_TAKES]
    )
    def test_local_take_reproduces_reference(
        self, mock_type, mock_bs, typer: ProgressiveTyper, take_name: str, take_dir: Path
    ) -> None:
        """Replaying a take's chunks must type the same text the model produced."""
        chunks = _load_chunks(take_dir)
        _replay_chunks(typer, chunks)

        reference = _load_reference(take_dir)
        expected = apply_formatting_commands(reference).strip()
        assert typer.displayed_text.strip(), f"{take_name}: produced no text"
        similarity = _similarity(typer.displayed_text, expected)
        assert similarity >= REFERENCE_SIMILARITY_THRESHOLD, (
            f"{take_name}: typed text diverged from reference "
            f"(similarity {similarity:.3f})\n"
            f"  typed:     {typer.displayed_text!r}\n"
            f"  reference: {expected!r}"
        )
        assert not _has_duplicate_segments(typer.displayed_text), f"{take_name}: duplicated text"

        fixture_name = take_name.split("/")[0]
        info = FIXTURES.get(fixture_name, {})
        if info.get("verify_against_script"):
            script_similarity = _similarity(reference, info["script"])
            assert script_similarity >= SCRIPT_SIMILARITY_THRESHOLD, (
                f"{take_name}: captured reference diverged from the script "
                f"(similarity {script_similarity:.3f})\n"
                f"  script:    {info['script']!r}\n"
                f"  reference: {reference!r}"
            )
