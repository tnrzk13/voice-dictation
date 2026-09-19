"""On-disk layout for local audio fixtures.

A fixture can hold multiple recordings ("takes"). Each take lives in its own
directory so its audio and captured chunks stay together:

    tests/audio_fixtures_local/<fixture>/takes/<take_id>/audio.wav
    tests/audio_fixtures_local/<fixture>/takes/<take_id>/chunks.jsonl
    tests/audio_fixtures_local/<fixture>/takes/<take_id>/reference.txt

Older fixtures stored ``audio.wav`` directly under the fixture directory; those
are migrated into ``takes/take-1/`` on first access.
"""

from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_FIXTURES_DIR = PROJECT_ROOT / "tests" / "audio_fixtures_local"
TAKES_DIRNAME = "takes"
AUDIO_FILENAME = "audio.wav"
CHUNKS_FILENAME = "chunks.jsonl"
REFERENCE_FILENAME = "reference.txt"
FIRST_TAKE_ID = "take-1"


def fixture_dir(fixture: str, base: Path = LOCAL_FIXTURES_DIR) -> Path:
    """Return the directory for a fixture."""
    return base / fixture


def takes_dir(fixture: str, base: Path = LOCAL_FIXTURES_DIR) -> Path:
    """Return the directory that holds a fixture's takes."""
    return fixture_dir(fixture, base) / TAKES_DIRNAME


def take_dir(fixture: str, take_id: str, base: Path = LOCAL_FIXTURES_DIR) -> Path:
    """Return the directory for a single take."""
    return takes_dir(fixture, base) / take_id


def take_audio_path(fixture: str, take_id: str, base: Path = LOCAL_FIXTURES_DIR) -> Path:
    """Return the WAV path for a take."""
    return take_dir(fixture, take_id, base) / AUDIO_FILENAME


def take_chunks_path(fixture: str, take_id: str, base: Path = LOCAL_FIXTURES_DIR) -> Path:
    """Return the captured chunk path for a take."""
    return take_dir(fixture, take_id, base) / CHUNKS_FILENAME


def take_reference_path(fixture: str, take_id: str, base: Path = LOCAL_FIXTURES_DIR) -> Path:
    """Return the reference transcript path for a take."""
    return take_dir(fixture, take_id, base) / REFERENCE_FILENAME


def list_take_ids(fixture: str, base: Path = LOCAL_FIXTURES_DIR) -> List[str]:
    """Return take ids that have audio, ordered by take number."""
    directory = takes_dir(fixture, base)
    if not directory.exists():
        return []
    take_ids = [
        path.name
        for path in directory.iterdir()
        if path.is_dir() and (path / AUDIO_FILENAME).exists()
    ]
    return sorted(take_ids, key=_take_sort_key)


def next_take_id(fixture: str, base: Path = LOCAL_FIXTURES_DIR) -> str:
    """Return the next unused take id (``take-N``)."""
    numbers = [
        number for number in map(_take_number, list_take_ids(fixture, base)) if number > 0
    ]
    return f"take-{max(numbers, default=0) + 1}"


def migrate_legacy_take(fixture: str, base: Path = LOCAL_FIXTURES_DIR) -> bool:
    """Move a legacy ``audio.wav`` fixture into ``takes/take-1/``.

    Returns True when a migration happened.
    """
    legacy_audio = fixture_dir(fixture, base) / AUDIO_FILENAME
    if not legacy_audio.exists():
        return False

    target = take_dir(fixture, FIRST_TAKE_ID, base)
    if target.exists():
        return False

    target.mkdir(parents=True, exist_ok=True)
    legacy_audio.rename(target / AUDIO_FILENAME)
    for filename in (CHUNKS_FILENAME, REFERENCE_FILENAME):
        source = fixture_dir(fixture, base) / filename
        if source.exists():
            source.rename(target / filename)
    return True


def _take_number(take_id: str) -> int:
    """Extract the integer suffix of a ``take-N`` id, or 0 if absent."""
    prefix, separator, suffix = take_id.rpartition("-")
    if separator and prefix == "take" and suffix.isdigit():
        return int(suffix)
    return 0


def _take_sort_key(take_id: str) -> tuple:
    return (_take_number(take_id), take_id)
