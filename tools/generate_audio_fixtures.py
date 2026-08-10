#!/usr/bin/env python3
"""Generate synthetic audio samples for dictation regression tests.

Uses gTTS (Google Text-to-Speech) to create MP3 files, then converts them to
16kHz mono PCM WAV using pydub. The generated files are intended to be processed
by tools/capture_chunks.py to produce golden chunk sequences.

Run with:
    python tools/generate_audio_fixtures.py

Requires: gTTS, pydub (already installed in this environment; not required at runtime)
"""

import argparse
from pathlib import Path

from gtts import gTTS
from pydub import AudioSegment

from dictate.config import BYTES_PER_SAMPLE, SAMPLE_RATE


# Fixture definitions: name -> text to synthesize.
# For "pause_and_continue" we generate two clips with silence between them.
FIXTURES = {
    "hello_world": "Hello world, this is a test.",
    "formatting_commands": "Tony slash pictures from the beach period",
    "pause_first_part": "This is the first part.",
    "pause_second_part": "And this is the second part.",
}

PAUSE_SILENCE_MS = 1000


def _generate_speech(text: str, output_path: Path) -> None:
    """Use gTTS to create a WAV file from the given text."""
    mp3_path = output_path.with_suffix(".mp3")
    tts = gTTS(text, lang="en", slow=False)
    tts.save(str(mp3_path))

    audio = AudioSegment.from_mp3(str(mp3_path))
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(BYTES_PER_SAMPLE)
    audio.export(str(output_path), format="wav")
    mp3_path.unlink()


def _generate_pause_fixture(output_dir: Path) -> None:
    """Concatenate two speech clips with silence to simulate a mid-sentence pause."""
    first = AudioSegment.from_wav(str(output_dir / "pause_first_part.wav"))
    second = AudioSegment.from_wav(str(output_dir / "pause_second_part.wav"))
    silence = AudioSegment.silent(duration=PAUSE_SILENCE_MS)

    combined = first + silence + second
    combined = combined.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(BYTES_PER_SAMPLE)
    combined.export(str(output_dir / "pause_and_continue.wav"), format="wav")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio fixtures for dictation tests")
    parser.add_argument("--output-dir", default="tests/audio_fixtures", help="Where to write audio files")
    parser.add_argument("--fixture", help="Generate only this fixture")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, text in FIXTURES.items():
        if args.fixture and name != args.fixture:
            continue
        wav_path = output_dir / f"{name}.wav"
        _generate_speech(text, wav_path)
        print(f"Generated {wav_path}")

    if not args.fixture or args.fixture == "pause_and_continue":
        _generate_pause_fixture(output_dir)
        print(f"Generated {output_dir / 'pause_and_continue.wav'}")


if __name__ == "__main__":
    main()
