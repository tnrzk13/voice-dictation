#!/usr/bin/env python3
"""Record your own voice for local audio regression fixtures.

Guides you through a script of dictation edge cases, records each one from
your microphone, and saves WAV files to tests/audio_fixtures_local/. These
recordings are gitignored by default and stay only on your local machine.

After recording, run capture_chunks.py to generate the golden chunk sequences:

    python tools/capture_chunks.py tests/audio_fixtures_local/*.wav \
        --output-dir tests/audio_fixtures_local

Requires: sounddevice, numpy (already installed for the main app)
"""

import argparse
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

# Allow running this script directly: python tools/record_local_fixtures.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from dictate.config import BYTES_PER_SAMPLE, BYTES_PER_SECOND, SAMPLE_RATE
from tools.fixture_definitions import FIXTURES
from tools.fixture_store import list_take_ids, next_take_id, take_audio_path


def _record_until_enter() -> bytes:
    """Record audio from the default microphone until the user presses Enter."""
    frames = []
    stop_event = threading.Event()

    def _input_thread() -> None:
        input()
        stop_event.set()

    def _audio_callback(indata: np.ndarray, frames_count: int, time_info, status) -> None:
        if status:
            print(f"Audio status: {status}", flush=True)
        frames.append(indata.copy())

    input_thread = threading.Thread(target=_input_thread, daemon=True)
    input_thread.start()

    print("Recording... press Enter to stop", flush=True)
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16,
        callback=_audio_callback,
    ):
        stop_event.wait()

    input_thread.join(timeout=1.0)

    if not frames:
        return b""

    audio = np.concatenate(frames, axis=0)
    return audio.tobytes()


def _save_wav(audio_bytes: bytes, path: Path) -> None:
    """Write raw PCM int16 bytes to a mono WAV file."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record local audio regression fixtures")
    parser.add_argument("--output-dir", default="tests/audio_fixtures_local", help="Where to save recordings")
    parser.add_argument("--skip-existing", action="store_true", help="Skip fixtures that already have audio.wav")
    parser.add_argument("--fixture", help="Record only this fixture")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, info in FIXTURES.items():
        if args.fixture and name != args.fixture:
            continue

        if args.skip_existing and list_take_ids(name, output_dir):
            print(f"Skipping {name} (already recorded)")
            continue

        print(f"\n=== {name} ===")
        print(f"Script: {info['script']}")
        print(f"Directions: {info['directions']}")
        print(f"Focus: {info['focus']}")
        input("Press Enter when ready to start recording...")

        audio_bytes = _record_until_enter()
        if not audio_bytes:
            print(f"No audio recorded for {name}, skipping")
            continue

        take_id = next_take_id(name, output_dir)
        audio_path = take_audio_path(name, take_id, output_dir)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        _save_wav(audio_bytes, audio_path)
        print(f"Saved {audio_path} ({len(audio_bytes) / BYTES_PER_SECOND:.1f}s)")

    print("\nNext step: capture chunks from the fixture manager GUI,")
    print("or run capture_chunks.py on each take's audio.wav.")


if __name__ == "__main__":
    main()
