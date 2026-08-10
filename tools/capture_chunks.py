#!/usr/bin/env python3
"""Capture golden chunk sequences from audio files for regression testing.

Loads a small Whisper model and streams the audio through the live dictation
pipeline in-process. Records the partial/final JSON messages the daemon would
send so they can be replayed in tests without loading the model.

Run with:
    python tools/capture_chunks.py tests/audio_fixtures/*.wav

The captured fixtures are written to tests/audio_fixtures/<stem>/ as:
    - chunks.jsonl  : newline-delimited JSON messages from the daemon
    - reference.txt : the final transcript text (used for regression assertions)

Requires: pydub (already installed in this environment; not required at runtime)
"""

import argparse
import json
import socket
import threading
import time
from pathlib import Path
from typing import List

import pydub

from dictate.config import BYTES_PER_SAMPLE, BYTES_PER_SECOND, SAMPLE_RATE
from dictate.live.daemon import handle_client
from dictate.model_loader import load_whisper_model


def load_audio(path: str) -> bytes:
    """Load an audio file and convert to 16kHz mono int16 PCM."""
    audio = pydub.AudioSegment.from_file(path)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(BYTES_PER_SAMPLE)
    return audio.raw_data


def capture_chunks(audio_bytes: bytes, model_size: str, device: str, compute_type: str) -> List[dict]:
    """Stream audio through the transcription pipeline and return captured messages."""
    model = load_whisper_model(model_size, device, compute_type, quiet=True)

    client_sock, daemon_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    chunks: List[dict] = []
    receiver_done = threading.Event()

    def receiver() -> None:
        buf = b""
        try:
            while True:
                data = client_sock.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    chunks.append(msg)
                    if msg.get("type") == "end":
                        receiver_done.set()
                        return
        except OSError:
            pass

    recv_thread = threading.Thread(target=receiver, daemon=True)
    recv_thread.start()

    daemon_thread = threading.Thread(target=handle_client, args=(daemon_sock, model), daemon=True)
    daemon_thread.start()

    # Stream audio at roughly real-time (16 kHz, 16-bit mono) to capture
    # realistic partial/final chunk sequences including re-transcriptions.
    frame_size = 8000
    frame_duration = frame_size / BYTES_PER_SECOND
    for i in range(0, len(audio_bytes), frame_size):
        client_sock.send(audio_bytes[i : i + frame_size])
        time.sleep(frame_duration)

    client_sock.shutdown(socket.SHUT_WR)
    daemon_thread.join(timeout=120)
    receiver_done.wait(timeout=30)

    client_sock.close()
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture chunk sequences from audio files")
    parser.add_argument("audio", nargs="+", help="Audio file(s) to process")
    parser.add_argument("--model", default="tiny", help="Whisper model size to use")
    parser.add_argument("--device", default="cpu", help="Compute device")
    parser.add_argument("--compute-type", default="int8", help="Model precision")
    parser.add_argument("--output-dir", default="tests/audio_fixtures", help="Directory to write fixtures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in args.audio:
        audio_path = Path(audio_path)
        audio_bytes = load_audio(str(audio_path))
        chunks = capture_chunks(audio_bytes, args.model, args.device, args.compute_type)

        # If the audio file is named audio.wav inside a fixture directory, use
        # the parent directory name as the fixture name. Otherwise fall back to
        # the audio file stem (e.g., hello_world.wav -> hello_world).
        if audio_path.name == "audio.wav" and audio_path.parent.name != output_dir.name:
            fixture_name = audio_path.parent.name
            fixture_dir = audio_path.parent
        else:
            fixture_name = audio_path.stem
            fixture_dir = output_dir / fixture_name
        fixture_dir.mkdir(parents=True, exist_ok=True)

        (fixture_dir / "chunks.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in chunks),
            encoding="utf-8",
        )

        final = next((c for c in reversed(chunks) if c["type"] == "final"), None)
        partial = next((c for c in reversed(chunks) if c["type"] == "partial"), None)
        reference = (final or partial or {}).get("text", "").strip()

        (fixture_dir / "reference.txt").write_text(reference, encoding="utf-8")

        print(f"Captured {len(chunks)} messages for {fixture_name}")
        print(f"  Reference: {reference}")
        print(f"  Written to {fixture_dir}")


if __name__ == "__main__":
    main()
