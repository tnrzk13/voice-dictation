# Voice Dictation - Project Guide

## What this is

Live dictation. A long-lived Whisper daemon (`dictate-daemon`) transcribes streamed
16 kHz PCM over a Unix socket; a short-lived client (`dictate`) types results with
xdotool. Pipeline details live in `docs/architecture.md`.

## Testing

Run before presenting any work:

    python -m pytest tests/ -q

Audio behavior (daemon trim/re-decode, typer diffs, formatting) has recorded
fixtures on top of unit tests:

- `tests/audio_fixtures/` - committed golden fixtures, replayed without the model.
- `tests/audio_fixtures_local/` - your own recordings, one or more takes per
  fixture at `<fixture>/takes/<take_id>/`. Gitignored.
- `tools/fixture_manager.py` - GUI to record takes, play, capture chunks, delete.

When changing anything under `daemon`, `typer`, or `formatting`:

1. `python -m pytest tests/test_audio_regression.py -q`
2. Re-capture affected takes in the GUI (`Capture Chunks`); it uses the daemon's
   configured model, so captures match production.
3. Fixtures marked `verify_against_script` are also checked against ground truth;
   the test fails if a capture loses or invents words.

Add new edge cases in `tools/fixture_definitions.py`. Set
`"verify_against_script": True` only when the model reliably transcribes the
script, so the reference can be compared to it.
