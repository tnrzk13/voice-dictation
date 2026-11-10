# Voice Dictation benchmark

Measured 2026-11-05 against the fixtures in `tests/audio_fixtures/` and the
gitignored `tests/audio_fixtures_local/`. Four numbers are reported: real-time
factor, gate-to-partial latency, word error rate, and GPU memory.

The harness is `tools/benchmark.py`. Raw results are in
`tools/benchmark-results.json` (committed fixtures) and
`tools/benchmark-results-local.json` (committed plus local takes).

> 2026-11-09: the recorded numbers predate the streaming glossary-prompt change; they were measured with 12 ms per-character typing and the bare hotword prompt. The current code types at 30 ms per character with a 3 ms backspace delay and seeds the session-start glossary instead.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, 8188 MiB |
| Driver / CUDA | 535.261.03 / 12.2 |
| OS | Linux 6.1.0-54-amd64, X11 session with Xorg and gnome-shell resident |
| Python | 3.11.9 |
| faster-whisper | 1.2.1 |
| CTranslate2 | 4.6.2 |
| numpy | 1.26.4 |
| Model | large-v3-turbo (`mobiuslabsgmbh/faster-whisper-large-v3-turbo`), cached locally |
| Model files | 1,621,665,983 bytes (1.51 GiB); `model.bin` 1,618,849,929 bytes (1.51 GiB) |
| Device / precision | `cuda` / `int8_float16` |
| Decode settings | beam 1, temperature 0.0, VAD on with 500 ms min silence, repetition penalty 1.2, no-repeat ngram 3 |
| Hotwords | `hotwords.txt` present; 10 terms (not listed here) |
| Streaming | 0.3 s new-audio gate, 2 s floor, 3 s keep-tail, 60 s session cap |
| Feed | 16 kHz mono int16 over a Unix socketpair, 64 ms (1024 frame) writes, real-time pacing |

Model load took 2.2 s from the local page cache. Download time is not included.

## Method

`tools/benchmark.py` replays each fixture through the daemon's own
`handle_client` over a socketpair, the same entry point used by
`tools/capture_chunks.py`. It loads the real model with
`load_whisper_model`, streams PCM at wall-clock audio speed, and never
reimplements the model call. A proxy around `WhisperModel.transcribe` records
the wall time and input audio length of every decode, including the final
full-context decode.

- **WER**: lowercase, replace non-word characters with spaces, collapse
  whitespace, then word-level Levenshtein alignment. Substitutions, deletions,
  and insertions are reported separately. `reference.txt` is the captured
  transcript shipped with each fixture. Script WER uses the known fixture
  script (`hello_world` maps to `simple_sentence`, `pause_and_continue` maps to
  `mid_sentence_pause`).
- **Visible WER**: the same word-level comparison against the text of the last
  partial emitted before EOF, using the same reference and script as the WER
  above. Both WERs are reported because the client drops every message after the
  stop key, so the final pass never types, and the gap between visible and final
  WER is the size of that discarded correction.
- **Visible backspaces**: backspaces accumulated up to and including that last
  partial, excluding anything the `final` message would cause, per 100 words.
- **Revisions per word**: partial revisions divided by words in the final
  transcript.
- **Word commit delay**: from a word's end timestamp to the moment the partial
  containing it is sent.
- **RTF**: total decode wall time divided by fixture audio duration. Per-decode
  RTF is decode wall time divided by the audio length of that decode. The live
  loop re-decodes overlapping context, so total RTF intentionally counts
  repeated audio work; `decode_audio_amplification` reports how much audio was
  decoded relative to the session.
- **Latency**: from the append that crosses the 0.3 s watermark to the moment
  the partial message is handed to the socket. It excludes the 0.3 s of audio
  collected before the gate and excludes typing. The first-partial figure is
  measured from the start of audio.
- **Percentiles**: nearest-rank.
- **VRAM**: `nvidia-smi --query-gpu=memory.used` every 100 ms. Peak and median
  are taken over samples that fall inside decode intervals. The baseline is
  sampled before the model is loaded. This is whole-GPU memory, not
  model-only memory.
- **Rounding**: the harness prints WER as a three-decimal ratio, latency in
  whole milliseconds, RTF to three decimals and VRAM in whole MiB. Tables round
  those printed values, RTF to two decimals; the JSON keeps raw values.

The recorded result JSONs predate these columns; rerun the harness to populate
them.

Two runs are reported per command. Run 1 is cold for the process (first CUDA
decode pays kernel initialization); run 2 is warm. The model stays loaded
between runs.

## Results: committed fixtures

Three fixtures, 10.65 s of audio, 24 words of script and 24 words of reference.

| Metric | Run 1 | Run 2 |
|---|---|---|
| WER vs reference | 0 errors / 24 words (0.0%) | 0 / 24 (0.0%) |
| WER vs script | 0 errors / 24 words (0.0%) | 0 / 24 (0.0%) |
| RTF total | 0.71 | 0.69 |
| Per-decode RTF p50 / p95 | 0.11 / 0.67 | 0.11 / 0.63 |
| Decode wall / audio / amplification | 7.56 s / 10.65 s / 6.17x | 7.33 s / 10.65 s / 6.12x |
| Gate-to-partial p50 / p95 / max | 211 / 313 / 431 ms | 208 / 218 / 223 ms |
| First partial after audio start (median) | 469 ms | 456 ms |
| Partial messages / revisions | 22 / 19 | 22 / 19 |
| Backspaces per 100 words | 562 | 579 |
| Finalized prefix regressions | 0 | 0 |
| VRAM baseline / peak / median | 1510 / 2712 / 2712 MiB | 1510 / 2712 / 2712 MiB |
| VRAM delta over baseline, peak / median | 1202 / 1202 MiB | 1202 / 1202 MiB |

Per fixture, with run 2 values for RTF and latency and run 1 in parentheses:

| Fixture | Audio | WER ref | WER script | RTF | Latency p95 | Revisions | Backspaces / 100 words | Regressions |
|---|---|---|---|---|---|---|---|---|
| formatting_commands | 3.02 s | 0/7 | 0/7 | 0.69 (0.77) | 218 ms (431) | 8 | 743 (686) | 0 |
| hello_world | 2.66 s | 0/6 | 0/6 | 0.70 (0.70) | 213 ms (219) | 6 | 733 (733) | 0 |
| pause_and_continue | 4.96 s | 0/11 | 0/11 | 0.68 (0.68) | 223 ms (225) | 5 | 391 (391) | 0 |

The first fixture in run 1 includes the cold first decode: 687 ms to first
partial and a 431 ms worst gate-to-partial. Run 2 removes that effect.

## Results: local takes

Twelve real-voice takes, 102.76 s of audio, 148 reference words. These are
supplementary: they are gitignored, one speaker, one take per script, and their
`reference.txt` files were captured earlier and contain visible recognition
errors (for example misspelled product names and `.Dite`). WER against those references is
agreement with an older capture, not accuracy.

| Metric | Run 1 | Run 2 |
|---|---|---|
| WER vs reference | 11 errors / 148 words (7.4%) | 11 / 148 (7.4%) |
| WER vs script, all included fixtures | 15 errors / 149 words (10.1%) | 15 / 149 (10.1%) |
| WER vs script, excluding numbers_and_symbols | 3 errors / 131 words (2.3%) | 3 / 131 (2.3%) |
| RTF total | 0.63 | 0.63 |
| Per-decode RTF p50 / p95 | 0.06 / 0.14 | 0.06 / 0.13 |
| Decode wall / audio / amplification | 64.65 s / 102.76 s / 11.19x | 64.87 s / 102.76 s / 11.19x |
| Gate-to-partial p50 / p95 / max | 215 / 229 / 239 ms | 215 / 233 / 249 ms |
| First partial after audio start (median) | 1427 ms | 1429 ms |
| Partial messages / revisions | 171 / 159 | 171 / 159 |
| Backspaces per 100 words | 1200 | 1200 |
| Finalized prefix regressions | 0 | 0 |
| VRAM baseline / peak / median | 1510 / 2712 / 2711 MiB | 1510 / 2668 / 2645 MiB |
| VRAM delta over baseline, peak / median | 1202 / 1201 MiB | 1158 / 1135 MiB |

Per fixture (run 2):

| Fixture | Audio | WER ref | WER script | RTF | Latency p95 | Revisions | Backspaces / 100 words |
|---|---|---|---|---|---|---|---|
| code_snippet | 11.18 s | 3/17 | 0/17 | 0.66 | 225 ms | 24 | 577 |
| dictation_test_phrase | 8.92 s | 0/17 | 0/17 | 0.63 | 229 ms | 12 | 747 |
| false_start | 5.24 s | 2/9 | 0/9 | 0.53 | 220 ms | 3 | 178 |
| filler_words | 6.95 s | 1/9 | 2/10 | 0.59 | 221 ms | 6 | 1425 |
| formatting_commands | 5.24 s | 0/7 | 0/7 | 0.61 | 218 ms | 8 | 486 |
| hotwords | 12.23 s | 2/18 | 0/19 | 0.69 | 239 ms | 29 | 2521 |
| long_sentence | 8.97 s | 0/16 | 0/16 | 0.63 | 238 ms | 17 | 1500 |
| mid_sentence_pause | 7.38 s | 0/11 | 0/11 | 0.58 | 226 ms | 7 | 546 |
| numbers_and_symbols | 9.31 s | 1/10 | 12/18 | 0.66 | 244 ms | 16 | 1020 |
| punctuation_variety | 14.14 s | 2/19 | 1/19 | 0.63 | 230 ms | 22 | 1826 |
| repetition_recovery | 9.16 s | 0/9 | excluded | 0.67 | 220 ms | 12 | 1289 |
| simple_sentence | 4.05 s | 0/6 | 0/6 | 0.53 | 216 ms | 3 | 717 |

All local rows have zero finalized prefix regressions. Local takes begin with
lead-in silence, so their first-partial median includes the wait for speech and
is not comparable to the committed set's 456 to 469 ms.

`numbers_and_symbols` accounts for 12 of the 15 script word errors. The script
writes numbers as words ("forty two dollars and fifty cents", "8, 4, 7, 2",
"one percent"); the transcript writes them as symbols ("$42.50", "8472", "1%").
Word-level WER charges that formatting convention as substitutions, so the
10.1% aggregate is not a recognition-error rate. The 2.3% figure removes only
that fixture and keeps every other take and error.

`filler_words` contributes 2 deletions because the script contains "um" and
"uh" and the transcript omits them. `repetition_recovery` is excluded from
script WER because its spoken audio contains extra repeated words not present
in its script.

## Correction behaviour

Across both runs the daemon emitted 22 partials for 24 committed words and
171 partials for 148 local words. Every partial after the first changed the
text (19 revisions committed, 159 local). The typer sent 135 to 139 backspaces
on the committed set and 1776 on the local set, which is 562 to 579 and 1200
backspaces per 100 words.

No committed prefix regressed in any run: no partial's `finalized` string
failed to extend the previous one, and the typer's committed prefix was
monotonic at every message. The final message was compared with the last
partial separately. On the committed set it changed only punctuation and
capitalization; on the local set it changed words in 4 of 12 takes
(`code_snippet`, `filler_words`, `hotwords`, `punctuation_variety`).

Backspace counts include every revision the typer would send through
`xdotool`. They are keypress counts, not a measure of visible flicker, and
typing is mocked.

## GPU memory

The committed runs are the cleaner VRAM sample: baseline 1510 MiB, peak and
median during decode 2712 MiB, so the model and its CUDA workspace add about
1.2 GiB on top of the desktop baseline. The local run peaked at 2712 MiB in
run 1 and 2668 MiB in run 2 with the same baseline, with medians of 2711 and
2645 MiB. The repeatable delta is therefore 1.16 to 1.20 GiB.

The weights are 1.51 GiB on disk. The measured delta is consistent with
int8 weights plus CUDA context and workspace; it is not a model-only
attribution.

## Limitations

- **Sample size.** Three committed fixtures and 24 script words are
  indicative, not a benchmark suite. One word error moves the committed WER by
  4.2 percentage points. The twelve local takes add 102.76 s and 131 to 149
  script words, still from one speaker with one take per script.
- **Synthetic audio in the committed set.** The three committed fixtures are
  gTTS speech, not human recordings. The local takes are human but are not
  committed and cannot be reproduced from a fresh clone.
- **References are not human annotation.** `reference.txt` is a previous
  model capture. Local references contain recognition errors. The script is
  the only human-authored ground truth, and it has the number-format and
  filler caveats above.
- **Typing is excluded.** `xdotool` calls are mocked, so latency ends at the
  socket. Screen latency adds the configured 12 ms per typed character plus
  X11 and target-application processing. Backspace counts are keystrokes, not
  observed screen updates.
- **Partial path only.** The benchmark exercises the daemon and the typer's
  diff logic, not the microphone, PortAudio, client JSON parsing, or the
  keyboard stop path. Audio is fed at a synthetic 64 ms cadence.
- **Short utterances.** Audio ranges from 2.66 s to 14.14 s. The 60 s session
  cap, long continuous speech, and the final-pass fallback for long sessions
  are untested.
- **One configuration.** One GPU, one model, one precision, English only, one
  hotword list. Results do not transfer to CPU, other GPUs, other models, or
  different hotwords.
- **Shared GPU.** The desktop used 1510 MiB before load and other processes can
  allocate during a run. VRAM is whole-GPU memory sampled at 10 Hz, so short
  peaks can be missed. The repeated 2712 MiB peak across two committed runs and
  one local run supports a delta of about 1.2 GiB; the local run 2 median of
  2645 MiB shows some drift.
- **Sampling overhead.** `nvidia-smi` polling every 100 ms consumes CPU during
  decode and is included in the timings. A run without VRAM sampling would be
  marginally faster.
- **Two runs, no intervals.** Run-to-run variance is visible: the committed
  gate-to-partial p95 was 313 ms in run 1 and 218 ms in run 2, the cold first
  decode took 687 ms in run 1 versus 456 ms in run 2, and the local VRAM peak
  moved from 2712 to 2668 MiB. No confidence intervals are claimed.
- **Final message vs typed text.** WER uses the daemon's `final` message. With
  the default stop-key flow the client can suppress messages after the stop
  gesture, so the punctuated final may not be typed (`docs/architecture.md`).
  Word-level WER after punctuation stripping was unchanged by the final pass
  on the committed fixtures, but the final changed words relative to the last
  partial in 4 of 12 local takes. Under the default stop flow those word
  changes may not reach the screen, so the local WER figures are an upper
  bound on what a user would see in those four cases.

## What would make these numbers stronger

1. **Human-annotated transcripts for the twelve local takes.** Replace the
   older model captures with checked transcripts. This removes the 7.4%
   agreement figure and the number-format ambiguity for the largest sample,
   and costs one listening pass per take.
2. **More and longer fixtures.** Add twenty or more utterances, including
   30 to 60 s of continuous speech and a second speaker. The harness already
   replays any 16 kHz mono WAV, so the cost is recording and transcription,
   not tooling.
3. **End-to-end screen latency.** Run the real client against a test input
   field and timestamp the first visible character, or instrument the client
   around `xdotool`. This closes the mocked-typing gap and is the only way to
   confirm the 12 ms per character estimate.

## Reproduce

From the repository root:

```bash
python3 tools/benchmark.py --runs 2
python3 tools/benchmark.py --runs 2 --include-local --output tools/benchmark-results-local.json
```

The first command writes `tools/benchmark-results.json`; the second writes
`tools/benchmark-results-local.json`. Both print the tables above. `--fixture`
filters by fixture name and `--model`, `--device`, and `--compute-type`
override the defaults.
