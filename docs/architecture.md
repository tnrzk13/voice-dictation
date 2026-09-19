# Voice Dictation - Technical Architecture

## The Big Picture

Three processes work together:

1. **Daemon** (`dictate-daemon`) - Long-lived background process that holds the Whisper AI model in memory. Accepts audio, returns transcriptions over a Unix socket.
2. **Client** (`dictate`) - Short-lived process started each time you dictate. Captures audio, streams it to the daemon, receives transcription results, and types them on screen.
3. **xdotool** - External X11 utility called as a subprocess to simulate keystrokes.

## Phase 1: Audio Capture (`recorder.py`)

Uses the `sounddevice` library (a Python wrapper around PortAudio). The microphone stream is configured at:
- **16,000 Hz** sample rate (what Whisper requires)
- **Mono** (1 channel)
- **int16** (2 bytes per sample) - so 32,000 bytes/second of raw PCM data

Audio arrives via a **callback function** on a dedicated thread managed by sounddevice. Each chunk of audio frames is immediately converted to bytes and **sent raw over the Unix socket** to the daemon. There's no buffering on the client side - it's a direct pipe from microphone to daemon.

## Phase 2: Speech Recognition (`daemon.py`)

The daemon uses **faster-whisper** (an optimized version of OpenAI's Whisper model). The default runs on GPU with `int8_float16` precision; CPU-only mode uses `int8`. The model stays loaded in memory between dictation sessions so you don't pay the load time after the first run.

Each client connection is handled in its own daemon thread, so the main thread can accept new connections while others are transcribing. Each connection then spawns **two worker threads**:

1. **Receiver thread** - Reads raw PCM bytes off the socket and appends them to an `_AudioBuffer` (a growing `bytearray` plus a `Condition` that wakes the transcriber), then signals the transcriber. If the model falls behind, the buffer is capped at 60 seconds and the oldest audio is dropped to prevent unbounded memory growth.
2. **Transcriber thread** - Wakes when enough *new* audio has accumulated (0.3s by default), or after **2 seconds** as a floor, takes a snapshot of the accumulated audio buffer, converts it to float32, and runs Whisper inference on it.

The length gate counts bytes received since the last snapshot, not raw buffer length - the buffer always retains a context tail (see the commit policy), so gating on buffer size would fire immediately every cycle. The timer floor keeps sparse or silent audio from stalling transcription; on continuous speech the gate dominates, producing partials roughly once per second instead of once per two.

### Partial vs Final results

**Partial results** are sent each transcription cycle while you're still speaking - as soon as 0.3s of new audio accumulates, or every 2s at the floor. Each partial represents Whisper's best guess at *everything said so far*. Crucially, partials can change - Whisper might hear "hello" at T=1s, then revise to "hello world" at T=2s, or even self-correct "hello ward" to "hello world".

**Final results** are sent when you stop recording (the client shuts down its write side of the socket, the daemon sees EOF). The final is not the last partial: the daemon re-decodes the whole utterance with full context (see "The final pass" below).

Completed segments are finalized continuously (see "The commit policy" below), not only at the end of a session.

### The final pass

Streaming finalizes chunks before Whisper has the rest of the sentence, so commas and sentence-final periods are lost. At EOF the daemon re-transcribes all of the session's audio in one decode, which gives Whisper the full context to punctuate and capitalize. It is primed with `FINAL_PUNCTUATION_PROMPT` because the decode settings tuned for streaming each suppress terminal punctuation:

- `vad_filter=False` - trimming trailing silence removes the cue for a sentence end
- `repetition_penalty=1.0`, `no_repeat_ngram_size=0` - these penalize the already-seen period token
- a punctuated seed - a bare hotword list primes the decoder to omit the closing period

Hotwords stay enabled so domain terms survive. Sessions longer than `MAX_SESSION_SECONDS` skip the re-decode and fall back to the last partial, bounding memory and final-pass latency.

The client treats a final differently once stopped. While the session is live it applies the full final (in-place punctuation). After a stop key it calls `apply_final_trailing`, which appends only the closing sentence mark and never backspaces: the cursor may have moved, so a mid-text rewrite could corrupt the text.

The protocol is newline-delimited JSON over the socket. Each partial's `text` is cumulative and carries `finalized`, the stable prefix the daemon has already committed:
```
daemon -> client:  {"type": "partial", "text": "hello", "finalized": ""}
daemon -> client:  {"type": "partial", "text": "hello world", "finalized": "hello"}
daemon -> client:  {"type": "final",   "text": "hello world."}
daemon -> client:  {"type": "end",     "text": ""}
```

### The commit policy

The daemon finalizes completed segments (bounded by silence) every cycle and keeps only the last segment's audio in the buffer. For continuous speech with no silence, it finalizes all but the last `KEEP_TAIL_SECONDS` (3s) of speech and trims the buffer to the start of the first kept word. The split comes from Whisper's **per-word timestamps** (`word_timestamps=True`), so pauses and speaking rate cannot skew it the way a word-count-over-duration estimate would. If timings are missing or misaligned (for example after repetition collapse rewrites a segment), the daemon defers finalization rather than guess.

After a trim the buffer starts mid-sentence, so each transcription is primed with `initial_prompt` set to the finalized text. This gives the decoder the preceding context; without it, re-decoding a contextless tail makes Whisper drop or invent the boundary words.

Each partial carries this finalized prefix so the client locks exactly the text the daemon will not revise, which is what makes low transcribe gates safe (see Phase 4).

## Phase 3: Text Formatting (`formatting.py`)

Before text reaches the screen, it passes through a formatting pipeline that converts spoken commands to symbols. There are **119 commands** across categories:

- "slash" becomes `/`, "period" becomes `.`, "new line" becomes `\n`
- "open parenthesis" becomes `(`, "close bracket" becomes `]`
- Programming symbols: "equals sign", "pipe", "tilde", etc.

Each command has a **spacing rule**:
- `REMOVE_BEFORE`: punctuation attaches to the previous word ("hello period" becomes "hello.")
- `REMOVE_AFTER`: opening brackets attach to the next word
- `REMOVE_BOTH`: path separators have no spaces ("tony slash pictures" becomes "tony/pictures")
- `DEFAULT`: normal spacing around operators

There's also punctuation deduplication - if Whisper auto-inserts a period AND you say "period", you get one period, not two.

## Phase 4: The Overwrite Mechanism (`typer.py`)

This is the core trick. The `ProgressiveTyper` class tracks:

- **`_committed`** - the prefix the daemon has finalized. Locked in.
- **`_pending`** - the still-revisable tail.

The commit boundary is not guessed client-side. `apply_partial()` takes the daemon's `finalized` string and uses the common prefix with it as `_committed`; because the daemon only finalizes words with enough following speech that Whisper stops revising them, that text is genuinely immutable. An earlier client-side stability heuristic committed words the daemon had not finalized yet, so a later Whisper revision (for example dropping a filler word) shifted the committed prefix and the whole partial was retyped - visible duplication.

### How partials overwrite themselves

When a new partial arrives, the typer computes a **minimal diff between the full screen and the formatted partial**:

```
Step 1: Format the cumulative partial (commands + capitalization)
Step 2: Find the longest common prefix between it and the current screen
Step 3: Send backspaces to delete the divergent suffix
Step 4: Type the new suffix
```

Diffing against the whole screen (not just `_pending`) means a revision that reaches into already-visible text corrects in place, instead of appending a second copy.

Here's a concrete example:

```
T=0s  You say: "hello world, how are you"

T=2s  Partial arrives: "hello"
      screen was: ""
      target now: "Hello"       (capitalized, finalized="")
      Common prefix: 0 chars
      Action: type "Hello"
      Screen: Hello|

T=4s  Partial arrives: "hello world"  (finalized="hello")
      screen was: "Hello"
      target now: "Hello world"
      Common prefix: "Hello" (5 chars)
      Backspaces needed: 0      (old was 5 chars, prefix is 5)
      Action: type " world"
      Screen: Hello world|
      _committed = "Hello"
      _pending = " world"

T=5s  Partial arrives: "hello ward"   (Whisper mishears momentarily)
      screen was: "Hello world"
      target now: "Hello ward"
      Common prefix: "Hello w" (7 chars)
      Backspaces needed: 4      ("orld" must go)
      Action: send 4 backspaces, then type "ard"
      Screen: Hello ward|

T=6s  Partial arrives: "hello world how"  (Whisper self-corrects)
      screen was: "Hello ward"
      target now: "Hello world how"
      Common prefix: "Hello w" (7 chars)
      Backspaces needed: 3      ("ard" must go)
      Action: send 3 backspaces, then type "orld how"
      Screen: Hello world how|

T=7s  You stop speaking. Final arrives: "hello world, how are you"
      screen was: "Hello world how"
      target now: "Hello world, how are you "   (trailing space added)
      Common prefix: "Hello world" (11 chars)
      Backspaces needed: 4      (" how" must go)
      Action: send 4 backspaces, type ", how are you "
      Screen: Hello world, how are you |
      _committed = "Hello world, how are you "
      _pending = ""
```

The key insight: **the typer never re-types text that's already correct on screen**. It only sends the minimal number of backspaces and new characters to morph the old screen into the new target. This makes corrections feel nearly instant.

## Phase 5: Keystroke Simulation (`xdotool.py`)

Text is physically typed using **xdotool**, an X11 automation tool:

- Regular text: `xdotool type --delay 5 "text"` (5ms between characters)
- Backspaces: `xdotool key --delay 0 BackSpace BackSpace ...` (no delay, fast deletion)
- Special keys: newlines become `xdotool key Return`, tabs become `xdotool key Tab`

There's a **50ms settle delay** (`BACKSPACE_SETTLE_DELAY`) between sending backspaces and typing new text, giving the target application time to process the deletions before new characters arrive.

## Phase 6: Stop Detection (`input_monitor.py`)

An `InputMonitor` runs two background threads using `pynput` - one for keyboard, one for mouse. Recording stops when:

- **Enter key** - Bypasses the xdotool grace period for instant response, but still respects the `is_typing` guard (xdotool generates Return when formatting commands like "new line" are spoken).
- **Mouse click** - Any button stops dictation immediately. A 1-second startup grace period prevents the click used to focus the target window from triggering an early stop.
- **Any other key** - Stops dictation, but only after passing both the `is_typing` check and the 0.5-second xdotool grace period to filter out synthetic keypresses.

## Threading Model

| Thread | Purpose |
|--------|---------|
| Main thread | Blocks on `stop_event.wait()` until recording should stop |
| sounddevice callback thread | Receives audio frames, sends to daemon socket |
| Keyboard listener thread | Listens for physical keypresses via pynput |
| Mouse listener thread | Listens for mouse clicks via pynput |
| Client receiver thread | Reads JSON from daemon, calls `typer.apply_partial()` / `apply_final()` |
| Daemon receiver thread | Reads raw PCM bytes from socket into `audio_buffer` |
| Daemon transcriber thread | Runs Whisper when 0.3s of new audio arrives (2s floor), sends JSON results back |

## Configuration Constants (`config.py`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `SAMPLE_RATE` | 16000 | Whisper requirement |
| `BYTES_PER_SAMPLE` | 2 | int16 format |
| `TRANSCRIBE_INTERVAL` | 2s | Floor between Whisper runs when new audio is sparse |
| `TRANSCRIBE_MIN_AUDIO_SECONDS` | 0.3s | New audio that triggers an early Whisper run |
| `MAX_SESSION_SECONDS` | 60s | Audio kept for the full-context final; longer sessions fall back |
| `FINAL_PUNCTUATION_PROMPT` | (sentence) | Punctuation-style seed for the final pass |
| `KEEP_TAIL_SECONDS` | 3s | Audio kept for context when finalizing a continuous segment |
| `WHISPER_MODEL_SIZE` | "large-v3-turbo" | Default model, ~1.6 GB download |
| `XDOTOOL_KEYSTROKE_DELAY` | 12ms | Delay between typed characters |
| `BACKSPACE_SETTLE_DELAY` | 50ms | Pause after backspaces before typing |
| `DAEMON_STARTUP_TIMEOUT` | 10s | Max wait for model to load |

## Socket Protocol

**Client to Daemon:** Raw PCM int16 bytes (continuous stream), then EOF (shutdown write side).

**Daemon to Client:** Newline-delimited JSON with `type` field: `partial`, `final`, or `end`.
