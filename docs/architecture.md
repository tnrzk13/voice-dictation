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

The daemon uses **faster-whisper** (an optimized version of OpenAI's Whisper model, quantized to int8 for CPU speed). The model stays loaded in RAM between dictation sessions so you don't pay the 3-5 second load time after the first run.

When a client connects, the daemon spawns **two threads per connection**:

1. **Receiver thread** - Reads raw PCM bytes off the socket and appends them to an `audio_buffer` (a growing `bytearray`).
2. **Transcriber thread** - Every **2 seconds**, takes a snapshot of the accumulated audio buffer, converts it to float32, and runs Whisper inference on it.

### Partial vs Final results

**Partial results** are sent every 2 seconds while you're still speaking. Each partial represents Whisper's best guess at *everything said so far*. Crucially, partials can change - Whisper might hear "hello" at T=2s, then revise to "hello world" at T=4s, or even self-correct "hello ward" to "hello world".

**Final results** are sent when:
- You stop recording (the client shuts down its write side of the socket, the daemon sees EOF)
- The audio buffer exceeds 20 seconds (forces a finalization to keep memory bounded)

The protocol is newline-delimited JSON over the socket:
```
daemon -> client:  {"type": "partial", "text": "hello"}
daemon -> client:  {"type": "partial", "text": "hello world"}
daemon -> client:  {"type": "final",   "text": "hello world."}
daemon -> client:  {"type": "end",     "text": ""}
```

### The 20-second window trick

If you speak for longer than 20 seconds, the daemon finalizes all completed segments but **keeps the last segment's audio in the buffer**. This prevents the buffer from growing unboundedly while preserving context for the in-progress sentence.

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

This is the core trick. The `ProgressiveTyper` class maintains **two text buffers**:

- **`_committed`** - Finalized text. Locked in. Will never change.
- **`_pending`** - Current partial text. Temporary. Will be revised or promoted.

### How partials overwrite themselves

When a new partial arrives, the typer computes a **minimal diff** against what's already on screen:

```
Step 1: Find the longest common prefix between old pending and new pending
Step 2: Send backspaces to delete everything after the common prefix
Step 3: Type the new suffix
```

Here's a concrete example:

```
T=0s  You say: "hello world, how are you"

T=2s  Partial arrives: "hello"
      _pending was: ""
      _pending now: "Hello"     (capitalized)
      Common prefix: 0 chars
      Action: type "Hello"
      Screen: Hello|

T=4s  Partial arrives: "hello world"
      _pending was: "Hello"
      _pending now: "Hello world"
      Common prefix: "Hello" (5 chars)
      Backspaces needed: 0      (old was 5 chars, prefix is 5)
      Action: type " world"
      Screen: Hello world|

T=5s  Partial arrives: "hello ward"   (Whisper mishears momentarily)
      _pending was: "Hello world"
      _pending now: "Hello ward"
      Common prefix: "Hello w" (7 chars)
      Backspaces needed: 4      ("orld" must go)
      Action: send 4 backspaces, then type "ard"
      Screen: Hello ward|

T=6s  Partial arrives: "hello world how"  (Whisper self-corrects)
      _pending was: "Hello ward"
      _pending now: "Hello world how"
      Common prefix: "Hello w" (7 chars)
      Backspaces needed: 3      ("ard" must go)
      Action: send 3 backspaces, then type "orld how"
      Screen: Hello world how|

T=7s  You stop speaking. Final arrives: "hello world, how are you"
      _pending was: "Hello world how"
      _pending now: "Hello world, how are you "   (trailing space added)
      Common prefix: "Hello world" (11 chars)
      Backspaces needed: 4      (" how" must go)
      Action: send 4 backspaces, type ", how are you "
      Screen: Hello world, how are you |
      _committed = "Hello world, how are you "
      _pending = ""                              (reset for next dictation)
```

The key insight: **the typer never re-types text that's already correct on screen**. It only sends the minimal number of backspaces and new characters to morph the old text into the new text. This makes corrections feel nearly instant.

### Prefix stripping (handling re-transcription)

When the daemon trims its 20-second buffer and re-transcribes, the new partial includes text that was already finalized. The typer handles this with `_strip_committed_prefix()` - it compares words case-insensitively (ignoring punctuation differences) and strips any words that match the committed text, so only the genuinely new portion gets typed.

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
| Daemon transcriber thread | Runs Whisper every 2s, sends JSON results back |

## Configuration Constants (`config.py`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `SAMPLE_RATE` | 16000 | Whisper requirement |
| `BYTES_PER_SAMPLE` | 2 | int16 format |
| `TRANSCRIBE_INTERVAL` | 2s | Time between Whisper runs |
| `MAX_WINDOW_SECONDS` | 20s | Force finalize to cap buffer growth |
| `WHISPER_MODEL_SIZE` | "base" | 150MB model, ~1s latency |
| `XDOTOOL_KEYSTROKE_DELAY` | 5ms | Delay between typed characters |
| `BACKSPACE_SETTLE_DELAY` | 50ms | Pause after backspaces before typing |
| `DAEMON_STARTUP_TIMEOUT` | 10s | Max wait for model to load |

## Socket Protocol

**Client to Daemon:** Raw PCM int16 bytes (continuous stream), then EOF (shutdown write side).

**Daemon to Client:** Newline-delimited JSON with `type` field: `partial`, `final`, or `end`.
