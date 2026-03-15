# Voice Dictation

Linux voice dictation - speak into your mic, text appears at your cursor. Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with real-time streaming transcription.

## Features

- **Real-time streaming**: Words appear and self-correct as you speak
- **Spoken formatting commands**: Say "slash", "comma", "open parenthesis", etc. to insert symbols
- **Daemon architecture**: Whisper model stays loaded in memory for fast transcription
- **Desktop integration**: Types text directly at your cursor via `xdotool`
- **Desktop notifications**: Visual feedback for daemon loading and errors

## System Requirements

- **OS**: Linux with X11 (tested on Debian/Ubuntu)
- **Python**: 3.8+
- **System tools**: `xdotool`, `notify-send` (libnotify)

```bash
# Debian/Ubuntu
sudo apt-get install xdotool libnotify-bin

# Fedora
sudo dnf install xdotool libnotify

# Arch
sudo pacman -S xdotool libnotify
```

## Installation

```bash
git clone https://github.com/tnrzk13/voice-dictation.git
cd voice-dictation
pip install -e .
```

This installs three commands: `dictate`, `dictate-daemon`, and `dictate-stop`.

The Whisper model (~150MB) downloads automatically on first use.

## Usage

```bash
# Start dictation - text streams in real-time
dictate

# Non-streaming - text appears only after you stop recording
dictate --no-stream

# Press Enter or Escape to stop recording
```

### Stopping the daemon

```bash
# Stop daemon and free memory
dictate-stop

# Or
dictate --stop
```

### Keyboard shortcut (optional)

Bind `dictate` to a shortcut in your desktop environment:

**GNOME/Ubuntu**:
Settings > Keyboard > Custom Shortcuts > Command: `dictate`

**i3/sway**:
```
bindsym $mod+d exec dictate
```

## Spoken Commands

Say these phrases while dictating to insert symbols and formatting:

| Category | Say | Get |
|----------|-----|-----|
| **Punctuation** | "period", "comma", "question mark", "exclamation mark" | `. , ? !` |
| **Brackets** | "open parenthesis ... close parenthesis" | `(...)` |
| **Quotes** | "open quote ... close quote" | `"..."` |
| **Path separators** | "slash", "backslash", "hyphen", "underscore" | `/ \ - _` |
| **Newlines** | "new line", "new paragraph", "tab key" | line break, double break, tab |
| **Programming** | "equals sign", "plus sign", "at sign", "hash sign" | `= + @ #` |

Commands are case-insensitive and work in real-time as you speak. Whisper auto-punctuation (periods, commas) is deduplicated so saying "period" after a natural pause won't produce a double period.

Full command list: `src/dictate/live/formatting.py`

## Configuration

Edit `src/dictate/config.py` for shared settings:

```python
SAMPLE_RATE = 16000          # Hz - Whisper expects 16kHz
WHISPER_MODEL_SIZE = "base"  # "tiny", "small", "medium", "large-v3"
```

Streaming settings are also in `src/dictate/config.py`:

```python
TRANSCRIBE_INTERVAL = 2   # seconds between transcription cycles
MAX_WINDOW_SECONDS = 20   # finalize segments when audio exceeds this
WHISPER_HOTWORDS = "Claude Code"  # space-separated terms Whisper often mishears
```

## Architecture

```
┌──────────┐
│  dictate  │
│ (streams  │
│ raw PCM   │
│  audio)   │
└─────┬─────┘
      │ Unix socket
      v
┌──────────────┐
│dictate-daemon│
│  (Whisper +  │
│ diff-typing) │
└─────┬────────┘
      │
      v
 xdotool types
 at cursor
```

The daemon keeps the Whisper model loaded in memory - no startup delay after the first launch.

## Project Structure

```
voice-dictation/
├── src/dictate/
│   ├── config.py            # Shared configuration (audio, Whisper)
│   ├── daemon_support.py    # Shared daemon utilities (socket, logging, lifecycle)
│   ├── system.py            # Desktop notifications, dependency checks
│   ├── xdotool.py           # Text typing via xdotool
│   ├── stop.py              # dictate-stop command
│   └── live/                # Streaming dictation
│       ├── cli.py           # dictate command
│       ├── daemon.py        # dictate-daemon
│       ├── client.py        # Streaming daemon client
│       ├── recorder.py      # Audio capture and streaming
│       ├── typer.py         # Progressive diff-based typing
│       ├── formatting.py    # Spoken command formatting (slash, comma, etc.)
│       └── keyboard_monitor.py  # Keystroke detection to stop dictation
├── tests/
├── scripts/
├── pyproject.toml
└── README.md
```

## Troubleshooting

### Daemon won't start
```bash
# Check logs
cat ~/.local/share/voice-dictation/live-daemon.log

# Remove stale socket
rm /tmp/dictate-live-daemon.sock
```

### Text not appearing
- Ensure `xdotool` is installed: `which xdotool`
- Check that you have an active window where text can be typed
- X11 required - Wayland is not supported

### Model download issues
The first run downloads the Whisper "base" model (~150MB). If it fails:
```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
```

## Dependencies

- `numpy` - Audio processing
- `sounddevice` - Microphone input
- `faster-whisper` - Whisper model inference (CTranslate2, CPU-optimized)
- `pynput` - Keyboard monitoring

## License

MIT License - see [LICENSE](LICENSE) for details.
