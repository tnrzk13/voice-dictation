# Voice Dictation

Linux voice dictation - speak into your mic, text appears at your cursor. Batch and live streaming modes powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Features

- **Two modes**: Batch (transcribes in chunks) and live (real-time streaming with corrections)
- **Daemon architecture**: Whisper model stays loaded in memory for fast transcription
- **Desktop integration**: Types text directly at your cursor via `xdotool`
- **Desktop notifications**: Visual feedback for recording status

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

This installs five commands: `dictate`, `dictate-live`, `dictate-stop`, `dictate-daemon`, and `dictate-live-daemon`.

The Whisper model (~150MB) downloads automatically on first use.

## Usage

### Live mode (recommended)

Real-time streaming - text appears and self-corrects as you speak.

```bash
dictate-live
# Press Escape or close the terminal to stop
```

### Batch mode

Transcribes in chunks while you speak, or all at once after recording.

```bash
# Streaming (default) - transcribes every 5 seconds
dictate

# Non-streaming - transcribes everything after you stop
dictate --no-stream

# Press Enter to stop recording
```

### Stopping daemons

```bash
# Stop both daemons and free memory
dictate-stop

# Stop batch daemon only
dictate --stop
```

### Keyboard shortcut (optional)

Bind `dictate-live` to a shortcut in your desktop environment:

**GNOME/Ubuntu**:
Settings > Keyboard > Custom Shortcuts > Command: `dictate-live`

**i3/sway**:
```
bindsym $mod+d exec dictate-live
```

## Configuration

Edit `src/dictate/config.py` for shared settings (audio, Whisper model, batch daemon):

```python
SAMPLE_RATE = 16000          # Hz - Whisper expects 16kHz
SILENCE_THRESHOLD = 0.01     # RMS energy below this = silence
SILENCE_DURATION = 2         # seconds of silence before auto-stop
CHUNK_DURATION = 5           # seconds between streaming transcriptions
WHISPER_MODEL_SIZE = "base"  # "tiny", "small", "medium", "large-v3"
```

Edit `src/dictate/live/config.py` for live-specific settings:

```python
TRANSCRIBE_INTERVAL = 2   # seconds between transcription cycles
MAX_WINDOW_SECONDS = 20   # finalize segments when audio exceeds this
```

### Finding your silence threshold

```bash
python scripts/test-silence-threshold.py
# Speak normally, then be quiet - watch the RMS values
```

## Architecture

```
Batch mode                          Live mode
┌──────────────┐                    ┌───────────────┐
│   dictate    │                    │  dictate-live  │
│  (records,   │                    │  (streams raw  │
│  sends chunks│                    │   PCM audio)   │
└──────┬───────┘                    └───────┬────────┘
       │ Unix socket                        │ Unix socket
       v                                    v
┌──────────────┐                    ┌────────────────┐
│dictate-daemon│                    │dictate-live-   │
│  (Whisper    │                    │daemon (Whisper │
│   transcribe)│                    │ + diff-typing) │
└──────┬───────┘                    └───────┬────────┘
       │                                    │
       v                                    v
  xdotool types                       xdotool types
  at cursor                           at cursor
```

Both daemons keep the Whisper model loaded in memory - no startup delay after the first launch.

## Project Structure

```
voice-dictation/
├── src/dictate/
│   ├── config.py            # Shared configuration (audio, Whisper, batch daemon)
│   ├── daemon_support.py    # Shared daemon utilities (socket, logging, lifecycle)
│   ├── system.py            # Desktop notifications, dependency checks
│   ├── xdotool.py           # Text typing via xdotool
│   ├── stop.py              # dictate-stop command
│   ├── silence.py           # Silence detection
│   ├── batch/               # Batch dictation mode
│   │   ├── cli.py           # dictate command
│   │   ├── daemon.py        # dictate-daemon
│   │   ├── client.py        # Daemon client
│   │   └── recorder.py      # Audio recording with streaming
│   └── live/                # Live streaming dictation mode
│       ├── cli.py           # dictate-live command
│       ├── daemon.py        # dictate-live-daemon
│       ├── client.py        # Streaming daemon client
│       ├── config.py        # Live-specific configuration
│       ├── recorder.py      # Audio capture and streaming
│       ├── typer.py         # Progressive diff-based typing
│       └── keyboard_monitor.py  # Escape key detection
├── tests/
├── scripts/
├── pyproject.toml
└── README.md
```

## Troubleshooting

### Daemon won't start
```bash
# Check logs
cat ~/.local/share/voice-dictation/daemon.log       # batch
cat ~/.local/share/voice-dictation/live-daemon.log   # live

# Remove stale socket
rm /tmp/dictate-daemon.sock      # batch
rm /tmp/dictate-live-daemon.sock # live
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
- `pynput` - Keyboard monitoring (live mode)

## License

MIT License - see [LICENSE](LICENSE) for details.
