# Voice Dictation

Real-time voice dictation with streaming transcription using OpenAI's Whisper model. Speak naturally into your microphone, and watch your words appear at the cursor position!

## Features

- **Automatic Pause Detection**: Stops recording after 4 seconds of silence
- **Streaming Transcription**: Text appears every 5 seconds while you speak (great for long sessions)
- **Daemon Architecture**: Keeps Whisper model loaded in memory for fast, low-latency transcription
- **Non-streaming Mode**: Option to transcribe everything at once after recording
- **Desktop Integration**: Uses `xdotool` to type text directly at your cursor position
- **Desktop Notifications**: Visual feedback for recording status

## System Requirements

- **OS**: Linux with X11 (tested on Debian/Ubuntu)
- **Python**: 3.8 or higher
- **Microphone**: Any working audio input device
- **System Tools**:
  - `xdotool` - for typing text at cursor
  - `notify-send` - for desktop notifications (libnotify)

## Installation

### 1. Install System Dependencies

```bash
# Debian/Ubuntu
sudo apt-get install xdotool libnotify-bin

# Fedora
sudo dnf install xdotool libnotify

# Arch
sudo pacman -S xdotool libnotify
```

### 2. Install Python Package

```bash
# Clone the repository
git clone https://github.com/tnrzk13/voice-dictation.git
cd voice-dictation

# Install in development mode
pip install -e .

# Or install directly (when published to PyPI)
# pip install voice-dictation
```

This will install the `dictate` and `dictate-daemon` commands globally.

## Usage

### Basic Usage

```bash
# Start dictation (streaming mode - default)
dictate

# The script will:
# 1. Wait 2 seconds (giving you time to switch windows)
# 2. Start recording automatically
# 3. Transcribe and type text every 5 seconds while you speak
# 4. Stop after 4 seconds of silence
```

### Non-Streaming Mode

```bash
# Transcribe all at once after recording finishes
dictate --no-stream
```

### Streaming vs Non-Streaming

**Streaming Mode (Default)**:
- Text appears every ~5 seconds while you speak
- Great for long dictation sessions
- Provides real-time feedback
- Command: `dictate` or `dictate --stream`

**Non-Streaming Mode**:
- Records everything first, then transcribes all at once
- Good for short commands or when you want text to appear together
- Command: `dictate --no-stream`

### Daemon Management

The daemon starts automatically on first use, but you can manage it manually:

```bash
# Start daemon manually
scripts/start-dictate-daemon.sh

# Stop daemon (frees memory)
scripts/stop-dictate-daemon.sh

# Check if daemon is running
ls /tmp/dictate-daemon.sock
```

## Configuration

Edit `src/dictate/config.py` to customize:

```python
SILENCE_THRESHOLD = 0.01  # Lower = more sensitive to sound
SILENCE_DURATION = 4      # Seconds of silence before stopping
CHUNK_DURATION = 5        # Seconds between transcriptions (streaming mode)
```

### Finding Your Silence Threshold

Different microphones have different noise levels. Run the calibration utility:

```bash
python scripts/test-silence-threshold.py

# Speak normally, then be quiet
# Watch the RMS values to find the right threshold
# Update SILENCE_THRESHOLD in src/dictate/config.py
```

## Architecture

```
┌─────────────────┐
│  dictate (CLI)  │  ← Records audio, detects silence
└────────┬────────┘
         │ Unix Socket
         ↓
┌─────────────────┐
│ dictate-daemon  │  ← Whisper model stays loaded
└─────────────────┘
         │
         ↓
    xdotool types text at cursor
```

**Benefits**:
- Model loads once, stays in memory
- Fast transcription (no startup delay)
- Multiple dictation sessions without reloading model

## Troubleshooting

### Daemon won't start
```bash
# Check logs
cat ~/.local/share/voice-dictation/daemon.log

# Manually remove stale socket
rm /tmp/dictate-daemon.sock
```

### Text not appearing
- Ensure `xdotool` is installed: `which xdotool`
- Check that you have an active window where text can be typed
- Try the 2-second delay to switch windows

### Silence detection too sensitive/not sensitive enough
- Run `python scripts/test-silence-threshold.py`
- Adjust `SILENCE_THRESHOLD` in `src/dictate/config.py`
- Lower value = more sensitive (picks up quieter sounds)
- Higher value = less sensitive (requires louder sounds)

### Model download issues
The first run downloads the Whisper "base" model (~150MB). If it fails:
```bash
# Manually trigger download
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
```

### Permission denied errors
```bash
# Make scripts executable
chmod +x scripts/*.sh scripts/*.py
```

## Development

### Project Structure

```
voice-dictation/
├── src/dictate/          # Main package
│   ├── cli.py           # Command-line interface
│   ├── client.py        # Daemon client
│   ├── config.py        # Configuration constants
│   ├── daemon.py        # Transcription daemon
│   ├── recorder.py      # Audio recording
│   ├── silence.py       # Silence detection
│   └── utils.py         # Helper functions
├── scripts/             # Utility scripts
├── pyproject.toml       # Package configuration
└── README.md
```

### Running from Source

```bash
# Without installing
python -m dictate.cli

# Or install in editable mode
pip install -e .
```

## Keyboard Shortcuts (Optional)

You can bind the `dictate` command to a keyboard shortcut in your desktop environment:

**GNOME/Ubuntu**:
```
Settings → Keyboard → Custom Shortcuts
Command: dictate
Shortcut: Super+D (or your preference)
```

**i3/sway**:
```
bindsym $mod+d exec dictate
```

## Dependencies

- `numpy` - Audio processing
- `sounddevice` - Microphone input
- `faster-whisper` - Whisper model inference (CPU-optimized)

## Performance

- **Model**: Whisper base (fastest, good accuracy)
- **Device**: CPU with int8 quantization
- **Memory**: ~400MB when daemon is running
- **Latency**: ~0.5-2 seconds per transcription chunk

To use a more accurate model, edit `src/dictate/daemon.py`:
```python
model = WhisperModel("small", ...)  # or "medium", "large"
```

## Contributing

Contributions welcome! Areas for improvement:

- [ ] macOS support (replace xdotool)
- [ ] Configuration file support (~/.config/voice-dictation/config.ini)
- [ ] systemd service for auto-start
- [ ] GUI for settings
- [ ] Support for other Whisper models
- [ ] Punctuation commands ("period", "comma")

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits

Built with:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Efficient Whisper implementation
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [sounddevice](https://python-sounddevice.readthedocs.io/) - Audio I/O

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
