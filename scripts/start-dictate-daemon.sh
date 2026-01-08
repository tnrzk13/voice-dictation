#!/bin/bash
# Start the dictation daemon in the background

# Check if daemon is already running
if [ -e /tmp/dictate-daemon.sock ]; then
    echo "Dictation daemon appears to be already running."
    echo "If it's not, remove /tmp/dictate-daemon.sock and try again."
    exit 1
fi

# Start daemon using the installed command (falls back to python module if not installed)
if command -v dictate-daemon &> /dev/null; then
    nohup dictate-daemon > /dev/null 2>&1 &
else
    nohup python3 -m dictate.daemon > /dev/null 2>&1 &
fi

# Wait a moment for it to start
sleep 2

# Check if it started successfully
if [ -e /tmp/dictate-daemon.sock ]; then
    echo "Dictation daemon started successfully!"
    notify-send "Dictation Daemon" "Started successfully - ready for dictation!"
else
    echo "Failed to start daemon. Check ~/.local/share/voice-dictation/daemon.log for errors."
    notify-send "Dictation Daemon" "Failed to start - check logs"
    exit 1
fi
