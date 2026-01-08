#!/bin/bash
# Stop the dictation daemon and free up memory

# Kill the daemon process (handles both old and new process names)
pkill -f "dictate.daemon" || pkill -f "dictate-daemon"

# Remove the socket file
rm -f /tmp/dictate-daemon.sock

# Notify user
notify-send "Dictation Daemon" "Stopped - memory freed"
