#!/usr/bin/env python3
"""
Test script to find the right silence threshold for your microphone.
Run this to see RMS values and adjust SILENCE_THRESHOLD accordingly.
"""
import sys
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


def test_callback(indata, frames, time_info, status):
    """Show RMS energy levels in real-time."""
    rms = np.sqrt(np.mean(indata**2))
    bars = int(rms * 500)  # Scale for visualization
    bar_str = '█' * bars
    print(f"RMS: {rms:.4f} {bar_str}", end='\r')


print("Testing microphone levels...")
print("Speak normally, then be quiet.")
print("Watch the RMS values to set your threshold.")
print("Press Ctrl+C to stop.\n")

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32, callback=test_callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n\nDone! Set SILENCE_THRESHOLD in dictate/config.py based on:")
    print("- RMS when speaking: should be ABOVE threshold")
    print("- RMS when silent: should be BELOW threshold")
    print("- Suggested: use a value between your silent and speaking RMS")
