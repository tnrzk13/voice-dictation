#!/usr/bin/env python3
"""GUI fixture manager for local audio regression recordings.

Provides a tkinter window to:
- View the script and directions for each edge case
- Record audio from the microphone
- Play back recordings
- Capture golden chunk sequences from recordings
- Delete recordings

Run with:
    python tools/fixture_manager.py
    or
    python -m tools.fixture_manager
"""

import json
import queue
import sys
import threading
import time
import tkinter as tk
import wave
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import pydub
import sounddevice as sd

# Allow running this script directly: python tools/fixture_manager.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from dictate.config import BYTES_PER_SAMPLE, BYTES_PER_SECOND, SAMPLE_RATE, SOCKET_PATH
from dictate.daemon_support import is_daemon_running
from tools.capture_chunks import capture_chunks, load_audio
from tools.fixture_definitions import FIXTURES


class FixtureManager:
    """Tkinter window for managing local audio regression fixtures."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Voice Dictation Fixture Manager")
        self.root.geometry("900x650")
        self.root.minsize(700, 500)

        self.output_dir = Path("tests/audio_fixtures_local")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_fixture: str = ""
        self.is_recording = False
        self.record_frames: list = []
        self.record_stream: sd.InputStream = None
        self.is_playing = False
        self.is_capturing = False

        # UI updates from worker threads must go through this queue and be
        # processed by the main thread; tkinter is not thread-safe.
        self._ui_queue: queue.Queue = queue.Queue()
        self._current_level = 0
        self._level_lock = threading.Lock()

        self._build_ui()
        self._populate_fixture_list()
        self._check_daemon_running()
        self._start_ui_queue_polling()
        self._start_level_meter_polling()

    def _build_ui(self) -> None:
        """Build the tkinter interface."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)

        # Left panel: fixture list
        left_frame = ttk.LabelFrame(main_frame, text="Edge Cases", padding="5")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)

        self.fixture_list = tk.Listbox(left_frame, selectmode=tk.SINGLE)
        self.fixture_list.grid(row=0, column=0, sticky="nsew")
        self.fixture_list.bind("<<ListboxSelect>>", self._on_fixture_select)

        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.fixture_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.fixture_list.config(yscrollcommand=scrollbar.set)

        # Right panel: details and controls
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        self.warning_label = ttk.Label(
            right_frame,
            text="",
            foreground="red",
            wraplength=500,
            justify=tk.LEFT,
        )
        self.warning_label.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.fixture_name_label = ttk.Label(
            right_frame, text="Select a fixture", font=("Helvetica", 14, "bold")
        )
        self.fixture_name_label.grid(row=1, column=0, sticky="w", pady=(0, 10))

        # Details area
        details_frame = ttk.LabelFrame(right_frame, text="Script & Directions", padding="10")
        details_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)

        self.details_text = tk.Text(details_frame, wrap=tk.WORD, height=12, state=tk.DISABLED)
        self.details_text.grid(row=0, column=0, sticky="nsew")

        details_scrollbar = ttk.Scrollbar(
            details_frame, orient=tk.VERTICAL, command=self.details_text.yview
        )
        details_scrollbar.grid(row=0, column=1, sticky="ns")
        self.details_text.config(yscrollcommand=details_scrollbar.set)

        # Status and level
        status_frame = ttk.Frame(right_frame)
        status_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)

        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        self.status_label = ttk.Label(status_frame, text="Idle")
        self.status_label.grid(row=0, column=1, sticky="w", padx=(5, 0))

        ttk.Label(status_frame, text="Level:").grid(row=1, column=0, sticky="w")
        self.level_meter = ttk.Progressbar(
            status_frame, orient=tk.HORIZONTAL, mode="determinate", maximum=100
        )
        self.level_meter.grid(row=1, column=1, sticky="ew", padx=(5, 0))

        # Buttons
        button_frame = ttk.Frame(right_frame)
        button_frame.grid(row=4, column=0, sticky="ew")

        self.record_button = ttk.Button(
            button_frame, text="Record", command=self._toggle_record
        )
        self.record_button.grid(row=0, column=0, padx=5)

        self.play_button = ttk.Button(
            button_frame, text="Play", command=self._play, state=tk.DISABLED
        )
        self.play_button.grid(row=0, column=1, padx=5)

        self.capture_button = ttk.Button(
            button_frame, text="Capture Chunks", command=self._capture, state=tk.DISABLED
        )
        self.capture_button.grid(row=0, column=2, padx=5)

        self.delete_button = ttk.Button(
            button_frame, text="Delete", command=self._delete, state=tk.DISABLED
        )
        self.delete_button.grid(row=0, column=3, padx=5)

        self.recording_info_label = ttk.Label(right_frame, text="")
        self.recording_info_label.grid(row=5, column=0, sticky="w", pady=(10, 0))

    def _populate_fixture_list(self) -> None:
        """Fill the fixture listbox with available edge cases."""
        for name in FIXTURES:
            self.fixture_list.insert(tk.END, name)

    def _on_fixture_select(self, _event=None) -> None:
        """Update the details panel when a fixture is selected."""
        selection = self.fixture_list.curselection()
        if not selection:
            return
        self.current_fixture = self.fixture_list.get(selection[0])
        self._update_details()
        self._update_buttons()

    def _update_details(self) -> None:
        """Show the script, directions, and focus for the selected fixture."""
        if not self.current_fixture:
            return
        info = FIXTURES[self.current_fixture]
        self.fixture_name_label.config(text=self.current_fixture)

        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert(tk.END, f"Script:\n{info['script']}\n\n")
        self.details_text.insert(tk.END, f"Directions:\n{info['directions']}\n\n")
        self.details_text.insert(tk.END, f"Focus:\n{info['focus']}")
        self.details_text.config(state=tk.DISABLED)

        self._update_recording_info()

    def _update_recording_info(self) -> None:
        """Show the duration of the existing recording, if any."""
        audio_path = self._audio_path()
        if audio_path.exists():
            duration = self._audio_duration(audio_path)
            self.recording_info_label.config(text=f"Recording: {duration:.1f}s")
        else:
            self.recording_info_label.config(text="No recording")

    def _audio_path(self) -> Path:
        """Return the WAV path for the selected fixture."""
        return self.output_dir / self.current_fixture / "audio.wav"

    def _audio_duration(self, path: Path) -> float:
        """Return the duration of a WAV file in seconds."""
        audio = pydub.AudioSegment.from_wav(str(path))
        return len(audio) / 1000.0

    def _update_buttons(self) -> None:
        """Enable/disable buttons based on current state."""
        has_audio = self._audio_path().exists()
        base_state = tk.NORMAL if has_audio else tk.DISABLED

        if self.is_recording:
            self.record_button.config(text="Stop")
            self.play_button.config(state=tk.DISABLED)
            self.capture_button.config(state=tk.DISABLED)
            self.delete_button.config(state=tk.DISABLED)
        else:
            self.record_button.config(text="Record")
            self.play_button.config(state=base_state if not self.is_playing else tk.DISABLED)
            self.capture_button.config(state=base_state if not self.is_capturing else tk.DISABLED)
            self.delete_button.config(state=base_state)

    def _toggle_record(self) -> None:
        """Start or stop recording."""
        if not self.current_fixture:
            messagebox.showinfo("No Fixture", "Select a fixture from the list first.")
            return
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Begin recording audio from the microphone."""
        try:
            self.is_recording = True
            self.record_frames = []
            self._update_status("Recording...")
            self._update_buttons()

            self.record_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.int16,
                callback=self._audio_callback,
            )
            self.record_stream.start()
        except Exception as exc:
            self.is_recording = False
            self._update_status(f"Error: {exc}")
            self._update_buttons()
            messagebox.showerror(
                "Recording Error",
                f"Could not start recording:\n{exc}\n\n"
                "Is another application (like the dictation daemon) using the microphone?",
            )

    def _stop_recording(self) -> None:
        """Stop recording and save the captured audio."""
        if not self.is_recording:
            return
        self.is_recording = False
        self.record_stream.stop()
        self.record_stream.close()
        self.record_stream = None

        if self.record_frames:
            audio = np.concatenate(self.record_frames, axis=0).tobytes()
            audio_path = self._audio_path()
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_wav(audio, audio_path)
            self._update_status(f"Saved {len(audio) / BYTES_PER_SECOND:.1f}s")
        else:
            self._update_status("No audio recorded")

        self._update_buttons()
        self._update_recording_info()
        self.level_meter.config(value=0)

    def _audio_callback(self, indata: np.ndarray, _frames: int, _time_info, _status) -> None:
        """SoundDevice callback: collect frames and update shared level."""
        self.record_frames.append(indata.copy())
        rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2))
        level = min(100, int(rms / 32768.0 * 200))
        with self._level_lock:
            self._current_level = level

    def _save_wav(self, audio_bytes: bytes, path: Path) -> None:
        """Write raw PCM int16 bytes to a mono WAV file."""
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(BYTES_PER_SAMPLE)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(audio_bytes)

    def _play(self) -> None:
        """Play the selected fixture's recording."""
        if not self._audio_path().exists():
            return
        self._update_status("Playing...")
        self.is_playing = True
        self._update_buttons()
        thread = threading.Thread(target=self._play_thread, daemon=True)
        thread.start()

    def _play_thread(self) -> None:
        """Background thread: playback audio and update UI when done."""
        try:
            audio = pydub.AudioSegment.from_wav(str(self._audio_path()))
            samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
            sd.play(samples, SAMPLE_RATE)
            sd.wait()
        except Exception as exc:
            self._schedule_ui_update(lambda: messagebox.showerror("Playback Error", str(exc)))
        finally:
            self.is_playing = False
            self._schedule_ui_update(self._update_status_idle)
            self._schedule_ui_update(self._update_buttons)

    def _capture(self) -> None:
        """Run the selected recording through Whisper and save chunk sequences."""
        if not self._audio_path().exists():
            return
        self.is_capturing = True
        self._update_status("Capturing chunks...")
        self._update_buttons()
        thread = threading.Thread(target=self._capture_thread, daemon=True)
        thread.start()

    def _capture_thread(self) -> None:
        """Background thread: transcribe audio and write golden chunks."""
        try:
            audio_bytes = load_audio(str(self._audio_path()))
            chunks = capture_chunks(audio_bytes, "tiny", "cpu", "int8")

            fixture_dir = self.output_dir / self.current_fixture
            fixture_dir.mkdir(parents=True, exist_ok=True)
            (fixture_dir / "chunks.jsonl").write_text(
                "".join(json.dumps(chunk) + "\n" for chunk in chunks),
                encoding="utf-8",
            )

            final = next((chunk for chunk in reversed(chunks) if chunk["type"] == "final"), None)
            partial = next(
                (chunk for chunk in reversed(chunks) if chunk["type"] == "partial"), None
            )
            reference = (final or partial or {}).get("text", "").strip()
            (fixture_dir / "reference.txt").write_text(reference, encoding="utf-8")

            self._schedule_ui_update(lambda: self._update_status(f"Captured {len(chunks)} messages"))
            self._schedule_ui_update(
                lambda: messagebox.showinfo("Capture Complete", f"Reference: {reference}")
            )
        except Exception as exc:
            self._schedule_ui_update(lambda: messagebox.showerror("Capture Error", str(exc)))
        finally:
            self.is_capturing = False
            self._schedule_ui_update(self._update_buttons)

    def _delete(self) -> None:
        """Delete the selected fixture's recording and captured chunks."""
        if not self._audio_path().exists():
            return
        if messagebox.askyesno(
            "Delete Recording", f"Delete recording and chunks for {self.current_fixture}?"
        ):
            self._audio_path().unlink()
            chunks_path = self._audio_path().parent / "chunks.jsonl"
            ref_path = self._audio_path().parent / "reference.txt"
            if chunks_path.exists():
                chunks_path.unlink()
            if ref_path.exists():
                ref_path.unlink()
            self._update_status("Deleted")
            self._update_buttons()
            self._update_recording_info()

    def _start_ui_queue_polling(self) -> None:
        """Poll the UI update queue from the main thread every 50 ms."""
        self._process_ui_queue()
        self.root.after(50, self._start_ui_queue_polling)

    def _process_ui_queue(self) -> None:
        """Run all pending UI updates from worker threads."""
        try:
            while True:
                update = self._ui_queue.get_nowait()
                update()
        except queue.Empty:
            pass

    def _schedule_ui_update(self, update: callable) -> None:
        """Schedule a UI update to run on the main thread."""
        self._ui_queue.put(update)

    def _start_level_meter_polling(self) -> None:
        """Poll the recorded audio level from the main thread."""
        if self.is_recording:
            with self._level_lock:
                level = self._current_level
            self.level_meter.config(value=level)
        self.root.after(50, self._start_level_meter_polling)

    def _update_status(self, text: str) -> None:
        """Set the status label text."""
        self.status_label.config(text=text)

    def _update_status_idle(self) -> None:
        """Set status back to Idle."""
        self.status_label.config(text="Idle")

    def _check_daemon_running(self) -> None:
        """Warn the user if the dictation daemon is using the microphone."""
        if is_daemon_running(SOCKET_PATH):
            self.warning_label.config(
                text=(
                    "Warning: the dictation daemon is running and may be using the microphone. "
                    "Stop it with 'dictate-stop' before recording fixtures."
                )
            )
        else:
            self.warning_label.config(text="")

    def _on_close(self) -> None:
        """Clean up any active recording before closing the window."""
        if self.is_recording:
            self._stop_recording()
        self.root.destroy()


def main() -> None:
    """Launch the fixture manager GUI."""
    root = tk.Tk()
    app = FixtureManager(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
