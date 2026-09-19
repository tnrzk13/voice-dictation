#!/usr/bin/env python3
"""GUI fixture manager for local audio regression recordings.

Provides a tkinter window to:
- View the script and directions for each edge case
- Record multiple takes per edge case
- See which takes exist, play them, and delete them
- Capture golden chunk sequences from a take
- Capture with the same model the daemon runs

Run with:
    python tools/fixture_manager.py
    or
    python -m tools.fixture_manager
"""

import json
import queue
import shutil
import sys
import threading
import tkinter as tk
import wave
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import pydub
import sounddevice as sd

# Allow running this script directly: python tools/fixture_manager.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from dictate.config import (
    BYTES_PER_SAMPLE,
    BYTES_PER_SECOND,
    SAMPLE_RATE,
    SOCKET_PATH,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)
from dictate.daemon_support import is_daemon_running, read_daemon_config
from tools.capture_chunks import capture_chunks, load_audio
from tools.fixture_definitions import FIXTURES
from tools.fixture_store import (
    LOCAL_FIXTURES_DIR,
    list_take_ids,
    migrate_legacy_take,
    next_take_id,
    take_audio_path,
    take_chunks_path,
    take_dir,
    take_reference_path,
)


class FixtureManager:
    """Tkinter window for managing local audio regression fixtures."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Voice Dictation Fixture Manager")
        self.root.geometry("900x680")
        self.root.minsize(720, 540)

        self.output_dir = LOCAL_FIXTURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in FIXTURES:
            migrate_legacy_take(name, self.output_dir)

        self.current_fixture: str = ""
        self.current_take: str = ""
        self.pending_take_id: str = ""
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
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)

        # Left panel: fixture list
        left_frame = ttk.LabelFrame(main_frame, text="Edge Cases", padding="5")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)

        self.fixture_list = tk.Listbox(left_frame, selectmode=tk.SINGLE, exportselection=False)
        self.fixture_list.grid(row=0, column=0, sticky="nsew")
        self.fixture_list.bind("<<ListboxSelect>>", self._on_fixture_select)

        fixture_scrollbar = ttk.Scrollbar(
            left_frame, orient=tk.VERTICAL, command=self.fixture_list.yview
        )
        fixture_scrollbar.grid(row=0, column=1, sticky="ns")
        self.fixture_list.config(yscrollcommand=fixture_scrollbar.set)

        # Right panel: details, takes, controls
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(3, weight=1)

        self.warning_label = ttk.Label(
            right_frame, text="", foreground="red", wraplength=500, justify=tk.LEFT
        )
        self.warning_label.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.fixture_name_label = ttk.Label(
            right_frame, text="Select an edge case", font=("Helvetica", 14, "bold")
        )
        self.fixture_name_label.grid(row=1, column=0, sticky="w", pady=(0, 10))

        details_frame = ttk.LabelFrame(right_frame, text="Script & Directions", padding="10")
        details_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        details_frame.columnconfigure(0, weight=1)

        self.details_text = tk.Text(details_frame, wrap=tk.WORD, height=7, state=tk.DISABLED)
        self.details_text.grid(row=0, column=0, sticky="nsew")

        details_scrollbar = ttk.Scrollbar(
            details_frame, orient=tk.VERTICAL, command=self.details_text.yview
        )
        details_scrollbar.grid(row=0, column=1, sticky="ns")
        self.details_text.config(yscrollcommand=details_scrollbar.set)

        # Recordings (takes) panel
        takes_frame = ttk.LabelFrame(right_frame, text="Recordings", padding="10")
        takes_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        takes_frame.rowconfigure(0, weight=1)
        takes_frame.columnconfigure(0, weight=1)

        self.takes_list = tk.Listbox(takes_frame, selectmode=tk.SINGLE, exportselection=False)
        self.takes_list.grid(row=0, column=0, sticky="nsew")
        self.takes_list.bind("<<ListboxSelect>>", self._on_take_select)

        takes_scrollbar = ttk.Scrollbar(
            takes_frame, orient=tk.VERTICAL, command=self.takes_list.yview
        )
        takes_scrollbar.grid(row=0, column=1, sticky="ns")
        self.takes_list.config(yscrollcommand=takes_scrollbar.set)

        # Status and level
        status_frame = ttk.Frame(right_frame)
        status_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
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
        button_frame.grid(row=5, column=0, sticky="ew")

        self.record_button = ttk.Button(
            button_frame, text="New Recording", command=self._toggle_record
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

    def _populate_fixture_list(self, select: str = "") -> None:
        """Fill the fixture listbox, showing each edge case's take count."""
        self.fixture_list.delete(0, tk.END)
        self._fixture_names = []
        for name in FIXTURES:
            count = len(list_take_ids(name, self.output_dir))
            label = f"{name} ({count})" if count else name
            self.fixture_list.insert(tk.END, label)
            self._fixture_names.append(name)

        if select:
            self._select_fixture(name=select)

    def _select_fixture(self, name: str) -> None:
        """Select a fixture by name, if present."""
        if name not in self._fixture_names:
            return
        index = self._fixture_names.index(name)
        self.fixture_list.selection_clear(0, tk.END)
        self.fixture_list.selection_set(index)
        self.fixture_list.see(index)
        self._on_fixture_select()

    def _on_fixture_select(self, _event=None) -> None:
        """Update details and takes when a fixture is selected."""
        selection = self.fixture_list.curselection()
        if not selection:
            return
        self.current_fixture = self._fixture_names[selection[0]]
        self.current_take = ""
        self._update_details()
        self._refresh_takes_list()
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

    def _refresh_takes_list(self, select: str = "") -> None:
        """Reload the takes listbox for the current fixture."""
        self.takes_list.delete(0, tk.END)
        self._take_ids = (
            list_take_ids(self.current_fixture, self.output_dir) if self.current_fixture else []
        )
        for take_id in self._take_ids:
            self.takes_list.insert(tk.END, self._take_label(take_id))

        if select and select in self._take_ids:
            self._select_take(select)
        elif self._take_ids and self.current_take not in self._take_ids:
            self.takes_list.selection_set(0)
            self._on_take_select()

    def _take_label(self, take_id: str) -> str:
        """Format a take row with its duration and capture state."""
        audio_path = take_audio_path(self.current_fixture, take_id, self.output_dir)
        duration = self._audio_duration(audio_path)
        captured = take_chunks_path(self.current_fixture, take_id, self.output_dir).exists()
        suffix = "  [captured]" if captured else ""
        return f"{take_id}  {duration:.1f}s{suffix}"

    def _select_take(self, take_id: str) -> None:
        """Select a take by id, if present."""
        if take_id not in self._take_ids:
            return
        index = self._take_ids.index(take_id)
        self.takes_list.selection_clear(0, tk.END)
        self.takes_list.selection_set(index)
        self.takes_list.see(index)
        self._on_take_select()

    def _on_take_select(self, _event=None) -> None:
        """Track the selected take."""
        selection = self.takes_list.curselection()
        self.current_take = self._take_ids[selection[0]] if selection else ""
        self._update_buttons()

    def _audio_duration(self, path: Path) -> float:
        """Return the duration of a WAV file in seconds."""
        audio = pydub.AudioSegment.from_wav(str(path))
        return len(audio) / 1000.0

    def _update_buttons(self) -> None:
        """Enable/disable buttons based on current state."""
        has_fixture = bool(self.current_fixture)
        has_take = bool(self.current_take)
        busy = self.is_playing or self.is_capturing

        if self.is_recording:
            self.record_button.config(text="Stop")
            self.play_button.config(state=tk.DISABLED)
            self.capture_button.config(state=tk.DISABLED)
            self.delete_button.config(state=tk.DISABLED)
            return

        self.record_button.config(
            text="New Recording",
            state=tk.NORMAL if has_fixture and not busy else tk.DISABLED,
        )
        take_state = tk.NORMAL if has_take and not busy else tk.DISABLED
        self.play_button.config(state=take_state)
        self.capture_button.config(state=take_state)
        self.delete_button.config(state=take_state)

    def _toggle_record(self) -> None:
        """Start or stop recording a new take."""
        if not self.current_fixture:
            messagebox.showinfo("No Edge Case", "Select an edge case from the list first.")
            return
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Begin recording a new take from the microphone."""
        self.pending_take_id = next_take_id(self.current_fixture, self.output_dir)
        try:
            self.is_recording = True
            self.record_frames = []
            self._update_status(f"Recording {self.pending_take_id}...")
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
                "Is another application (like the dictation client) using the microphone?",
            )

    def _stop_recording(self) -> None:
        """Stop recording and save the take."""
        if not self.is_recording:
            return
        self.is_recording = False
        self.record_stream.stop()
        self.record_stream.close()
        self.record_stream = None

        if not self.record_frames:
            self._update_status("No audio recorded")
            self._update_buttons()
            self.level_meter.config(value=0)
            return

        audio = np.concatenate(self.record_frames, axis=0).tobytes()
        audio_path = take_audio_path(self.current_fixture, self.pending_take_id, self.output_dir)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_wav(audio, audio_path)
        self._update_status(f"Saved {self.pending_take_id} ({len(audio) / BYTES_PER_SECOND:.1f}s)")

        recorded_take = self.pending_take_id
        self._populate_fixture_list(select=self.current_fixture)
        self._refresh_takes_list(select=recorded_take)
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
        """Play the selected take."""
        if not self.current_take:
            return
        self._update_status("Playing...")
        self.is_playing = True
        self._update_buttons()
        thread = threading.Thread(target=self._play_thread, daemon=True)
        thread.start()

    def _play_thread(self) -> None:
        """Background thread: playback audio and update UI when done."""
        try:
            audio_path = take_audio_path(self.current_fixture, self.current_take, self.output_dir)
            audio = pydub.AudioSegment.from_wav(str(audio_path))
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
        """Run the selected take through Whisper and save its chunks."""
        if not self.current_take:
            return
        self.is_capturing = True
        model, device, compute_type = self._capture_model_args()
        self._update_status(f"Capturing with {model} ({device})...")
        self._update_buttons()
        thread = threading.Thread(
            target=self._capture_thread,
            args=(model, device, compute_type),
            daemon=True,
        )
        thread.start()

    def _capture_model_args(self) -> tuple:
        """Use the daemon's configured model so captures match production."""
        config = read_daemon_config(SOCKET_PATH)
        if config:
            return (
                config.get("model", WHISPER_MODEL_SIZE),
                config.get("device", WHISPER_DEVICE),
                config.get("compute_type", WHISPER_COMPUTE_TYPE),
            )
        return WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

    def _capture_thread(self, model: str, device: str, compute_type: str) -> None:
        """Background thread: transcribe the take and write golden chunks."""
        try:
            audio_path = take_audio_path(self.current_fixture, self.current_take, self.output_dir)
            chunks = capture_chunks(load_audio(str(audio_path)), model, device, compute_type)

            take_path = take_dir(self.current_fixture, self.current_take, self.output_dir)
            take_path.mkdir(parents=True, exist_ok=True)
            take_chunks_path(self.current_fixture, self.current_take, self.output_dir).write_text(
                "".join(json.dumps(chunk) + "\n" for chunk in chunks),
                encoding="utf-8",
            )

            final = next((chunk for chunk in reversed(chunks) if chunk["type"] == "final"), None)
            partial = next(
                (chunk for chunk in reversed(chunks) if chunk["type"] == "partial"), None
            )
            reference = (final or partial or {}).get("text", "").strip()
            take_reference_path(
                self.current_fixture, self.current_take, self.output_dir
            ).write_text(reference, encoding="utf-8")

            self._schedule_ui_update(
                lambda: self._update_status(f"Captured {len(chunks)} messages")
            )
            self._schedule_ui_update(
                lambda: messagebox.showinfo("Capture Complete", f"Reference: {reference}")
            )
        except Exception as exc:
            self._schedule_ui_update(lambda: messagebox.showerror("Capture Error", str(exc)))
        finally:
            self.is_capturing = False
            self._schedule_ui_update(self._refresh_takes_list)
            self._schedule_ui_update(self._update_buttons)

    def _delete(self) -> None:
        """Delete the selected take and its captured chunks."""
        if not self.current_take:
            return
        if messagebox.askyesno(
            "Delete Recording", f"Delete {self.current_fixture}/{self.current_take}?"
        ):
            shutil.rmtree(take_dir(self.current_fixture, self.current_take, self.output_dir))
            self._update_status(f"Deleted {self.current_take}")
            self.current_take = ""
            self._populate_fixture_list(select=self.current_fixture)
            self._refresh_takes_list()
            self._update_buttons()

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
                    "Warning: the dictation daemon is running. Stop it with 'dictate-stop' "
                    "before recording so it does not compete for the microphone."
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
