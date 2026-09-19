"""Progressive typer - diff-based text correction for live dictation.

Each partial from the daemon carries the text the daemon has finalized plus
the still-revisable tail. The typer diffs the cumulative partial against what
is already on screen and sends the minimal backspaces and new characters.

The commit boundary comes from the daemon, not a client-side stability guess:
the daemon only finalizes words with enough following speech that Whisper
stops revising them, so committed text is genuinely immutable. Deriving it
client-side raced ahead of that and let a later revision retype visible text.
"""

import re
import time
from typing import Tuple

from dictate.config import BACKSPACE_SETTLE_DELAY
from dictate.live.formatting import apply_formatting_commands
from dictate.xdotool import type_text as _type_text, send_backspaces as _send_backspaces


class ProgressiveTyper:
    """Applies partial and final transcription results with minimal retyping.

    One instance handles one dictation session; a new client creates a new
    typer, so the daemon's per-connection finalized prefix maps directly onto
    ``_committed``.
    """

    def __init__(self) -> None:
        self._committed = ""  # Daemon-finalized text - won't change
        self._pending = ""  # Revisable tail - may change
        self.last_typed_at: float = 0.0
        self.is_typing: bool = False

    @property
    def committed(self) -> str:
        return self._committed

    @property
    def pending(self) -> str:
        return self._pending

    @property
    def displayed_text(self) -> str:
        return self._committed + self._pending

    def apply_partial(self, text: str, finalized: str = "") -> Tuple[int, str]:
        """Update the display with a cumulative partial that may still change.

        Returns:
            Tuple of (backspaces_needed, text_to_type) for the display update.
        """
        if not text.strip():
            return 0, ""
        target = _capitalize_first(apply_formatting_commands(text))
        committed = _finalized_prefix(target, finalized)
        backspaces, to_type = self._compute_edit(self.displayed_text, target)
        self._committed = committed
        self._pending = target[len(committed):]
        self._execute_edit(backspaces, to_type)
        return backspaces, to_type

    def apply_final(self, text: str) -> Tuple[int, str]:
        """Lock in a final result - this text won't be revised.

        Returns:
            Tuple of (backspaces_needed, text_to_type) for the display update.
        """
        if not text.strip():
            return 0, ""
        target = _capitalize_first(apply_formatting_commands(text)) + " "
        backspaces, to_type = self._compute_edit(self.displayed_text, target)
        self._committed = target
        self._pending = ""
        self._execute_edit(backspaces, to_type)
        return backspaces, to_type

    def apply_final_trailing(self, text: str) -> Tuple[int, str]:
        """Append only the sentence mark from a final result, never rewrite text.

        Used when the session was already stopped by a key: the cursor may no
        longer be where the typer left it, so backspacing into the existing
        text could corrupt it. Only a trailing period/question/exclamation is
        added, and only when the final's last word matches the display.
        """
        target = apply_formatting_commands(text).strip()
        match = _TERMINAL_PUNCT_RE.search(target)
        if not match:
            return 0, ""
        body = target[: match.start()].rstrip()
        current = self.displayed_text.rstrip()
        if not current or current[-1] in _TRAILING_PUNCT:
            return 0, ""
        if _last_word(current) != _last_word(body):
            return 0, ""
        to_type = match.group(0) + " "
        self._committed = current + to_type
        self._pending = ""
        self._execute_edit(0, to_type)
        return 0, to_type

    def _compute_edit(self, old: str, new: str) -> Tuple[int, str]:
        """Compute minimal backspaces and new text to transform old into new."""
        common_length = _find_common_prefix_length(old, new)
        backspaces = len(old) - common_length
        to_type = new[common_length:]
        return backspaces, to_type

    def _execute_edit(self, backspaces: int, to_type: str) -> None:
        """Send backspaces and type new text via xdotool."""
        if not (backspaces > 0 or to_type):
            return
        self.is_typing = True
        try:
            if backspaces > 0:
                _send_backspaces(backspaces)
            if backspaces > 0 and to_type:
                time.sleep(BACKSPACE_SETTLE_DELAY)
            if to_type:
                _type_text(to_type)
        finally:
            self.last_typed_at = time.time()
            self.is_typing = False


def _capitalize_first(text: str) -> str:
    """Capitalize the first character, leaving the rest unchanged."""
    if not text:
        return text
    return text[0].upper() + text[1:]


_TERMINAL_PUNCT_RE = re.compile(r"[.?!]+$")
_TRAILING_PUNCT = ".,?!;:\"')]}"


def _last_word(text: str) -> str:
    """Return the final word lowercased with surrounding punctuation removed."""
    words = text.split()
    return words[-1].strip(".,?!;:\"'").lower() if words else ""


def _finalized_prefix(target: str, finalized: str) -> str:
    """Return the portion of formatted target covered by the daemon's finalized text.

    The split is the common prefix with the formatted finalized text, which
    tolerates the boundary re-formatting that happens when a formatting command
    in the finalized text only resolves once the next word is present.
    """
    if not finalized:
        return ""
    formatted = _capitalize_first(apply_formatting_commands(finalized))
    return target[: _find_common_prefix_length(target, formatted)]


def _find_common_prefix_length(a: str, b: str) -> int:
    """Return the length of the longest common prefix between two strings."""
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit
