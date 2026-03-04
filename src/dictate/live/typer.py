"""Progressive typer - diff-based text correction for live dictation.

Tracks committed (finalized) and pending (partial) text. On each partial
result, diffs against what's already typed and sends minimal backspaces
followed by new characters. On final result, locks text in so it won't
be revised.
"""

import re
import time
from typing import Tuple

from dictate.config import BACKSPACE_SETTLE_DELAY
from dictate.xdotool import type_text as _type_text, send_backspaces as _send_backspaces


class ProgressiveTyper:
    """Applies partial and final transcription results with minimal retyping."""

    def __init__(self) -> None:
        self._committed = ""  # Finalized text - won't change
        self._pending = ""  # Currently displayed partial - may be revised
        self.last_typed_at: float = 0.0

    @property
    def committed(self) -> str:
        return self._committed

    @property
    def pending(self) -> str:
        return self._pending

    @property
    def displayed_text(self) -> str:
        return self._committed + self._pending

    def apply_partial(self, text: str) -> Tuple[int, str]:
        """Update display with a partial result that may change later.

        Returns:
            Tuple of (backspaces_needed, text_to_type) for the display update.
        """
        new_pending = self._strip_committed_prefix(text)
        new_pending = _capitalize_first(new_pending)
        backspaces, to_type = self._compute_edit(self._pending, new_pending)
        self._pending = new_pending
        self._execute_edit(backspaces, to_type)
        return backspaces, to_type

    def apply_final(self, text: str) -> Tuple[int, str]:
        """Lock in a final result - this text won't be revised.

        Returns:
            Tuple of (backspaces_needed, text_to_type) for the display update.
        """
        new_pending = self._strip_committed_prefix(text)
        capitalized = _capitalize_first(new_pending)
        spaced = capitalized + " "
        backspaces, to_type = self._compute_edit(self._pending, spaced)
        self._committed += spaced
        self._pending = ""
        self._execute_edit(backspaces, to_type)
        return backspaces, to_type

    def _strip_committed_prefix(self, text: str) -> str:
        """Remove the committed portion from the beginning of new text.

        Uses word-based comparison so punctuation differences don't
        break matching against partial results.
        """
        committed_words = _strip_punctuation(self._committed).split()
        if not committed_words:
            return text
        text_words = text.split()
        if len(text_words) < len(committed_words):
            return text
        for i, committed_word in enumerate(committed_words):
            if text_words[i].lower() != committed_word.lower():
                return text
        remaining_words = text_words[len(committed_words) :]
        return " ".join(remaining_words) if remaining_words else ""

    def _compute_edit(self, old: str, new: str) -> Tuple[int, str]:
        """Compute minimal backspaces and new text to transform old into new."""
        common_length = _find_common_prefix_length(old, new)
        backspaces = len(old) - common_length
        to_type = new[common_length:]
        return backspaces, to_type

    def _execute_edit(self, backspaces: int, to_type: str) -> None:
        """Send backspaces and type new text via xdotool."""
        if backspaces > 0 or to_type:
            self.last_typed_at = time.time()
        if backspaces > 0:
            _send_backspaces(backspaces)
        if backspaces > 0 and to_type:
            time.sleep(BACKSPACE_SETTLE_DELAY)
        if to_type:
            _type_text(to_type)
        if backspaces > 0 or to_type:
            self.last_typed_at = time.time()


def _capitalize_first(text: str) -> str:
    """Capitalize the first character, leaving the rest unchanged."""
    if not text:
        return text
    return text[0].upper() + text[1:]


def _find_common_prefix_length(a: str, b: str) -> int:
    """Return the length of the longest common prefix between two strings."""
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


_PUNCTUATION_RE = re.compile(r"[,.\?!]")


def _strip_punctuation(text: str) -> str:
    """Remove punctuation marks for word-based prefix comparison."""
    return _PUNCTUATION_RE.sub("", text)


