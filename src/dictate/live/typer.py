"""Progressive typer - diff-based text correction for live dictation.

Tracks committed (finalized) and pending (partial) text. On each partial
result, diffs against what's already typed and sends minimal backspaces
followed by new characters. On final result, locks text in so it won't
be revised.
"""

import re
import time
from typing import Tuple

from dictate.config import (
    BACKSPACE_SETTLE_DELAY,
    KEEP_TAIL_WORDS,
    MAX_PREFIX_MISMATCH_FRACTION,
    MAX_PREFIX_MISMATCHES,
    STABILITY_THRESHOLD,
)
from dictate.live.formatting import apply_formatting_commands
from dictate.xdotool import type_text as _type_text, send_backspaces as _send_backspaces


class ProgressiveTyper:
    """Applies partial and final transcription results with minimal retyping."""

    def __init__(self) -> None:
        self._committed = ""  # Finalized text - won't change
        self._pending = ""  # Currently displayed partial - may be revised
        self._stable_count: int = 0  # Consecutive partials with matching prefix
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

    def apply_partial(self, text: str) -> Tuple[int, str]:
        """Update display with a partial result that may change later.

        Returns:
            Tuple of (backspaces_needed, text_to_type) for the display update.
        """
        if not text.strip():
            return 0, ""
        formatted = apply_formatting_commands(text)
        new_pending = self._strip_committed_prefix(formatted)
        new_pending = _capitalize_first(new_pending)
        new_pending = self._auto_commit_stable_words(new_pending)
        backspaces, to_type = self._compute_edit(self._pending, new_pending)
        self._pending = new_pending
        self._execute_edit(backspaces, to_type)
        return backspaces, to_type

    def apply_final(self, text: str) -> Tuple[int, str]:
        """Lock in a final result - this text won't be revised.

        Returns:
            Tuple of (backspaces_needed, text_to_type) for the display update.
        """
        if not text.strip():
            return 0, ""
        formatted = apply_formatting_commands(text)
        new_pending = self._strip_committed_prefix(formatted)
        capitalized = _capitalize_first(new_pending)
        spaced = capitalized + " "
        backspaces, to_type = self._compute_edit(self._pending, spaced)
        self._committed += spaced
        self._pending = ""
        self._stable_count = 0
        self._execute_edit(backspaces, to_type)
        return backspaces, to_type

    def _auto_commit_stable_words(self, new_pending: str) -> str:
        """Promote words stable across consecutive partials to committed text.

        Words that match at the start of both old and new pending for
        STABILITY_THRESHOLD consecutive calls get moved to _committed,
        making them immune to future Whisper revisions.

        Skips auto-commit when either pending text contains formatting
        whitespace (newlines or tabs) because split()/join would lose it.
        """
        if _contains_formatting_whitespace(self._pending) or _contains_formatting_whitespace(new_pending):
            self._stable_count = 0
            return new_pending
        old_words = self._pending.split()
        new_words = new_pending.split()
        match = _count_common_prefix_words(old_words, new_words)
        if match > 0:
            self._stable_count += 1
        else:
            self._stable_count = 0
            return new_pending
        if self._stable_count >= STABILITY_THRESHOLD and match > KEEP_TAIL_WORDS:
            commit_count = match - KEEP_TAIL_WORDS
            commit_text = " ".join(old_words[:commit_count]) + " "
            self._committed += commit_text
            self._pending = _capitalize_first(" ".join(old_words[commit_count:]))
            new_pending = _capitalize_first(" ".join(new_words[commit_count:]))
            self._stable_count = 0
        return new_pending

    def _strip_committed_prefix(self, text: str) -> str:
        """Remove the committed portion from the beginning of new text.

        Uses word-based comparison so punctuation differences don't
        break matching against partial results. Tolerates up to
        MAX_PREFIX_MISMATCHES word substitutions (Whisper revisions).
        """
        committed_words = _strip_punctuation(self._committed).split()
        if not committed_words:
            return text
        text_words = text.split()
        text_words_stripped = [_strip_punctuation(w) for w in text_words]
        if len(text_words_stripped) < len(committed_words):
            return text
        max_allowed = max(MAX_PREFIX_MISMATCHES, int(len(committed_words) * MAX_PREFIX_MISMATCH_FRACTION))
        mismatches = 0
        for i, committed_word in enumerate(committed_words):
            if text_words_stripped[i].lower() != committed_word.lower():
                mismatches += 1
                if mismatches > max_allowed:
                    return text
        if mismatches >= len(committed_words):
            return text  # all words differ - new text, not a revision
        remaining_words = text_words[len(committed_words):]
        return " ".join(remaining_words) if remaining_words else ""

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


def _count_common_prefix_words(old_words: list, new_words: list) -> int:
    """Count leading words that match, case-insensitive."""
    limit = min(len(old_words), len(new_words))
    for i in range(limit):
        if old_words[i].lower() != new_words[i].lower():
            return i
    return limit


def _find_common_prefix_length(a: str, b: str) -> int:
    """Return the length of the longest common prefix between two strings."""
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


_PUNCTUATION_RE = re.compile(r"[,\.\?!;:\"]")


def _strip_punctuation(text: str) -> str:
    """Remove punctuation marks for word-based prefix comparison."""
    return _PUNCTUATION_RE.sub("", text)


def _contains_formatting_whitespace(text: str) -> bool:
    """Return True if text contains whitespace produced by formatting commands."""
    return "\n" in text or "\t" in text
