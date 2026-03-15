"""Spoken command formatting - converts spoken words to symbols and punctuation.

Maps trigger phrases like "slash", "new line", "open parenthesis" to their
corresponding characters, applied as post-processing on final transcription
results. Inspired by macOS/Windows dictation lookup tables.
"""

import re
from enum import Enum, auto
from typing import List, NamedTuple, Set, Tuple


class SpacingRule(Enum):
    DEFAULT = auto()       # Normal word spacing (space before and after)
    REMOVE_BEFORE = auto() # Attach to preceding word (punctuation)
    REMOVE_AFTER = auto()  # Attach to following word (opening brackets)
    REMOVE_BOTH = auto()   # No spaces on either side (path separators, newlines)


class FormattingCommand(NamedTuple):
    trigger: str           # Lowercase spoken phrase, e.g. "new line"
    replacement: str       # Output character(s)
    spacing: SpacingRule


# Sorted longest-trigger-first so multi-word triggers match before single-word
FORMATTING_COMMANDS: List[FormattingCommand] = sorted(
    [
        # Formatting
        FormattingCommand("new paragraph", "\n\n", SpacingRule.REMOVE_BOTH),
        FormattingCommand("new line", "\n", SpacingRule.REMOVE_BOTH),
        FormattingCommand("tab key", "\t", SpacingRule.REMOVE_BOTH),
        # Punctuation
        FormattingCommand("exclamation mark", "!", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("exclamation point", "!", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("question mark", "?", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("period", ".", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("dot", ".", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("comma", ",", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("colon", ":", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("semicolon", ";", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("ellipsis", "...", SpacingRule.REMOVE_BEFORE),
        # Brackets
        FormattingCommand("open parenthesis", "(", SpacingRule.REMOVE_AFTER),
        FormattingCommand("close parenthesis", ")", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("open parentheses", "(", SpacingRule.REMOVE_AFTER),
        FormattingCommand("close parentheses", ")", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("open paren", "(", SpacingRule.REMOVE_AFTER),
        FormattingCommand("close paren", ")", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("open bracket", "[", SpacingRule.REMOVE_AFTER),
        FormattingCommand("close bracket", "]", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("open brace", "{", SpacingRule.REMOVE_AFTER),
        FormattingCommand("close brace", "}", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("open angle bracket", "<", SpacingRule.REMOVE_AFTER),
        FormattingCommand("close angle bracket", ">", SpacingRule.REMOVE_BEFORE),
        # Quotes
        FormattingCommand("open quote", '"', SpacingRule.REMOVE_AFTER),
        FormattingCommand("close quote", '"', SpacingRule.REMOVE_BEFORE),
        FormattingCommand("open single quote", "'", SpacingRule.REMOVE_AFTER),
        FormattingCommand("close single quote", "'", SpacingRule.REMOVE_BEFORE),
        FormattingCommand("backtick", "`", SpacingRule.REMOVE_BOTH),
        # Path/infix symbols
        FormattingCommand("slash", "/", SpacingRule.REMOVE_BOTH),
        FormattingCommand("backslash", "\\", SpacingRule.REMOVE_BOTH),
        FormattingCommand("hyphen", "-", SpacingRule.REMOVE_BOTH),
        FormattingCommand("dash", "-", SpacingRule.REMOVE_BOTH),
        FormattingCommand("underscore", "_", SpacingRule.REMOVE_BOTH),
        # Programming symbols
        FormattingCommand("equals sign", "=", SpacingRule.DEFAULT),
        FormattingCommand("plus sign", "+", SpacingRule.DEFAULT),
        FormattingCommand("minus sign", "-", SpacingRule.DEFAULT),
        FormattingCommand("at sign", "@", SpacingRule.REMOVE_BOTH),
        FormattingCommand("hash sign", "#", SpacingRule.REMOVE_AFTER),
        FormattingCommand("dollar sign", "$", SpacingRule.REMOVE_AFTER),
        FormattingCommand("percent sign", "%", SpacingRule.REMOVE_BOTH),
        FormattingCommand("caret", "^", SpacingRule.REMOVE_BOTH),
        FormattingCommand("ampersand", "&", SpacingRule.DEFAULT),
        FormattingCommand("asterisk", "*", SpacingRule.REMOVE_BOTH),
        FormattingCommand("pipe", "|", SpacingRule.DEFAULT),
        FormattingCommand("tilde", "~", SpacingRule.REMOVE_AFTER),
    ],
    key=lambda cmd: -len(cmd.trigger.split()),
)

# Precomputed trigger words - avoids re-splitting on every match attempt
_TRIGGER_WORDS: List[Tuple[List[str], FormattingCommand]] = [
    (cmd.trigger.split(), cmd) for cmd in FORMATTING_COMMANDS
]

# Punctuation characters that Whisper may auto-insert before we add our own
_DEDUP_PUNCTUATION: Set[str] = set(".?!,;:")
_ELLIPSIS_PLACEHOLDER = "\x00ELLIPSIS\x00"

# Spacing rules that attach to the preceding word (strip space + punct before)
_ATTACHES_BEFORE = {SpacingRule.REMOVE_BEFORE, SpacingRule.REMOVE_BOTH}


def apply_formatting_commands(text: str) -> str:
    """Replace spoken command phrases with their corresponding symbols.

    Public orchestrator: trigger replacement, then punctuation dedup.
    """
    text = _replace_triggers(text)
    text = _deduplicate_punctuation(text)
    return text


def _replace_triggers(text: str) -> str:
    """Scan words and replace trigger sequences with formatted output."""
    words = text.split()
    if not words:
        return text

    result_parts: List[str] = []
    suppress_next_space = False
    i = 0

    while i < len(words):
        matched = _match_trigger_at(words, i)
        if matched:
            cmd, word_count = matched
            suppress_next_space = _append_replacement(result_parts, cmd)
            i += word_count
        else:
            _append_word(result_parts, words[i], suppress_next_space)
            suppress_next_space = False
            i += 1

    return "".join(result_parts)


_TRAILING_PUNCT_RE = re.compile(r"[.,?!;:]+$")


def _match_trigger_at(words: List[str], start: int):
    """Try to match a trigger phrase starting at the given word index.

    Strips trailing punctuation from candidate words before comparison,
    since Whisper may auto-insert punctuation (e.g. "slash," or "world.").

    Returns (FormattingCommand, word_count) or None.
    """
    for trigger_words, cmd in _TRIGGER_WORDS:
        trigger_len = len(trigger_words)
        if start + trigger_len > len(words):
            continue
        candidate = [
            _TRAILING_PUNCT_RE.sub("", w.lower())
            for w in words[start : start + trigger_len]
        ]
        if candidate == trigger_words:
            return cmd, trigger_len
    return None


def _append_replacement(result_parts: List[str], cmd: FormattingCommand) -> bool:
    """Append a command's replacement to the result, applying spacing rules.

    Also strips Whisper-inserted trailing punctuation from the preceding word
    when the trigger attaches to it (REMOVE_BEFORE, REMOVE_BOTH).

    Returns True if the next word should suppress its leading space.
    """
    if cmd.spacing in _ATTACHES_BEFORE:
        _strip_trailing_space(result_parts)
        _strip_trailing_punct_from_last(result_parts)
    elif result_parts:
        result_parts.append(" ")

    result_parts.append(cmd.replacement)
    return cmd.spacing in (SpacingRule.REMOVE_AFTER, SpacingRule.REMOVE_BOTH)


def _append_word(result_parts: List[str], word: str, suppress_space: bool) -> None:
    """Append a regular word, adding a space separator if needed."""
    if result_parts and not suppress_space:
        result_parts.append(" ")
    result_parts.append(word)


def _strip_trailing_space(parts: List[str]) -> None:
    """Remove a trailing space from the parts list."""
    if parts and parts[-1] == " ":
        parts.pop()


def _strip_trailing_punct_from_last(parts: List[str]) -> None:
    """Strip Whisper-inserted trailing punctuation from the last word.

    Whisper often adds commas/periods between words that the user intended
    as formatting commands, e.g. "Picture, slash" -> "Picture," + "/".
    Stripping the comma gives "Picture/" as intended.
    """
    if not parts or parts[-1] == " ":
        return
    stripped = _TRAILING_PUNCT_RE.sub("", parts[-1])
    if stripped:
        parts[-1] = stripped


def _deduplicate_punctuation(text: str) -> str:
    """Collapse adjacent duplicate punctuation separated by whitespace.

    Handles the case where Whisper auto-inserts punctuation (e.g. "Hello.")
    and the user also speaks "period", resulting in "Hello..".
    Preserves "..." (ellipsis) but collapses ".." to ".".
    """
    text = text.replace("...", _ELLIPSIS_PLACEHOLDER)

    for char in _DEDUP_PUNCTUATION:
        # Collapse "X<space(s)>X" into "X" where X is the same punctuation
        while char + " " + char in text:
            text = text.replace(char + " " + char, char)
        # Collapse "XX" into "X" (no space between)
        double = char + char
        while double in text:
            text = text.replace(double, char)

    text = text.replace(_ELLIPSIS_PLACEHOLDER, "...")
    return text
