#!/usr/bin/env python3
"""Shared fixture definitions for audio regression tests.

Both the command-line recorder and the GUI fixture manager use these.
Each fixture has a human-readable script, directions for how to perform it,
and a note on what edge case it exercises.
"""

from typing import Dict


FIXTURES: Dict[str, Dict[str, str]] = {
    "simple_sentence": {
        "script": "Hello world, this is a test.",
        "directions": "Speak at a normal, steady pace.",
        "focus": "Tests clean transcription and full commit.",
    },
    "formatting_commands": {
        "script": "Tony slash pictures from the beach period.",
        "directions": "Speak clearly. Use the literal words 'slash' and 'period'.",
        "focus": "Tests spoken formatting commands become symbols.",
    },
    "mid_sentence_pause": {
        "script": "This is the first part. And this is the second part.",
        "directions": "Say the first sentence, then pause for about 2 seconds before continuing.",
        "focus": "Tests re-transcription across a pause without duplicating text.",
    },
    "false_start": {
        "script": "I went to the store, I mean, the park.",
        "directions": "Start saying 'store', hesitate, then correct yourself to 'park'.",
        "focus": "Tests recovery from a self-correction.",
    },
    "filler_words": {
        "script": "So um I think we should uh consider this option.",
        "directions": "Include natural-sounding 'um' and 'uh' between words.",
        "focus": "Tests that filler words don't break the chunk processor.",
    },
    "long_sentence": {
        "script": (
            "The quick brown fox jumps over the lazy dog "
            "while the farmer watches from the porch."
        ),
        "directions": "Speak continuously for more than 5 seconds.",
        "focus": "Tests multiple 2-second transcription intervals.",
    },
    "hotwords": {
        "script": (
            "The Claude Code workflow uses Jira tickets, git commits, "
            "GitHub pull requests, sudo access, tmux sessions, and YAML configs."
        ),
        "directions": (
            "Speak naturally and include each word from hotwords.txt: "
            "Claude, Code, Jira, git, sudo, tmux, YAML, GitHub."
        ),
        "focus": "Tests that Whisper hotword bias is applied to your actual terms.",
    },
    "numbers_and_symbols": {
        "script": (
            "The price is forty two dollars and fifty cents. "
            "Order number 8, 4, 7, 2, plus one percent."
        ),
        "directions": "Speak numbers naturally, not as individual digits unless shown as digits.",
        "focus": "Tests number recognition and spoken symbols like plus and percent.",
    },
    "punctuation_variety": {
        "script": (
            "Wait, what? Yes, please. Open quote hello world close quote. "
            "Use a colon, then a semicolon; finally, an ellipsis."
        ),
        "directions": (
            "Speak the punctuation commands clearly and separately: "
            "comma, question mark, period, open quote, close quote, colon, semicolon, ellipsis."
        ),
        "focus": "Tests a variety of spoken punctuation commands in one sentence.",
    },
    "code_snippet": {
        "script": (
            "Open bracket apple close bracket dot open parenthesis banana close parenthesis "
            "slash open brace cherry close brace."
        ),
        "directions": (
            "Speak code-like syntax clearly. Use 'open bracket', 'close bracket, "
            "'open parenthesis', 'close parenthesis', 'open brace', 'close brace', 'slash', 'dot'."
        ),
        "focus": "Tests bracket, brace, and path formatting useful for coding dictation.",
    },
    "repetition_recovery": {
        "script": "The quick brown fox jumps over the lazy dog.",
        "directions": (
            "Speak the first three words twice: 'the the the quick brown fox jumps over the lazy dog'. "
            "This simulates a Whisper stutter to test the typer does not duplicate."
        ),
        "focus": "Tests that repeated words from the model do not cause duplicated text.",
    },
}
