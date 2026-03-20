"""Tests for ProgressiveTyper diff-based text correction."""

from unittest.mock import call, patch

from dictate.live.typer import (
    ProgressiveTyper,
    _count_common_prefix_words,
    _find_common_prefix_length,
    _capitalize_first,
    _strip_punctuation,
)


class TestCapitalizeFirst:
    def test_capitalizes_lowercase(self):
        assert _capitalize_first("hello") == "Hello"

    def test_already_capitalized(self):
        assert _capitalize_first("Hello") == "Hello"

    def test_empty_string(self):
        assert _capitalize_first("") == ""

    def test_single_char(self):
        assert _capitalize_first("h") == "H"

    def test_preserves_rest(self):
        assert _capitalize_first("hELLO") == "HELLO"


class TestStripPunctuation:
    def test_removes_comma(self):
        assert _strip_punctuation("hello, world") == "hello world"

    def test_removes_period(self):
        assert _strip_punctuation("hello world.") == "hello world"

    def test_removes_question_mark(self):
        assert _strip_punctuation("how are you?") == "how are you"

    def test_removes_exclamation(self):
        assert _strip_punctuation("wow!") == "wow"

    def test_no_punctuation(self):
        assert _strip_punctuation("hello world") == "hello world"

    def test_empty_string(self):
        assert _strip_punctuation("") == ""


class TestFindCommonPrefixLength:
    def test_identical_strings(self):
        assert _find_common_prefix_length("hello", "hello") == 5

    def test_no_common_prefix(self):
        assert _find_common_prefix_length("abc", "xyz") == 0

    def test_partial_prefix(self):
        assert _find_common_prefix_length("hello world", "hello there") == 6

    def test_one_empty(self):
        assert _find_common_prefix_length("", "hello") == 0
        assert _find_common_prefix_length("hello", "") == 0

    def test_both_empty(self):
        assert _find_common_prefix_length("", "") == 0

    def test_shorter_is_prefix_of_longer(self):
        assert _find_common_prefix_length("hel", "hello") == 3
        assert _find_common_prefix_length("hello", "hel") == 3


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestProgressiveTyperPartials:
    def test_first_partial_capitalizes(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        backspaces, typed = typer.apply_partial("hello")
        assert backspaces == 0
        assert typed == "Hello"
        assert typer.displayed_text == "Hello"

    def test_partial_extends_previous(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hel")  # pending = "Hel"
        backspaces, typed = typer.apply_partial("hello")  # capitalize -> "Hello"
        assert backspaces == 0
        assert typed == "lo"
        assert typer.displayed_text == "Hello"

    def test_partial_corrects_previous(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello wor")  # pending = "Hello wor"
        backspaces, typed = typer.apply_partial("hello world")  # "Hello world"
        assert backspaces == 0
        assert typed == "ld"

    def test_partial_replaces_divergent_text(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello there")  # pending = "Hello there"
        backspaces, typed = typer.apply_partial("hello world")  # "Hello world"
        assert backspaces == 5  # delete "there"
        assert typed == "world"
        assert typer.displayed_text == "Hello world"

    def test_partial_completely_replaces(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("foo")  # pending = "Foo"
        backspaces, typed = typer.apply_partial("bar")  # "Bar"
        assert backspaces == 3
        assert typed == "Bar"

    def test_partial_shortens_text(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello world")  # pending = "Hello world"
        backspaces, typed = typer.apply_partial("hello")  # "Hello"
        assert backspaces == 6  # delete " world"
        assert typed == ""
        assert typer.displayed_text == "Hello"


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestProgressiveTyperFinals:
    def test_final_after_partial_only_adds_space(self, mock_type, mock_bs):
        """Partial already capitalized, so final just adds trailing space."""
        typer = ProgressiveTyper()
        typer.apply_partial("hello")
        assert typer.pending == "Hello"
        backspaces, typed = typer.apply_final("hello world")
        assert backspaces == 0
        assert typed == " world "
        assert typer.committed == "Hello world "
        assert typer.displayed_text == "Hello world "

    def test_final_without_partial(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        backspaces, typed = typer.apply_final("hello")
        assert backspaces == 0
        assert typed == "Hello "
        assert typer.committed == "Hello "

    def test_final_capitalizes_each_sentence(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_final("hello")
        typer.apply_final("world")
        assert typer.committed == "Hello World "
        assert typer.displayed_text == "Hello World "

    def test_partials_after_final_capitalize_new_sentence(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_final("hello")
        assert typer.committed == "Hello "

        # First partial of new sentence capitalizes
        backspaces, typed = typer.apply_partial("world")
        assert backspaces == 0
        assert typed == "World"
        assert typer.displayed_text == "Hello World"

    def test_final_corrects_partial(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("helo")
        # Partial was capitalized to "Helo", final capitalizes to "Hello "
        backspaces, typed = typer.apply_final("hello")
        assert backspaces == 1  # delete "o" from "Helo" (common prefix "Hel")
        assert typed == "lo "
        assert typer.committed == "Hello "

    def test_case_insensitive_prefix_stripping(self, mock_type, mock_bs):
        """Partial result may include previously committed text in lowercase."""
        typer = ProgressiveTyper()
        typer.apply_final("hello")
        assert typer.committed == "Hello "
        # Partial sends "hello world" (lowercase prefix matching committed)
        backspaces, typed = typer.apply_partial("hello world")
        assert typed == "World"
        assert typer.displayed_text == "Hello World"

    def test_prefix_stripping_ignores_punctuation_in_committed(self, mock_type, mock_bs):
        """Committed text may have punctuation, but partials may not."""
        typer = ProgressiveTyper()
        # Simulate a punctuated final being committed
        typer._committed = "Hello, world. "
        typer._pending = ""
        # Next partial arrives without punctuation
        backspaces, typed = typer.apply_partial("hello world this is new")
        assert typed == "This is new"
        assert typer.displayed_text == "Hello, world. This is new"

    def test_prefix_stripping_with_question_mark(self, mock_type, mock_bs):
        """Question marks in committed text don't break prefix matching."""
        typer = ProgressiveTyper()
        typer._committed = "How are you? "
        typer._pending = ""
        backspaces, typed = typer.apply_partial("how are you doing")
        assert typed == "Doing"
        assert typer.displayed_text == "How are you? Doing"


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestProgressiveTyperXdotoolCalls:
    def test_no_xdotool_calls_when_nothing_changes(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello")  # "Hello" on screen
        mock_type.reset_mock()
        mock_bs.reset_mock()

        typer.apply_partial("hello")  # same text, no change
        mock_bs.assert_not_called()
        mock_type.assert_not_called()

    def test_backspaces_sent_before_typing(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("abc")  # "Abc" on screen
        mock_type.reset_mock()
        mock_bs.reset_mock()

        typer.apply_partial("axyz")  # pending is "Abc", new is "axyz" (not capitalized since pending exists)
        mock_bs.assert_called_once_with(2)
        mock_type.assert_called_once_with("xyz")


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestIsTypingFlag:
    def test_is_typing_true_during_xdotool_calls(self, mock_type, mock_bs):
        """is_typing is True while xdotool subprocess is running."""
        typer = ProgressiveTyper()
        observed = []

        def capture_flag(text):
            observed.append(typer.is_typing)

        mock_type.side_effect = capture_flag
        typer._execute_edit(0, "hello")

        assert observed == [True]
        assert typer.is_typing is False

    def test_is_typing_false_after_exception(self, mock_type, mock_bs):
        """is_typing resets to False even if xdotool raises."""
        typer = ProgressiveTyper()
        mock_type.side_effect = OSError("xdotool crashed")

        try:
            typer._execute_edit(0, "hello")
        except OSError:
            pass

        assert typer.is_typing is False

    def test_is_typing_false_when_nothing_to_do(self, mock_type, mock_bs):
        """is_typing stays False when there's no edit to perform."""
        typer = ProgressiveTyper()
        typer._execute_edit(0, "")
        assert typer.is_typing is False


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestFormattingIntegration:
    def test_final_applies_formatting_and_capitalization(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        backspaces, typed = typer.apply_final("tony slash pictures")
        assert backspaces == 0
        assert typed == "Tony/pictures "
        assert typer.committed == "Tony/pictures "

    def test_final_formatting_with_punctuation(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        backspaces, typed = typer.apply_final("hello period")
        assert backspaces == 0
        assert typed == "Hello. "
        assert typer.committed == "Hello. "

    def test_final_formatting_new_line(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        backspaces, typed = typer.apply_final("hello new line world")
        assert backspaces == 0
        assert typed == "Hello\nworld "

    def test_final_whisper_period_dedup(self, mock_type, mock_bs):
        """Whisper auto-period + spoken 'period' collapses to single period."""
        typer = ProgressiveTyper()
        backspaces, typed = typer.apply_final("hello. period")
        assert typed == "Hello. "


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
@patch("dictate.live.typer.time")
class TestBackspaceSettleDelay:
    def test_sleeps_between_backspaces_and_typing(self, mock_time, mock_type, mock_bs):
        """When both backspaces and typing are needed, a settle delay is inserted."""
        mock_time.time.return_value = 1000.0
        typer = ProgressiveTyper()
        typer._pending = "Hello there"
        typer._execute_edit(5, "world")

        mock_bs.assert_called_once_with(5)
        mock_time.sleep.assert_called_once_with(0.05)
        mock_type.assert_called_once_with("world")

    def test_no_sleep_when_only_backspaces(self, mock_time, mock_type, mock_bs):
        mock_time.time.return_value = 1000.0
        typer = ProgressiveTyper()
        typer._execute_edit(3, "")

        mock_bs.assert_called_once_with(3)
        mock_time.sleep.assert_not_called()
        mock_type.assert_not_called()

    def test_no_sleep_when_only_typing(self, mock_time, mock_type, mock_bs):
        mock_time.time.return_value = 1000.0
        typer = ProgressiveTyper()
        typer._execute_edit(0, "hello")

        mock_bs.assert_not_called()
        mock_time.sleep.assert_not_called()
        mock_type.assert_called_once_with("hello")


class TestCountCommonPrefixWords:
    def test_identical_lists(self):
        assert _count_common_prefix_words(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_partial_overlap(self):
        assert _count_common_prefix_words(["a", "b", "c"], ["a", "b", "x"]) == 2

    def test_no_overlap(self):
        assert _count_common_prefix_words(["a"], ["b"]) == 0

    def test_empty_lists(self):
        assert _count_common_prefix_words([], []) == 0
        assert _count_common_prefix_words(["a"], []) == 0
        assert _count_common_prefix_words([], ["a"]) == 0

    def test_case_insensitive(self):
        assert _count_common_prefix_words(["Hello", "World"], ["hello", "world"]) == 2

    def test_different_lengths(self):
        assert _count_common_prefix_words(["a", "b"], ["a", "b", "c"]) == 2
        assert _count_common_prefix_words(["a", "b", "c"], ["a", "b"]) == 2


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestAutoCommitStableWords:
    def test_no_commit_below_threshold(self, mock_type, mock_bs):
        """One matching partial isn't enough to commit (threshold is 2)."""
        typer = ProgressiveTyper()
        typer.apply_partial("hello world foo bar")
        typer.apply_partial("hello world foo bar baz")
        # stable_count=1, below threshold - nothing committed
        assert typer.committed == ""
        assert "Hello world foo bar baz" in typer.displayed_text

    def test_commits_after_threshold(self, mock_type, mock_bs):
        """Two consecutive matching partials triggers auto-commit."""
        typer = ProgressiveTyper()
        typer.apply_partial("hello world foo bar")
        typer.apply_partial("hello world foo bar baz")
        typer.apply_partial("hello world foo bar baz qux")
        # 3 words match ("Hello world foo") across 2 consecutive partials
        # commit_count = match - KEEP_TAIL_WORDS = at least 1
        assert typer.committed != ""

    def test_keeps_tail_words_uncommitted(self, mock_type, mock_bs):
        """Last KEEP_TAIL_WORDS words stay in pending for corrections."""
        typer = ProgressiveTyper()
        typer.apply_partial("alpha bravo charlie delta")
        typer.apply_partial("alpha bravo charlie delta echo")
        typer.apply_partial("alpha bravo charlie delta echo foxtrot")
        # After 3rd partial: match=5 words, commit 5-2=3 words
        # committed should have first 3 words, pending has last 2+ new
        assert "Alpha bravo charlie " in typer.committed
        assert typer.pending.startswith("delta echo")

    def test_no_commit_when_match_lte_keep_tail(self, mock_type, mock_bs):
        """Don't commit if matching prefix is <= KEEP_TAIL_WORDS."""
        typer = ProgressiveTyper()
        typer.apply_partial("hello world")
        typer.apply_partial("hello world")
        typer.apply_partial("hello world")
        # match=2, KEEP_TAIL_WORDS=2, so match <= KEEP_TAIL_WORDS - no commit
        assert typer.committed == ""

    def test_stability_resets_on_divergence(self, mock_type, mock_bs):
        """Stability counter resets when words don't match."""
        typer = ProgressiveTyper()
        typer.apply_partial("hello world foo")
        typer.apply_partial("hello world foo bar")
        # stable_count=1
        typer.apply_partial("completely different text here")
        # stable_count reset to 0
        typer.apply_partial("completely different text here now")
        # stable_count=1 again, not enough
        assert typer.committed == ""

    def test_case_insensitive_matching(self, mock_type, mock_bs):
        """Auto-commit uses case-insensitive matching for stability."""
        typer = ProgressiveTyper()
        # First partial capitalizes to "Hello world foo bar"
        typer.apply_partial("hello world foo bar")
        # Second partial: _capitalize_first -> "Hello world foo bar baz"
        # Old pending "Hello world foo bar" vs new "Hello world foo bar baz"
        # Case-insensitive match on all 4 words
        typer.apply_partial("hello world foo bar baz")
        typer.apply_partial("hello world foo bar baz qux")
        # Should have committed despite capitalize_first casing
        assert typer.committed != ""

    def test_committed_prefix_stripping_after_auto_commit(self, mock_type, mock_bs):
        """After auto-commit, _strip_committed_prefix removes committed words."""
        typer = ProgressiveTyper()
        typer.apply_partial("alpha bravo charlie delta")
        typer.apply_partial("alpha bravo charlie delta echo")
        typer.apply_partial("alpha bravo charlie delta echo foxtrot")
        committed_before = typer.committed
        # Next partial includes the committed words - they should be stripped
        typer.apply_partial("alpha bravo charlie delta echo foxtrot golf")
        # The committed text should still contain the auto-committed words
        assert committed_before in typer.committed or typer.committed.startswith(committed_before)


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestBackspaceCap:
    def test_skips_partial_exceeding_cap(self, mock_type, mock_bs):
        """Partial needing >20 backspaces is skipped entirely."""
        typer = ProgressiveTyper()
        typer._pending = "This is a long sentence that was typed out already"
        mock_type.reset_mock()
        mock_bs.reset_mock()
        # New partial would replace most of the text (>20 backspaces)
        backspaces, typed = typer.apply_partial("completely different")
        assert backspaces == 0
        assert typed == ""
        # Pending unchanged
        assert typer._pending == "This is a long sentence that was typed out already"
        mock_bs.assert_not_called()
        mock_type.assert_not_called()

    def test_allows_partial_within_cap(self, mock_type, mock_bs):
        """Partial needing <=20 backspaces proceeds normally."""
        typer = ProgressiveTyper()
        typer.apply_partial("hello world")
        mock_type.reset_mock()
        mock_bs.reset_mock()
        backspaces, typed = typer.apply_partial("hello there")
        assert backspaces == 5  # "world" -> "there"
        assert typed == "there"

    def test_cap_does_not_apply_to_final(self, mock_type, mock_bs):
        """Finals are authoritative - cap doesn't apply."""
        typer = ProgressiveTyper()
        typer._pending = "This is a long sentence that was typed out already"
        backspaces, typed = typer.apply_final("completely different text")
        # Final should proceed even with many backspaces
        assert typer.committed == "Completely different text "
        assert typer.pending == ""

    def test_recovery_after_skip(self, mock_type, mock_bs):
        """After a skipped partial, next reasonable partial applies normally."""
        typer = ProgressiveTyper()
        typer._pending = "Hello world this is a long test sentence here"
        mock_type.reset_mock()
        mock_bs.reset_mock()
        # This would need too many backspaces - skipped
        typer.apply_partial("completely different")
        assert typer._pending == "Hello world this is a long test sentence here"
        # Next partial is reasonable - applies
        backspaces, typed = typer.apply_partial("Hello world this is a long test sentence here now")
        assert typed == " now"
        assert backspaces == 0


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestAutoCommitAndCapIntegration:
    def test_auto_commit_reduces_backspaces_below_cap(self, mock_type, mock_bs):
        """Auto-commit shrinks pending, keeping revisions under the cap."""
        typer = ProgressiveTyper()
        # Build up a long sentence over several partials
        typer.apply_partial("I would like to visit Japan and see how")
        typer.apply_partial("I would like to visit Japan and see how a small")
        typer.apply_partial("I would like to visit Japan and see how a small town")
        # By now, early words should be auto-committed
        assert typer.committed != ""
        # A revision to the tail stays under the cap because most text is committed
        backspaces, typed = typer.apply_partial(
            "I would like to visit Japan and see how a small village"
        )
        # "town" -> "village" is a small edit, not a catastrophic rewrite
        assert backspaces <= 20

    def test_bug_scenario_whisper_revision_preserves_stable_text(self, mock_type, mock_bs):
        """Reproduces the bug: Whisper drops middle of a long dictation."""
        typer = ProgressiveTyper()
        # Simulate growing partials during a long dictation
        typer.apply_partial("I would like to visit Japan")
        typer.apply_partial("I would like to visit Japan and see")
        typer.apply_partial("I would like to visit Japan and see how")
        typer.apply_partial("I would like to visit Japan and see how a small")
        typer.apply_partial("I would like to visit Japan and see how a small town is like")
        typer.apply_partial(
            "I would like to visit Japan and see how a small town is like especially"
        )
        # At this point, early words should be auto-committed
        committed_snapshot = typer.committed
        assert "I would like to visit Japan " in committed_snapshot
        # Whisper now revises, dropping the middle section
        # This would previously cause 50+ backspaces
        typer.apply_partial(
            "I would like to visit Japan and see to Osaka and see how the average"
        )
        # The committed text should be preserved (not overwritten)
        assert typer.committed.startswith(committed_snapshot)
        # The displayed text should still contain the stable prefix
        assert typer.displayed_text.startswith(committed_snapshot)
