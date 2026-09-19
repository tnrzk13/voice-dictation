"""Tests for ProgressiveTyper diff-based text correction."""

from unittest.mock import patch

from dictate.live.typer import (
    ProgressiveTyper,
    _capitalize_first,
    _finalized_prefix,
    _find_common_prefix_length,
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


class TestFinalizedPrefix:
    def test_empty_finalized_yields_no_committed_text(self):
        assert _finalized_prefix("Hello world", "") == ""

    def test_returns_common_prefix_with_finalized(self):
        assert _finalized_prefix("Hello world foo", "hello world") == "Hello world"

    def test_trailing_formatting_command_merges_with_next_word(self):
        """A finalized "slash" formats only once the next word is present."""
        assert _finalized_prefix("Tony/pictures", "tony slash") == "Tony/"

    def test_finalized_punctuation_not_present_in_target(self):
        """Whisper may drop punctuation the finalized text carried."""
        assert _finalized_prefix("Hello world again", "hello world.") == "Hello world"


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
class TestDaemonDrivenCommit:
    def test_finalized_sets_committed_boundary(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello world foo", finalized="hello world")
        assert typer.committed == "Hello world"
        assert typer.pending == " foo"
        assert typer.displayed_text == "Hello world foo"

    def test_without_finalized_nothing_is_committed(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello world")
        assert typer.committed == ""
        assert typer.pending == "Hello world"

    def test_committed_grows_as_finalized_grows(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("picture slash pictures", finalized="")
        assert typer.displayed_text == "Picture/pictures"

        typer.apply_partial(
            "picture slash pictures from", finalized="picture slash pictures"
        )
        assert typer.committed == "Picture/pictures"
        assert typer.pending == " from"
        assert typer.displayed_text == "Picture/pictures from"

    def test_dropped_word_in_tail_does_not_duplicate(self, mock_type, mock_bs):
        """Whisper deleting "Um" after it scrolled into the committed region
        must correct in place, not retype the visible prefix.
        """
        typer = ProgressiveTyper()
        typer.apply_partial("Great, I like this a lot better. Um, but can you")
        typer.apply_partial(
            "Great, I like this a lot better. Um, but can you tell me",
            finalized="Great, I like this a lot better.",
        )
        typer.apply_partial(
            "Great, I like this a lot better but can you tell me what's",
            finalized="Great, I like this a lot better.",
        )

        text = typer.displayed_text
        assert text.count("Great, I like this a lot better") == 1
        assert text.endswith("but can you tell me what's")

    def test_revision_inside_committed_region_corrects_in_place(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("I saw a quick brown fox", finalized="I saw a quick")
        assert typer.committed == "I saw a quick"

        # "quick" is revised to "quik" after it was committed
        typer.apply_partial("I saw a quik brown fox", finalized="I saw a quick")
        assert typer.displayed_text.count("brown fox") == 1
        assert "quick brown" not in typer.displayed_text


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestProgressiveTyperFinals:
    def test_final_after_partial_only_adds_space(self, mock_type, mock_bs):
        """Partial already capitalized, so final just adds trailing space."""
        typer = ProgressiveTyper()
        typer.apply_partial("hello")
        backspaces, typed = typer.apply_final("hello world")
        assert backspaces == 0
        assert typed == " world "
        assert typer.committed == "Hello world "
        assert typer.pending == ""
        assert typer.displayed_text == "Hello world "

    def test_final_without_partial(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        backspaces, typed = typer.apply_final("hello")
        assert backspaces == 0
        assert typed == "Hello "
        assert typer.committed == "Hello "

    def test_final_corrects_partial(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("helo")
        # Partial was capitalized to "Helo", final capitalizes to "Hello "
        backspaces, typed = typer.apply_final("hello")
        assert backspaces == 1  # delete "o" from "Helo" (common prefix "Hel")
        assert typed == "lo "
        assert typer.committed == "Hello "

    def test_final_uses_daemon_finalized_as_committed(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello world foo", finalized="hello world")
        typer.apply_final("hello world foo")
        assert typer.committed == "Hello world foo "
        assert typer.pending == ""


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestProgressiveTyperXdotoolCalls:
    def test_no_xdotool_calls_when_nothing_changes(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello")  # "Hello" on screen
        mock_type.reset_mock()
        mock_bs.reset_mock()

        typer.apply_partial("hello", finalized="hello")  # same text, no change
        mock_bs.assert_not_called()
        mock_type.assert_not_called()

    def test_backspaces_sent_before_typing(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("abc")  # "Abc" on screen
        mock_type.reset_mock()
        mock_bs.reset_mock()

        typer.apply_partial("axyz")  # pending is "Abc", new is "Axyz"
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

    def test_retranscribed_partial_with_formatting_command_does_not_duplicate(
        self, mock_type, mock_bs
    ):
        """A cumulative partial that repeats committed formatting commands is
        diffed against the screen, so only the new suffix is typed.
        """
        typer = ProgressiveTyper()
        typer.apply_partial("hello slash world", finalized="")
        assert typer.displayed_text == "Hello/world"

        backspaces, typed = typer.apply_partial(
            "hello slash world how are you", finalized="hello slash world"
        )
        assert typed == " how are you"
        assert typer.displayed_text == "Hello/world how are you"


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


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestEmptyPartials:
    def test_empty_partial_does_not_delete_pending(self, mock_type, mock_bs):
        """A transient empty partial from Whisper should not erase text."""
        typer = ProgressiveTyper()
        typer.apply_partial("hello")
        mock_type.reset_mock()
        mock_bs.reset_mock()
        backspaces, typed = typer.apply_partial("")
        assert backspaces == 0
        assert typed == ""
        assert typer.displayed_text == "Hello"
        mock_bs.assert_not_called()
        mock_type.assert_not_called()

    def test_whitespace_only_partial_is_ignored(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        backspaces, typed = typer.apply_partial("   ")
        assert backspaces == 0
        assert typed == ""
        mock_bs.assert_not_called()
        mock_type.assert_not_called()
