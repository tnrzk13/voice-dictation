"""Tests for ProgressiveTyper diff-based text correction."""

from unittest.mock import patch

from dictate.live.typer import (
    ProgressiveTyper,
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
