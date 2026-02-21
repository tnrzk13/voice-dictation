"""Tests for ProgressiveTyper diff-based text correction."""

from unittest.mock import patch

from dictate.live.typer import ProgressiveTyper, _find_common_prefix_length


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
    def test_first_partial_types_everything(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        backspaces, typed = typer.apply_partial("hello")
        assert backspaces == 0
        assert typed == "hello"
        assert typer.displayed_text == "hello"

    def test_partial_extends_previous(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hel")
        backspaces, typed = typer.apply_partial("hello")
        assert backspaces == 0
        assert typed == "lo"
        assert typer.displayed_text == "hello"

    def test_partial_corrects_previous(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello wor")
        backspaces, typed = typer.apply_partial("hello world")
        assert backspaces == 0
        assert typed == "ld"

    def test_partial_replaces_divergent_text(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello there")
        backspaces, typed = typer.apply_partial("hello world")
        assert backspaces == 5  # delete "there"
        assert typed == "world"
        assert typer.displayed_text == "hello world"

    def test_partial_completely_replaces(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("foo")
        backspaces, typed = typer.apply_partial("bar")
        assert backspaces == 3
        assert typed == "bar"

    def test_partial_shortens_text(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello world")
        backspaces, typed = typer.apply_partial("hello")
        assert backspaces == 6  # delete " world"
        assert typed == ""
        assert typer.displayed_text == "hello"


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestProgressiveTyperFinals:
    def test_final_locks_text_with_trailing_space(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello")
        typer.apply_final("hello world")
        assert typer.committed == "hello world "
        assert typer.pending == ""
        assert typer.displayed_text == "hello world "

    def test_final_without_partial(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        backspaces, typed = typer.apply_final("hello")
        assert backspaces == 0
        assert typed == "hello "
        assert typer.committed == "hello "

    def test_partials_after_final_build_on_committed(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_final("hello")
        assert typer.committed == "hello "

        # Next partial - Vosk doesn't include the committed prefix
        backspaces, typed = typer.apply_partial("world")
        assert backspaces == 0
        assert typed == "world"
        assert typer.displayed_text == "hello world"

    def test_multiple_finals_accumulate(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_final("hello")
        typer.apply_final("world")
        assert typer.committed == "hello world "
        assert typer.displayed_text == "hello world "

    def test_final_corrects_partial(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("helo")
        backspaces, typed = typer.apply_final("hello")
        assert backspaces == 1  # delete "o" from "helo"
        assert typed == "lo "
        assert typer.committed == "hello "


@patch("dictate.live.typer._send_backspaces")
@patch("dictate.live.typer._type_text")
class TestProgressiveTyperXdotoolCalls:
    def test_no_xdotool_calls_when_nothing_changes(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("hello")
        mock_type.reset_mock()
        mock_bs.reset_mock()

        typer.apply_partial("hello")
        mock_bs.assert_not_called()
        mock_type.assert_not_called()

    def test_backspaces_sent_before_typing(self, mock_type, mock_bs):
        typer = ProgressiveTyper()
        typer.apply_partial("abc")
        mock_type.reset_mock()
        mock_bs.reset_mock()

        typer.apply_partial("axyz")
        mock_bs.assert_called_once_with(2)
        mock_type.assert_called_once_with("xyz")
