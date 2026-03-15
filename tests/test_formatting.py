"""Tests for spoken command formatting."""

from dictate.live.formatting import apply_formatting_commands


class TestPunctuation:
    def test_period(self):
        assert apply_formatting_commands("hello period") == "hello."

    def test_comma(self):
        assert apply_formatting_commands("hello comma world") == "hello, world"

    def test_question_mark(self):
        assert apply_formatting_commands("how are you question mark") == "how are you?"

    def test_exclamation_mark(self):
        assert apply_formatting_commands("wow exclamation mark") == "wow!"

    def test_exclamation_point_alias(self):
        assert apply_formatting_commands("wow exclamation point") == "wow!"

    def test_colon(self):
        assert apply_formatting_commands("note colon details") == "note: details"

    def test_semicolon(self):
        assert apply_formatting_commands("first semicolon second") == "first; second"

    def test_ellipsis(self):
        assert apply_formatting_commands("wait ellipsis") == "wait..."

    def test_dot(self):
        assert apply_formatting_commands("hello dot") == "hello."


class TestBrackets:
    def test_parentheses(self):
        assert (
            apply_formatting_commands("open parenthesis hello close parenthesis")
            == "(hello)"
        )

    def test_paren_alias(self):
        assert apply_formatting_commands("open paren hello close paren") == "(hello)"

    def test_square_brackets(self):
        assert (
            apply_formatting_commands("open bracket hello close bracket") == "[hello]"
        )

    def test_braces(self):
        assert apply_formatting_commands("open brace hello close brace") == "{hello}"

    def test_angle_brackets(self):
        assert (
            apply_formatting_commands("open angle bracket hello close angle bracket")
            == "<hello>"
        )


class TestQuotes:
    def test_double_quotes(self):
        assert apply_formatting_commands("open quote hello close quote") == '"hello"'

    def test_single_quotes(self):
        assert (
            apply_formatting_commands("open single quote hello close single quote")
            == "'hello'"
        )

    def test_backtick(self):
        assert apply_formatting_commands("say backtick code backtick end") == "say`code`end"


class TestPathSeparators:
    def test_slash(self):
        assert (
            apply_formatting_commands("tony slash pictures slash screenshots")
            == "tony/pictures/screenshots"
        )

    def test_backslash(self):
        assert apply_formatting_commands("C backslash Users") == "C\\Users"

    def test_hyphen(self):
        assert apply_formatting_commands("well hyphen known") == "well-known"

    def test_dash(self):
        assert apply_formatting_commands("well dash known") == "well-known"

    def test_underscore(self):
        assert apply_formatting_commands("my underscore variable") == "my_variable"


class TestProgrammingSymbols:
    def test_equals_sign(self):
        assert apply_formatting_commands("x equals sign y") == "x = y"

    def test_plus_sign(self):
        assert apply_formatting_commands("x plus sign y") == "x + y"

    def test_minus_sign(self):
        assert apply_formatting_commands("x minus sign y") == "x - y"

    def test_at_sign(self):
        assert apply_formatting_commands("user at sign example") == "user@example"

    def test_hash_sign(self):
        assert apply_formatting_commands("hash sign include") == "#include"

    def test_dollar_sign(self):
        assert apply_formatting_commands("dollar sign HOME") == "$HOME"

    def test_percent_sign(self):
        assert apply_formatting_commands("100 percent sign done") == "100%done"

    def test_caret(self):
        assert apply_formatting_commands("x caret 2") == "x^2"

    def test_ampersand(self):
        assert apply_formatting_commands("a ampersand b") == "a & b"

    def test_asterisk(self):
        assert apply_formatting_commands("x asterisk y") == "x*y"

    def test_pipe(self):
        assert apply_formatting_commands("a pipe b") == "a | b"

    def test_tilde(self):
        assert apply_formatting_commands("tilde home") == "~home"


class TestNewlines:
    def test_new_line(self):
        assert apply_formatting_commands("hello new line world") == "hello\nworld"

    def test_new_paragraph(self):
        assert apply_formatting_commands("hello new paragraph world") == "hello\n\nworld"

    def test_tab_key(self):
        assert apply_formatting_commands("hello tab key world") == "hello\tworld"


class TestPluralAliases:
    def test_open_parentheses_plural(self):
        assert apply_formatting_commands("open parentheses hello close parentheses") == "(hello)"

    def test_singular_still_works(self):
        assert apply_formatting_commands("open parenthesis hello close parenthesis") == "(hello)"


class TestWhisperPunctuation:
    def test_trigger_with_trailing_comma(self):
        """Whisper may auto-insert comma after a trigger word."""
        assert apply_formatting_commands("tony slash, pictures") == "tony/pictures"

    def test_trigger_with_trailing_period(self):
        assert apply_formatting_commands("hello period.") == "hello."

    def test_multi_word_trigger_with_trailing_punct(self):
        assert apply_formatting_commands("open parenthesis, hello") == "(hello"

    def test_comma_on_word_before_trigger(self):
        """Whisper comma on the word before a trigger gets stripped."""
        assert (
            apply_formatting_commands("Tony Slash, Picture, Slash, screenshots")
            == "Tony/Picture/screenshots"
        )

    def test_comma_before_period_trigger(self):
        assert apply_formatting_commands("hello, period") == "hello."

    def test_preserves_punct_on_non_adjacent_words(self):
        """Punctuation on words not adjacent to triggers is preserved."""
        assert apply_formatting_commands("hello, world") == "hello, world"


class TestCaseInsensitive:
    def test_capitalized_trigger(self):
        assert apply_formatting_commands("hello Period") == "hello."

    def test_all_caps_trigger(self):
        assert apply_formatting_commands("hello PERIOD") == "hello."

    def test_mixed_case_multi_word(self):
        assert apply_formatting_commands("Open Parenthesis hello Close Parenthesis") == "(hello)"


class TestWordBoundaries:
    def test_slashing_not_replaced(self):
        assert apply_formatting_commands("slashing prices") == "slashing prices"

    def test_periodically_not_replaced(self):
        assert apply_formatting_commands("periodically updated") == "periodically updated"

    def test_dashboard_not_replaced(self):
        assert apply_formatting_commands("dashboard view") == "dashboard view"

    def test_piped_not_replaced(self):
        assert apply_formatting_commands("piped water") == "piped water"


class TestPunctuationDedup:
    def test_whisper_period_plus_spoken(self):
        """Whisper auto-inserts period, user also says 'period'."""
        assert apply_formatting_commands("Hello. period") == "Hello."

    def test_whisper_question_plus_spoken(self):
        assert apply_formatting_commands("How are you? question mark") == "How are you?"

    def test_whisper_comma_plus_spoken(self):
        assert apply_formatting_commands("Hello, comma world") == "Hello, world"

    def test_no_dedup_when_not_adjacent(self):
        assert apply_formatting_commands("a period b period") == "a. b."


class TestNoOp:
    def test_no_commands_returns_unchanged(self):
        assert apply_formatting_commands("hello world") == "hello world"

    def test_empty_string(self):
        assert apply_formatting_commands("") == ""

    def test_single_word(self):
        assert apply_formatting_commands("hello") == "hello"


class TestMixed:
    def test_parenthesized_list(self):
        assert (
            apply_formatting_commands(
                "open parenthesis hello comma world close parenthesis"
            )
            == "(hello, world)"
        )

    def test_path(self):
        assert (
            apply_formatting_commands("tony slash pictures slash screenshots")
            == "tony/pictures/screenshots"
        )

    def test_adjacent_commands(self):
        assert apply_formatting_commands("close paren period") == ")."

    def test_sentence_with_punctuation(self):
        assert (
            apply_formatting_commands("hello comma how are you question mark")
            == "hello, how are you?"
        )

    def test_programming_expression(self):
        assert (
            apply_formatting_commands("result equals sign x plus sign y")
            == "result = x + y"
        )
