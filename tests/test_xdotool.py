"""Tests for xdotool subprocess wrappers."""

from unittest.mock import MagicMock, patch

from dictate.xdotool import _run_xdotool, send_backspaces, type_text


class TestRunXdotool:
    @patch("dictate.xdotool.subprocess.run")
    def test_logs_warning_on_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr=b"window not found")
        with patch("dictate.xdotool.logger.warning") as mock_log:
            _run_xdotool(["type", "hello"])
            mock_log.assert_called_once()
            message = mock_log.call_args[0][0]
            assert "type" in message
            assert "window not found" in message

    @patch("dictate.xdotool.subprocess.run")
    def test_no_log_on_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        with patch("dictate.xdotool.logger.warning") as mock_log:
            _run_xdotool(["key", "Return"])
            mock_log.assert_not_called()


class TestTypeText:
    @patch("dictate.xdotool.subprocess.run")
    def test_type_call_uses_exact_argv(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        type_text("hello")
        assert mock_run.call_args[0][0] == [
            "xdotool",
            "type",
            "--clearmodifiers",
            "--delay",
            "30",
            "--",
            "hello",
        ]

    @patch("dictate.xdotool.subprocess.run")
    def test_splits_special_keys(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        type_text("hello\tworld\n")
        argvs = [call[0][0] for call in mock_run.call_args_list]
        assert argvs == [
            ["xdotool", "type", "--clearmodifiers", "--delay", "30", "--", "hello"],
            ["xdotool", "key", "--clearmodifiers", "Tab"],
            ["xdotool", "type", "--clearmodifiers", "--delay", "30", "--", "world"],
            ["xdotool", "key", "--clearmodifiers", "Return"],
        ]

    @patch("dictate.xdotool.subprocess.run")
    def test_payload_survives_verbatim_including_leading_dash(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        payload = "-n --delay 99 literal"
        type_text(payload)
        assert mock_run.call_args[0][0] == [
            "xdotool",
            "type",
            "--clearmodifiers",
            "--delay",
            "30",
            "--",
            payload,
        ]


class TestSendBackspaces:
    @patch("dictate.xdotool.subprocess.run")
    def test_backspace_call_uses_exact_argv(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        send_backspaces(2)
        assert mock_run.call_args[0][0] == [
            "xdotool",
            "key",
            "--clearmodifiers",
            "--delay",
            "3",
            "BackSpace",
            "BackSpace",
        ]
