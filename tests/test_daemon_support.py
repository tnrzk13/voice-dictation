"""Tests for shared daemon lifecycle helpers."""

from unittest.mock import MagicMock, patch

from dictate.daemon_support import (
    cleanup_daemon_config,
    read_daemon_config,
    start_daemon_process,
    wait_for_socket,
    write_daemon_config,
)


class TestDaemonConfig:
    def test_round_trip(self, tmp_path):
        socket_path = str(tmp_path / "daemon.sock")
        config = {
            "model": "base",
            "device": "cpu",
            "compute_type": "int8",
            "quiet": True,
        }
        write_daemon_config(socket_path, config)
        assert read_daemon_config(socket_path) == config

    def test_read_missing_returns_none(self, tmp_path):
        socket_path = str(tmp_path / "daemon.sock")
        assert read_daemon_config(socket_path) is None

    def test_cleanup_removes_config(self, tmp_path):
        socket_path = str(tmp_path / "daemon.sock")
        write_daemon_config(socket_path, {"model": "base"})
        cleanup_daemon_config(socket_path)
        assert read_daemon_config(socket_path) is None


class TestWaitForSocket:
    def test_short_timeout_sleeps_at_least_once(self):
        """A timeout shorter than poll_interval should still sleep and check."""
        with patch("dictate.daemon_support.is_daemon_running", return_value=False):
            result = wait_for_socket("/tmp/nonexistent.sock", timeout=0.01, poll_interval=0.1)
            assert result is False


class TestStartDaemonProcess:
    def test_returns_false_when_process_dies_immediately(self):
        with patch("dictate.daemon_support.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 1
            mock_popen.return_value = mock_proc
            result = start_daemon_process(
                "entry", "module", "/tmp/test.sock", timeout=0.01, poll_interval=0.01
            )
            assert result is False

