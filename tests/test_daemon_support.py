"""Tests for shared daemon lifecycle helpers."""

from dictate.daemon_support import (
    cleanup_daemon_config,
    read_daemon_config,
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
