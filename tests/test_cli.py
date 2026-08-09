"""Tests for the dictate CLI."""

import argparse
from unittest.mock import patch

from dictate.live.cli import _daemon_config_matches


class TestDaemonConfigMatches:
    def _make_args(self, model="large-v3-turbo", device="cuda", compute_type="int8_float16"):
        return argparse.Namespace(
            model=model, device=device, compute_type=compute_type, quiet=False
        )

    def test_matches_when_config_identical(self):
        with patch(
            "dictate.live.cli.read_daemon_config",
            return_value={
                "model": "large-v3-turbo",
                "device": "cuda",
                "compute_type": "int8_float16",
            },
        ):
            assert _daemon_config_matches(self._make_args()) is True

    def test_mismatches_when_model_differs(self):
        with patch(
            "dictate.live.cli.read_daemon_config",
            return_value={
                "model": "base",
                "device": "cuda",
                "compute_type": "int8_float16",
            },
        ):
            assert _daemon_config_matches(self._make_args()) is False

    def test_mismatches_when_device_differs(self):
        with patch(
            "dictate.live.cli.read_daemon_config",
            return_value={
                "model": "large-v3-turbo",
                "device": "cpu",
                "compute_type": "int8_float16",
            },
        ):
            assert _daemon_config_matches(self._make_args()) is False

    def test_mismatches_when_config_missing(self):
        with patch("dictate.live.cli.read_daemon_config", return_value=None):
            assert _daemon_config_matches(self._make_args()) is False
