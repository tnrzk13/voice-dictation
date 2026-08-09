"""Tests for the input monitor."""

import threading
import time
from unittest.mock import MagicMock

from dictate.live.input_monitor import InputMonitor, XDOTOOL_GRACE_PERIOD


class TestInputMonitor:
    def _make_monitor(self):
        stop_event = threading.Event()
        typer = MagicMock()
        typer.is_typing = False
        typer.last_typed_at = 0.0
        return InputMonitor(stop_event, typer), stop_event, typer

    def test_click_during_typing_grace_period_is_ignored(self):
        monitor, stop_event, typer = self._make_monitor()
        typer.last_typed_at = time.time()
        monitor._started_at = time.time() - 2
        monitor._on_click(0, 0, None, True)
        assert not stop_event.is_set()

    def test_click_after_typing_grace_period_stops(self):
        monitor, stop_event, typer = self._make_monitor()
        typer.last_typed_at = time.time() - XDOTOOL_GRACE_PERIOD - 0.1
        monitor._started_at = time.time() - 2
        monitor._on_click(0, 0, None, True)
        assert stop_event.is_set()

    def test_click_during_startup_grace_period_is_ignored(self):
        monitor, stop_event, typer = self._make_monitor()
        typer.last_typed_at = 0
        monitor._started_at = time.time()
        monitor._on_click(0, 0, None, True)
        assert not stop_event.is_set()

    def test_release_click_is_ignored(self):
        monitor, stop_event, _ = self._make_monitor()
        monitor._on_click(0, 0, None, False)
        assert not stop_event.is_set()

    def test_key_during_typing_is_ignored(self):
        monitor, stop_event, typer = self._make_monitor()
        typer.is_typing = True
        monitor._on_key_press(None)
        assert not stop_event.is_set()
