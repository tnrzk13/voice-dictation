"""Tests for shared model loading helpers."""

import argparse
from unittest.mock import patch

import pytest

from dictate.model_loader import add_model_args, is_model_cached


class TestAddModelArgs:
    def test_adds_defaults_from_config(self):
        parser = argparse.ArgumentParser()
        add_model_args(parser)
        args = parser.parse_args([])
        assert args.model == "large-v3-turbo"
        assert args.device == "cuda"
        assert args.compute_type == "int8_float16"
        assert args.quiet is False

    def test_parses_custom_values(self):
        parser = argparse.ArgumentParser()
        add_model_args(parser)
        args = parser.parse_args(
            ["--model", "base", "--device", "cpu", "--compute-type", "int8", "--quiet"]
        )
        assert args.model == "base"
        assert args.device == "cpu"
        assert args.compute_type == "int8"
        assert args.quiet is True


class TestIsModelCached:
    def test_returns_true_when_download_model_succeeds(self):
        faster_whisper = pytest.importorskip("faster_whisper")
        with patch.object(faster_whisper.utils, "download_model") as mock_download:
            assert is_model_cached("base") is True
            mock_download.assert_called_once_with("base", local_files_only=True)

    def test_returns_false_on_exception(self):
        faster_whisper = pytest.importorskip("faster_whisper")
        with patch.object(
            faster_whisper.utils, "download_model", side_effect=ValueError("not found")
        ):
            assert is_model_cached("base") is False
