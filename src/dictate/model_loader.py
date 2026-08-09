"""Shared Whisper model loading and CLI argument helpers."""

import argparse
import ctypes
import logging
import os
import sys
from typing import Optional

from .config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)
from .system import notify


def is_model_cached(model_size: str) -> bool:
    """Check if the Whisper model is already downloaded locally."""
    try:
        from faster_whisper.utils import download_model

        download_model(model_size, local_files_only=True)
        return True
    except Exception:
        return False


def load_whisper_model(
    model_size: str,
    device: str,
    compute_type: str,
    quiet: bool = False,
):
    """Load a faster-whisper model, downloading with progress if needed."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logging.error(
            "faster-whisper package not installed. "
            "Install with: pip install faster-whisper>=1.1.0"
        )
        sys.exit(1)

    if not is_model_cached(model_size):
        _download_model_with_progress(model_size, quiet)

    _configure_cuda_paths()

    logging.info(f"Loading Whisper model ({model_size})...")
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
    logging.info("Whisper model loaded.")
    return model


def _download_model_with_progress(model_size: str, quiet: bool) -> None:
    """Download the model with optional desktop notification progress."""
    from faster_whisper.utils import _MODELS
    import huggingface_hub
    from tqdm import tqdm

    repo_id = _MODELS.get(model_size, model_size)
    allow_patterns = [
        "config.json",
        "preprocessor_config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]

    notify_fn = notify if not quiet else lambda msg: None
    last_milestone = [0]

    class ProgressTqdm(tqdm):
        def update(self, n=1):
            super().update(n)
            if not self.total or "Fetching" in (self.desc or ""):
                return
            percent = int(self.n / self.total * 100)
            milestone = percent // 10 * 10
            if milestone > last_milestone[0]:
                last_milestone[0] = milestone
                notify_fn(f"Downloading {model_size}: {milestone}%")

    logging.info(f"Downloading model {model_size} ({repo_id})...")
    notify_fn(f"Downloading {model_size}...")

    huggingface_hub.snapshot_download(
        repo_id,
        allow_patterns=allow_patterns,
        tqdm_class=ProgressTqdm,
    )

    logging.info("Download complete.")
    notify_fn(f"Download complete - loading {model_size}")


def _configure_cuda_paths() -> None:
    """Preload pip-installed NVIDIA libraries so CTranslate2 can find them."""
    try:
        import nvidia.cublas
        import nvidia.cudnn

        cudnn_dir = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib")
        cublas_dir = os.path.join(os.path.dirname(nvidia.cublas.__file__), "lib")

        for lib_dir, pattern in [
            (cublas_dir, "libcublas.so"),
            (cudnn_dir, "libcudnn_ops.so"),
            (cudnn_dir, "libcudnn_cnn.so"),
            (cudnn_dir, "libcudnn.so"),
        ]:
            if not os.path.isdir(lib_dir):
                logging.debug(f"CUDA library directory missing: {lib_dir}")
                continue
            _preload_first_matching_lib(lib_dir, pattern)
    except (ImportError, OSError) as e:
        logging.debug(f"CUDA library preload skipped: {e}")


def _preload_first_matching_lib(lib_dir: str, pattern: str) -> None:
    """Preload the first library in lib_dir matching the given base name."""
    prefix = pattern.replace(".so", "")
    for filename in sorted(os.listdir(lib_dir)):
        if filename.startswith(prefix) and ".so" in filename:
            ctypes.CDLL(os.path.join(lib_dir, filename), mode=ctypes.RTLD_GLOBAL)
            return
    logging.debug(f"No CUDA library matching {pattern} found in {lib_dir}")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add shared model selection arguments to an argument parser."""
    parser.add_argument(
        "--model",
        default=WHISPER_MODEL_SIZE,
        help=f"Whisper model size (default: {WHISPER_MODEL_SIZE})",
    )
    parser.add_argument(
        "--device",
        default=WHISPER_DEVICE,
        help=f"Compute device: cuda or cpu (default: {WHISPER_DEVICE})",
    )
    parser.add_argument(
        "--compute-type",
        default=WHISPER_COMPUTE_TYPE,
        help=f"Model precision: float16, int8, etc. (default: {WHISPER_COMPUTE_TYPE})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress download progress notifications",
    )
