"""Punctuation and case restoration for live dictation.

Wraps the vendored recasepunc model to provide a simple restore(text) interface.
Falls back gracefully when the model or dependencies are missing.
"""

import logging
import os
from typing import Optional


class PunctuationRestorer:
    """Restores punctuation and casing using a recasepunc BERT model."""

    def __init__(self, model_path: str) -> None:
        from dictate.vendor.recasepunc import CasePuncPredictor

        checkpoint = os.path.join(model_path, "checkpoint")
        self._predictor = CasePuncPredictor(checkpoint, device="cuda")

    def restore(self, text: str) -> str:
        """Apply punctuation and recasing to raw lowercase text."""
        if not text.strip():
            return text
        tokens = self._predictor.tokenize(text)
        result = ""
        for token, case_label, punc_label in self._predictor.predict(tokens):
            mapped = self._predictor.map_case_label(token, case_label)
            mapped = self._predictor.map_punc_label(mapped, punc_label)
            if mapped.startswith("##"):
                result += mapped[2:]
            elif result:
                result += " " + mapped
            else:
                result = mapped
        return result


def try_load_punctuation(model_dir: str) -> Optional[PunctuationRestorer]:
    """Load the punctuation model, returning None if unavailable.

    Fails gracefully when the model directory or torch/transformers are missing,
    so live dictation works without punctuation support.
    """
    if not os.path.isdir(model_dir):
        logging.info(
            f"Punctuation model not found at {model_dir} - "
            "live dictation will run without punctuation. "
            "Download with: scripts/download-model.sh vosk-recasepunc-en-0.22"
        )
        return None

    try:
        restorer = PunctuationRestorer(model_dir)
        logging.info("Punctuation model loaded.")
        return restorer
    except ImportError as e:
        logging.info(
            f"Punctuation dependencies missing ({e}) - "
            "install with: pip install -e '.[punctuation]'"
        )
        return None
    except Exception as e:
        logging.warning(f"Failed to load punctuation model: {e}")
        return None
