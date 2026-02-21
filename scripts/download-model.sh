#!/usr/bin/env bash
# Download and extract the Vosk speech recognition model.
# Usage: scripts/download-model.sh [model-name]
#
# Default model: vosk-model-en-us-0.22 (1.8GB, 5.69% WER)
# Models list: https://alphacephei.com/vosk/models

set -euo pipefail

MODEL_NAME="${1:-vosk-model-en-us-0.22}"
MODEL_DIR="$HOME/.local/share/voice-dictation/models"
MODEL_URL="https://alphacephei.com/vosk/models/${MODEL_NAME}.zip"

if [ -d "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "Model already exists at $MODEL_DIR/$MODEL_NAME"
    exit 0
fi

echo "Downloading $MODEL_NAME..."
echo "URL: $MODEL_URL"
echo "Destination: $MODEL_DIR/$MODEL_NAME"
echo ""

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

wget -q --show-progress "$MODEL_URL" -O "${MODEL_NAME}.zip"
echo "Extracting..."
unzip -q "${MODEL_NAME}.zip"
rm "${MODEL_NAME}.zip"

echo ""
echo "Model installed at $MODEL_DIR/$MODEL_NAME"
echo "You can now run: dictate-live"
