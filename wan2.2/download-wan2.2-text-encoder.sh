#!/bin/bash
set -euo pipefail

# Download Wan 2.1/2.2 text encoder (shared across flavors)
source "$MUSUBI_SCRIPTS_DIR/scripts/env.sh"

SUBDIR="text_encoders"

FILE_NAME="models_t5_umt5-xxl-enc-bf16.pth"

python "$MUSUBI_SCRIPTS_DIR/scripts/download_models.py" \
  --hf Wan-AI/Wan2.1-I2V-14B-720P \
  --file "$FILE_NAME" \
  --base-dir "$BASE_DIR/models" \
  --output-dir "$SUBDIR"
