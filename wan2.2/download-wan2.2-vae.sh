#!/bin/bash
set -euo pipefail

# Download Wan 2.1 VAE (shared across flavors)
source "$MUSUBI_SCRIPTS_DIR/scripts/env.sh"

SUBDIR="vae"
REL_PATH="split_files/vae/wan_2.1_vae.safetensors"
BASE_NAME="wan_2.1_vae.safetensors"

python "$MUSUBI_SCRIPTS_DIR/scripts/download_models.py" \
  --hf Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  --file "$REL_PATH" \
  --base-dir "$BASE_DIR/models" \
  --output-dir "$SUBDIR" \
  --dest-name "$BASE_NAME"
