#!/bin/bash
set -euo pipefail

 # Source shared environment (prefers MUSUBI_HOME if set)
source "$MUSUBI_SCRIPTS_DIR/scripts/env.sh"

SUBDIR="diffusion_models"

# Invoke shared component download scripts (expected to exist in image)
"$MUSUBI_SCRIPTS_DIR/wan2.2/download-wan2.2-text-encoder.sh"
"$MUSUBI_SCRIPTS_DIR/wan2.2/download-wan2.2-vae.sh"

python "$MUSUBI_SCRIPTS_DIR/scripts/download_models.py" \
	--hf Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
	--file split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
	--base-dir "$BASE_DIR/models" \
	--output-dir "$SUBDIR"

python "$MUSUBI_SCRIPTS_DIR/scripts/download_models.py" \
	--hf Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
	--file split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
	--base-dir "$BASE_DIR/models" \
	--output-dir "$SUBDIR"

echo "All model components downloaded to $BASE_DIR/models"
