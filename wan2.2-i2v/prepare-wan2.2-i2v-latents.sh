#!/bin/bash
set -euo pipefail

 # Source shared environment (prefers MUSUBI_HOME if set)
source "$MUSUBI_SCRIPTS_DIR/scripts/env.sh"

python "$MUSUBI_HOME/src/musubi_tuner/wan_cache_latents.py" \
    --dataset_config "$BASE_DIR/dataset/dataset.toml" \
    --vae "$BASE_DIR/models/vae/wan_2.1_vae.safetensors" \
    --i2v