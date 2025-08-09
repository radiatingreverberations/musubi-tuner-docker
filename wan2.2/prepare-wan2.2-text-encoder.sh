#!/bin/bash
set -euo pipefail

 # Source shared environment (prefers MUSUBI_HOME if set)
source "$MUSUBI_SCRIPTS_DIR/scripts/env.sh"

python "$MUSUBI_HOME/src/musubi_tuner/wan_cache_text_encoder_outputs.py" \
    --dataset_config "$BASE_DIR/dataset/dataset.toml" \
    --t5 "$BASE_DIR/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
    --batch_size 16