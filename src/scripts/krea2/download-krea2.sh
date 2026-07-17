#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
SCRIPTS_DIR="${MUSUBI_SCRIPTS_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPTS_DIR/env.sh"

if ! command -v python >/dev/null 2>&1; then
    echo "Python is not available on PATH; the image virtual environment is not active." >&2
    exit 2
fi

if [[ ! -f "$SCRIPTS_DIR/download_models.py" ]]; then
    echo "Missing model downloader: $SCRIPTS_DIR/download_models.py" >&2
    exit 2
fi

dry_run=false
download_args=()
if [[ $# -gt 0 ]]; then
    if [[ $# -eq 1 && "$1" == "--dry-run" ]]; then
        dry_run=true
        download_args+=(--dry-run)
    else
        echo "Usage: $(basename "$0") [--dry-run]" >&2
        exit 64
    fi
fi

python "$SCRIPTS_DIR/download_models.py" \
    --hf krea/Krea-2-Raw \
    --file raw.safetensors \
    --base-dir "$BASE_DIR/models" \
    --output-dir krea2 \
    "${download_args[@]}"

python "$SCRIPTS_DIR/download_models.py" \
    --hf krea/Krea-2-Turbo \
    --file turbo.safetensors \
    --base-dir "$BASE_DIR/models" \
    --output-dir krea2 \
    "${download_args[@]}"

python "$SCRIPTS_DIR/download_models.py" \
    --hf Comfy-Org/Qwen-Image_ComfyUI \
    --file split_files/vae/qwen_image_vae.safetensors \
    --base-dir "$BASE_DIR/models" \
    --output-dir vae \
    "${download_args[@]}"

python "$SCRIPTS_DIR/download_models.py" \
    --hf Comfy-Org/Qwen3-VL \
    --file text_encoders/qwen3vl_4b_bf16.safetensors \
    --base-dir "$BASE_DIR/models" \
    --output-dir text_encoders \
    "${download_args[@]}"

if [[ "$dry_run" == true ]]; then
    echo "Krea2 model download dry run complete."
else
    echo "All Krea2 model components are available under $BASE_DIR/models"
fi
