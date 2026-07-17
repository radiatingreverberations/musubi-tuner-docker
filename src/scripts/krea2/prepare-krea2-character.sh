#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
SCRIPTS_DIR="${MUSUBI_SCRIPTS_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPTS_DIR/env.sh"

WORKFLOW_DIR="$BASE_DIR/dataset/krea2"
DATASET_CONFIG="$WORKFLOW_DIR/dataset.toml"
VAE="$BASE_DIR/models/vae/qwen_image_vae.safetensors"
TEXT_ENCODER="$BASE_DIR/models/text_encoders/qwen3vl_4b_bf16.safetensors"
LATENT_SCRIPT="$MUSUBI_HOME/src/musubi_tuner/krea2_cache_latents.py"
TEXT_ENCODER_SCRIPT="$MUSUBI_HOME/src/musubi_tuner/krea2_cache_text_encoder_outputs.py"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing required file: $1" >&2
        echo "$2" >&2
        exit 2
    fi
}

require_file "$DATASET_CONFIG" "Run init-krea2-character.sh first."
require_file "$VAE" "Run download-krea2.sh first."
require_file "$TEXT_ENCODER" "Run download-krea2.sh first."
require_file "$LATENT_SCRIPT" "Use a Musubi Tuner image with Krea2 support (v0.3.4 or newer)."
require_file "$TEXT_ENCODER_SCRIPT" "Use a Musubi Tuner image with Krea2 support (v0.3.4 or newer)."

if ! command -v python >/dev/null 2>&1; then
    echo "Python is not available on PATH; the image virtual environment is not active." >&2
    exit 2
fi

cd "$BASE_DIR"

# Validate the same image datasets and extensions that Musubi will cache. This
# covers edited image_directory paths and additional datasets such as the
# optional regularization dataset in the scaffold.
python - "$DATASET_CONFIG" <<'PY'
import os
import sys

import toml

from musubi_tuner.dataset.media_utils import glob_images


config = toml.load(sys.argv[1])
general = config.get("general", {})
datasets = config.get("datasets", [])
errors = []

if not datasets:
    errors.append(f"No datasets are configured in {sys.argv[1]}")

for index, dataset in enumerate(datasets, start=1):
    image_directory = dataset.get("image_directory")
    image_jsonl_file = dataset.get("image_jsonl_file")

    if image_directory:
        image_directory = os.path.abspath(image_directory)
        if not os.path.isdir(image_directory):
            errors.append(f"Dataset {index} image directory does not exist: {image_directory}")
            continue

        images = glob_images(image_directory)
        if not images:
            errors.append(f"Dataset {index} has no supported training images: {image_directory}")
            continue

        caption_extension = dataset.get("caption_extension", general.get("caption_extension"))
        if caption_extension:
            for image_path in images:
                caption_path = os.path.splitext(image_path)[0] + caption_extension
                if not os.path.isfile(caption_path):
                    errors.append(f"Missing caption for dataset {index} training image: {caption_path}")
    elif image_jsonl_file:
        image_jsonl_file = os.path.abspath(image_jsonl_file)
        if not os.path.isfile(image_jsonl_file):
            errors.append(f"Dataset {index} image JSONL file does not exist: {image_jsonl_file}")
    else:
        errors.append(f"Dataset {index} has neither image_directory nor image_jsonl_file configured")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(2)
PY

# Musubi trains from every matching cache file, not by reconciling caches with
# the current image directory. Remove only generated Krea2 cache files so
# deleted, renamed, or rebucketed images cannot survive a requested rebuild.
cache_directories_file="$(mktemp)"
trap 'rm -f -- "$cache_directories_file"' EXIT
python - "$DATASET_CONFIG" >"$cache_directories_file" <<'PY'
import os
import sys

import toml


config = toml.load(sys.argv[1])
seen = set()
for dataset in config.get("datasets", []):
    cache_directory = dataset.get("cache_directory") or dataset.get("image_directory")
    if not cache_directory:
        continue
    cache_directory = os.path.abspath(cache_directory)
    if cache_directory not in seen:
        print(cache_directory)
        seen.add(cache_directory)
PY

mapfile -t cache_directories <"$cache_directories_file"
rm -f -- "$cache_directories_file"
trap - EXIT

if [[ ${#cache_directories[@]} -eq 0 ]]; then
    echo "No cache_directory or image_directory was found in $DATASET_CONFIG" >&2
    exit 2
fi

for cache_directory in "${cache_directories[@]}"; do
    if [[ -z "$cache_directory" || "$cache_directory" == "/" ]]; then
        echo "Refusing to clean unsafe cache directory: $cache_directory" >&2
        exit 2
    fi

    mkdir -p "$cache_directory"
    find "$cache_directory" -maxdepth 1 -type f \
        \( -name '*_kr2.safetensors' -o -name '*_kr2_te.safetensors' \) \
        -print -delete
done

mkdir -p "$BASE_DIR/output/krea2-character/logs"

python "$LATENT_SCRIPT" \
    --dataset_config "$DATASET_CONFIG" \
    --vae "$VAE"

python "$TEXT_ENCODER_SCRIPT" \
    --dataset_config "$DATASET_CONFIG" \
    --text_encoder "$TEXT_ENCODER" \
    --batch_size 1
