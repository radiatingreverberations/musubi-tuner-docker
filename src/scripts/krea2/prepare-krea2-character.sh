#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
SCRIPTS_DIR="${MUSUBI_SCRIPTS_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPTS_DIR/env.sh"

WORKFLOW_DIR="$BASE_DIR/dataset/krea2"
VAE="$BASE_DIR/models/vae/qwen_image_vae.safetensors"
TEXT_ENCODER="$BASE_DIR/models/text_encoders/qwen3vl_4b_bf16.safetensors"
SAMPLES_FILE="$WORKFLOW_DIR/samples.txt"
TRIGGER_WORDS_SCRIPT="$SCRIPTS_DIR/trigger_words.py"
LATENT_SCRIPT="$MUSUBI_HOME/src/musubi_tuner/krea2_cache_latents.py"
TEXT_ENCODER_SCRIPT="$MUSUBI_HOME/src/musubi_tuner/krea2_cache_text_encoder_outputs.py"

print_usage() {
    printf 'Usage: %s [--preset default|baseline|quality|10gb] [--trigger unique-token] [--skip-trigger-check]\n' "$(basename "$0")"
}

PRESET="default"
TRIGGER_OVERRIDE=""
TRIGGER_EXPLICIT=false
SKIP_TRIGGER_CHECK=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            if [[ $# -lt 2 ]]; then
                echo "--preset requires a value." >&2
                print_usage >&2
                exit 64
            fi
            PRESET="$2"
            shift 2
            ;;
        --preset=*)
            PRESET="${1#*=}"
            shift
            ;;
        --trigger)
            if [[ $# -lt 2 ]]; then
                echo "--trigger requires a value." >&2
                print_usage >&2
                exit 64
            fi
            TRIGGER_OVERRIDE="$2"
            TRIGGER_EXPLICIT=true
            shift 2
            ;;
        --trigger=*)
            TRIGGER_OVERRIDE="${1#*=}"
            TRIGGER_EXPLICIT=true
            shift
            ;;
        --skip-trigger-check)
            SKIP_TRIGGER_CHECK=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            print_usage >&2
            exit 64
            ;;
    esac
done

if [[ "$TRIGGER_EXPLICIT" == true ]] && \
    [[ -z "${TRIGGER_OVERRIDE//[[:space:]]/}" || "$TRIGGER_OVERRIDE" == *$'\n'* || "$TRIGGER_OVERRIDE" == *$'\r'* ]]; then
    echo "The Krea2 trigger must contain visible text on one line." >&2
    exit 64
fi

TEXT_ENCODER_DEVICE_ARGS=()
case "$PRESET" in
    default|baseline|quality)
        DATASET_CONFIG="$WORKFLOW_DIR/dataset.toml"
        ;;
    10gb)
        DATASET_CONFIG="$WORKFLOW_DIR/dataset-$PRESET.toml"
        TEXT_ENCODER_DEVICE_ARGS=(--device cpu)
        ;;
    *)
        echo "Unknown Krea2 preset: $PRESET" >&2
        print_usage >&2
        exit 64
        ;;
esac

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing required file: $1" >&2
        echo "$2" >&2
        exit 2
    fi
}

require_file "$DATASET_CONFIG" "Run init-krea2-character.sh first."
require_file "$TRIGGER_WORDS_SCRIPT" "Use an image containing the bundled trigger-word helper."
if [[ "$SKIP_TRIGGER_CHECK" != true ]]; then
    require_file "$SAMPLES_FILE" "Run init-krea2-character.sh first."
fi

echo "Preparing Krea2 preset: $PRESET"

if ! command -v python >/dev/null 2>&1; then
    echo "Python is not available on PATH; the image virtual environment is not active." >&2
    exit 2
fi

cd "$BASE_DIR"

# Validate the same image datasets and extensions that Musubi will cache. This
# covers edited image_directory paths and additional datasets such as the
# optional regularization dataset in the scaffold.
TRIGGER_VALIDATION_ARGS=(validate --dataset-config "$DATASET_CONFIG")
if [[ "$SKIP_TRIGGER_CHECK" == true ]]; then
    TRIGGER_VALIDATION_ARGS+=(--skip-trigger-check)
else
    TRIGGER_VALIDATION_ARGS+=(--samples "$SAMPLES_FILE")
    if [[ "$TRIGGER_EXPLICIT" == true ]]; then
        TRIGGER_VALIDATION_ARGS+=(--trigger "$TRIGGER_OVERRIDE")
    fi
fi
python "$TRIGGER_WORDS_SCRIPT" "${TRIGGER_VALIDATION_ARGS[@]}"

require_file "$VAE" "Run download-krea2.sh first."
require_file "$TEXT_ENCODER" "Run download-krea2.sh first."
require_file "$LATENT_SCRIPT" "Use a Musubi Tuner image with Krea2 support (v0.3.4 or newer)."
require_file "$TEXT_ENCODER_SCRIPT" "Use a Musubi Tuner image with Krea2 support (v0.3.4 or newer)."

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
    --batch_size 1 \
    "${TEXT_ENCODER_DEVICE_ARGS[@]}"
