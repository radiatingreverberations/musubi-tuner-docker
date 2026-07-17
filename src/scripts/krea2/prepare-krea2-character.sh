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
LATENT_SCRIPT="$MUSUBI_HOME/src/musubi_tuner/krea2_cache_latents.py"
TEXT_ENCODER_SCRIPT="$MUSUBI_HOME/src/musubi_tuner/krea2_cache_text_encoder_outputs.py"

print_usage() {
    printf 'Usage: %s [--preset default|32gb-quality|32gb-attention|10gb] [--trigger "token class"] [--skip-trigger-check]\n' "$(basename "$0")"
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
    default|32gb-quality|32gb-attention)
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
python - "$DATASET_CONFIG" "$SAMPLES_FILE" "$TRIGGER_OVERRIDE" "$SKIP_TRIGGER_CHECK" <<'PY'
import os
from pathlib import Path
import re
import sys

import toml

from musubi_tuner.dataset.media_utils import glob_images


config = toml.load(sys.argv[1])
samples_path = Path(sys.argv[2])
trigger_override = sys.argv[3]
skip_trigger_check = sys.argv[4] == "true"
general = config.get("general", {})
datasets = config.get("datasets", [])
errors = []
trigger = None
active_prompts = []
validated_captions = 0
primary_uses_jsonl = False

if not skip_trigger_check:
    samples_text = samples_path.read_text(encoding="utf-8-sig")
    header = re.search(r"(?mi)^#\s*trigger:\s*([^\r\n]*?)\s*$", samples_text)
    legacy = re.search(
        r'(?mi)^#\s*Replace every occurrence of\s+"([^"\r\n]+)"', samples_text
    )
    if trigger_override:
        trigger = trigger_override
    elif header and header.group(1).strip():
        trigger = header.group(1).strip()
    elif legacy and legacy.group(1).strip():
        trigger = legacy.group(1).strip()
    else:
        errors.append(
            f"No trigger metadata was found in {samples_path}. Add "
            "'# trigger: token class' or pass --trigger."
        )

    allow_next_prompt_without_trigger = False
    for line_number, line in enumerate(samples_text.splitlines(), start=1):
        prompt = line.strip()
        if not prompt:
            continue
        if prompt.lower() == "# trigger-check: allow-next":
            allow_next_prompt_without_trigger = True
            continue
        if prompt.startswith("#"):
            allow_next_prompt_without_trigger = False
            continue
        active_prompts.append(
            (line_number, prompt, allow_next_prompt_without_trigger)
        )
        allow_next_prompt_without_trigger = False

    if not active_prompts:
        errors.append(f"No active sample prompts were found in {samples_path}")
    elif trigger:
        for line_number, prompt, trigger_optional in active_prompts:
            if not trigger_optional and trigger not in prompt:
                errors.append(
                    f'Trigger "{trigger}" is missing from active sample prompt '
                    f"{samples_path}:{line_number}"
                )

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
                elif not skip_trigger_check and index == 1 and trigger:
                    try:
                        caption = Path(caption_path).read_text(encoding="utf-8-sig")
                    except UnicodeDecodeError:
                        errors.append(f"Caption is not valid UTF-8: {caption_path}")
                        continue
                    if trigger not in caption:
                        errors.append(
                            f'Trigger "{trigger}" is missing from primary training caption: '
                            f"{caption_path}"
                        )
                    else:
                        validated_captions += 1
        elif not skip_trigger_check and index == 1:
            errors.append(
                "Dataset 1 has no caption_extension, so its trigger cannot be validated."
            )
    elif image_jsonl_file:
        image_jsonl_file = os.path.abspath(image_jsonl_file)
        if not os.path.isfile(image_jsonl_file):
            errors.append(f"Dataset {index} image JSONL file does not exist: {image_jsonl_file}")
        elif not skip_trigger_check and index == 1:
            primary_uses_jsonl = True
    else:
        errors.append(f"Dataset {index} has neither image_directory nor image_jsonl_file configured")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(2)

if skip_trigger_check:
    print("Trigger validation skipped by request.")
elif primary_uses_jsonl:
    print(
        f'Validated trigger rules for {len(active_prompts)} active sample prompts; '
        "primary JSONL captions were not inspected."
    )
else:
    print(
        f'Validated trigger "{trigger}" in {validated_captions} primary captions '
        f"and trigger rules for {len(active_prompts)} active sample prompts."
    )
PY

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
