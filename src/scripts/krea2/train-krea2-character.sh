#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
SCRIPTS_DIR="${MUSUBI_SCRIPTS_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPTS_DIR/env.sh"

WORKFLOW_DIR="$BASE_DIR/dataset/krea2"
TRAIN_SCRIPT="$MUSUBI_HOME/src/musubi_tuner/krea2_train_network.py"
TRIGGER_FILE="$WORKFLOW_DIR/samples.txt"
TRIGGER_WORDS_SCRIPT="$SCRIPTS_DIR/trigger_words.py"

print_preset_usage() {
    printf 'Launcher preset: --preset default|quality|attention|10gb\n' >&2
}

PRESET="default"
FORWARDED_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            if [[ $# -lt 2 ]]; then
                echo "--preset requires a value." >&2
                print_preset_usage
                exit 64
            fi
            PRESET="$2"
            shift 2
            ;;
        --preset=*)
            PRESET="${1#*=}"
            shift
            ;;
        --)
            shift
            FORWARDED_ARGS+=("$@")
            break
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
            ;;
    esac
done

OUTPUT_NAME_EXPLICIT=false
for argument in "${FORWARDED_ARGS[@]}"; do
    if [[ "$argument" == "--output_name" || "$argument" == --output_name=* ]]; then
        OUTPUT_NAME_EXPLICIT=true
        break
    fi
done

case "$PRESET" in
    default)
        TRAIN_CONFIG="$WORKFLOW_DIR/train.toml"
        ;;
    quality|attention|10gb)
        TRAIN_CONFIG="$WORKFLOW_DIR/train-$PRESET.toml"
        ;;
    *)
        echo "Unknown Krea2 preset: $PRESET" >&2
        print_preset_usage
        exit 64
        ;;
esac

require_effective_file() {
    local path="$1"
    local option="$2"
    local hint="$3"

    if [[ -z "$path" ]]; then
        echo "No $option path is configured after applying command-line overrides." >&2
        echo "$hint" >&2
        exit 2
    fi

    if [[ ! -f "$path" ]]; then
        echo "Missing required file for $option: $path" >&2
        echo "$hint" >&2
        exit 2
    fi
}

require_effective_file "$TRAIN_SCRIPT" "Krea2 training script" "Use a Musubi Tuner image with Krea2 support (v0.3.4 or newer)."

if ! command -v python >/dev/null 2>&1 || ! command -v accelerate >/dev/null 2>&1; then
    echo "Python or Accelerate is not available on PATH; the image virtual environment is not active." >&2
    exit 2
fi

for argument in "${FORWARDED_ARGS[@]}"; do
    if [[ "$argument" == "-h" || "$argument" == "--help" ]]; then
        exec python "$TRAIN_SCRIPT" "${FORWARDED_ARGS[@]}"
    fi
done

cd "$BASE_DIR"
echo "Training Krea2 preset: $PRESET"

# Resolve the same config-plus-CLI precedence as Musubi before checking paths.
# This keeps preflight compatible with overrides for models, data, prompts,
# output directories, and even --config_file itself.
effective_paths_file="$(mktemp)"
trap 'rm -f -- "$effective_paths_file"' EXIT
python - "$TRAIN_CONFIG" "${FORWARDED_ARGS[@]}" >"$effective_paths_file" <<'PY'
import contextlib
import os
import sys


default_config = sys.argv[1]
forwarded_args = sys.argv[2:]

with contextlib.redirect_stdout(sys.stderr):
    from musubi_tuner.krea2_train_network import krea2_setup_parser
    from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common

parser = krea2_setup_parser(setup_parser_common())
sys.argv = ["krea2_train_network.py", "--config_file", default_config, *forwarded_args]
args = parser.parse_args()

if args.config_file is None:
    print("No training config is selected. Run init-krea2-character.sh or pass --config_file.", file=sys.stderr)
    raise SystemExit(2)

config_path = args.config_file if args.config_file.endswith(".toml") else args.config_file + ".toml"
if not os.path.isfile(config_path):
    print(f"Missing training config: {config_path}", file=sys.stderr)
    print("Run init-krea2-character.sh or pass --config_file with an existing TOML file.", file=sys.stderr)
    raise SystemExit(2)

args = read_config_from_file(args, parser)
for value in (
    args.dataset_config,
    args.dit,
    args.vae,
    args.turbo_dit,
    args.text_encoder,
    args.sample_prompts,
    args.output_dir,
    args.logging_dir,
    args.output_name,
):
    print("" if value is None else value)
PY

mapfile -t effective_paths <"$effective_paths_file"
rm -f -- "$effective_paths_file"
trap - EXIT

if [[ ${#effective_paths[@]} -ne 9 ]]; then
    echo "Unable to resolve the effective Krea2 training configuration." >&2
    exit 2
fi

DATASET_CONFIG="${effective_paths[0]}"
DIT="${effective_paths[1]}"
VAE="${effective_paths[2]}"
TURBO_DIT="${effective_paths[3]}"
TEXT_ENCODER="${effective_paths[4]}"
SAMPLE_PROMPTS="${effective_paths[5]}"
OUTPUT_DIR="${effective_paths[6]}"
LOGGING_DIR="${effective_paths[7]}"
OUTPUT_NAME="${effective_paths[8]}"

require_effective_file "$DATASET_CONFIG" "--dataset_config" "Run init-krea2-character.sh or provide an existing dataset config."
require_effective_file "$DIT" "--dit" "Run download-krea2.sh or override --dit with an existing Krea-2-Raw checkpoint."
require_effective_file "$VAE" "--vae" "Run download-krea2.sh or override --vae with an existing Qwen-Image VAE."

if [[ -n "$SAMPLE_PROMPTS" ]]; then
    require_effective_file "$SAMPLE_PROMPTS" "--sample_prompts" "Provide an existing sample prompt file or disable training-time previews."
    require_effective_file "$TEXT_ENCODER" "--text_encoder" "Run download-krea2.sh or override --text_encoder with an existing Qwen3-VL checkpoint."
    if [[ -n "$TURBO_DIT" ]]; then
        require_effective_file "$TURBO_DIT" "--turbo_dit" "Run download-krea2.sh or override --turbo_dit with an existing Krea-2-Turbo checkpoint."
    fi
fi

AUTO_OUTPUT_NAME_ARGS=()
if [[ "$OUTPUT_NAME_EXPLICIT" != true ]]; then
    require_effective_file "$TRIGGER_FILE" "Krea2 trigger metadata" \
        "Run init-krea2-character.sh --trigger \"token class\" or pass --output_name explicitly."
    require_effective_file "$TRIGGER_WORDS_SCRIPT" "trigger-word helper" \
        "Use an image containing the bundled trigger-word helper."
    if [[ -z "$OUTPUT_NAME" ]]; then
        echo "No --output_name is configured after applying command-line overrides." >&2
        exit 2
    fi

    if ! GENERATED_OUTPUT_NAME="$(
        python "$TRIGGER_WORDS_SCRIPT" output-name \
            --samples "$TRIGGER_FILE" \
            --base-name "$OUTPUT_NAME" \
            --insert-after-prefix krea2
    )"; then
        echo "Run init-krea2-character.sh --trigger \"token class\" or pass --output_name explicitly." >&2
        exit 2
    fi
    AUTO_OUTPUT_NAME_ARGS=(--output_name "$GENERATED_OUTPUT_NAME")
    echo "LoRA output name: $GENERATED_OUTPUT_NAME"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
fi
if [[ -n "$LOGGING_DIR" ]]; then
    mkdir -p "$LOGGING_DIR"
fi

exec accelerate launch \
    --num_cpu_threads_per_process 1 \
    --mixed_precision bf16 \
    "$TRAIN_SCRIPT" \
    --config_file "$TRAIN_CONFIG" \
    "${FORWARDED_ARGS[@]}" \
    "${AUTO_OUTPUT_NAME_ARGS[@]}"
