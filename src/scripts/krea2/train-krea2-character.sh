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
HF_UPLOAD_SCRIPT="$SCRIPT_DIR/huggingface_checkpoint_upload.py"

print_preset_usage() {
    printf 'Launcher options: --preset default|baseline|quality|10gb [--hf-repo OWNER/REPO] [--hf-path PATH]\n' >&2
}

PRESET="default"
HF_REPO=""
HF_PATH=""
HF_REPO_SET=false
HF_PATH_SET=false
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
        --hf-repo)
            if [[ $# -lt 2 ]]; then
                echo "--hf-repo requires an OWNER/REPO value." >&2
                print_preset_usage
                exit 64
            fi
            HF_REPO="$2"
            HF_REPO_SET=true
            shift 2
            ;;
        --hf-repo=*)
            HF_REPO="${1#*=}"
            HF_REPO_SET=true
            shift
            ;;
        --hf-path)
            if [[ $# -lt 2 ]]; then
                echo "--hf-path requires a repository path." >&2
                print_preset_usage
                exit 64
            fi
            HF_PATH="$2"
            HF_PATH_SET=true
            shift 2
            ;;
        --hf-path=*)
            HF_PATH="${1#*=}"
            HF_PATH_SET=true
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

if [[ "$HF_REPO_SET" == true && -z "$HF_REPO" ]]; then
    echo "--hf-repo requires a non-empty OWNER/REPO value." >&2
    exit 64
fi
if [[ "$HF_PATH_SET" == true && -z "$HF_PATH" ]]; then
    echo "--hf-path requires a non-empty repository path." >&2
    exit 64
fi
if [[ "$HF_PATH_SET" == true && "$HF_REPO_SET" != true ]]; then
    echo "--hf-path requires --hf-repo." >&2
    exit 64
fi

OUTPUT_NAME_EXPLICIT=false
RAW_HF_OPTION=""
for argument in "${FORWARDED_ARGS[@]}"; do
    if [[ "$argument" == "--output_name" || "$argument" == --output_name=* ]]; then
        OUTPUT_NAME_EXPLICIT=true
    fi
    if [[ "$argument" == "--huggingface_repo_id" || \
          "$argument" == --huggingface_repo_id=* || \
          "$argument" == "--huggingface_repo_type" || \
          "$argument" == --huggingface_repo_type=* || \
          "$argument" == "--huggingface_path_in_repo" || \
          "$argument" == --huggingface_path_in_repo=* || \
          "$argument" == "--huggingface_token" || \
          "$argument" == --huggingface_token=* || \
          "$argument" == "--huggingface_repo_visibility" || \
          "$argument" == --huggingface_repo_visibility=* || \
          "$argument" == "--save_state_to_huggingface" || \
          "$argument" == "--async_upload" ]]; then
        RAW_HF_OPTION="${argument%%=*}"
    fi
done

if [[ -n "$HF_REPO" && -n "$RAW_HF_OPTION" ]]; then
    echo "--hf-repo cannot be combined with upstream Hugging Face option: $RAW_HF_OPTION" >&2
    echo "Use the convenience options or the raw upstream options, not both." >&2
    exit 64
fi

case "$PRESET" in
    default)
        TRAIN_CONFIG="$WORKFLOW_DIR/train.toml"
        ;;
    baseline|quality|10gb)
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
require_effective_file "$TRIGGER_WORDS_SCRIPT" "trigger-word helper" "Use an image containing the bundled trigger-word helper."
if [[ -n "$HF_REPO" ]]; then
    require_effective_file "$HF_UPLOAD_SCRIPT" "Hugging Face checkpoint upload helper" \
        "Use an image containing the bundled Krea2 Hugging Face helper."
fi

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

# Resolve the same config-plus-CLI precedence as Musubi before checking paths.
# This keeps preflight compatible with overrides for models, data, prompts,
# output directories, planning values, and even --config_file itself.
effective_paths_file="$(mktemp)"
trap 'rm -f -- "$effective_paths_file"' EXIT
PYTHONPATH="$SCRIPTS_DIR${PYTHONPATH:+:$PYTHONPATH}" \
python - "$TRAIN_CONFIG" "${FORWARDED_ARGS[@]}" >"$effective_paths_file" <<'PY'
import contextlib
import os
import re
import sys

from trigger_words import TriggerWordsError, inspect_dataset


default_config = sys.argv[1]
forwarded_args = sys.argv[2:]
canonical_attention_exclusions = [
    r".*\.mlp\..*",
    "first",
    r"last\.linear",
    r"tmlp\..*",
    r"txtmlp\..*",
    r"tproj\.1",
    r"txtfusion\..*",
]


def positive_int(value):
    if isinstance(value, bool):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def format_number(value):
    if value is None:
        return ""
    if isinstance(value, float):
        value = f"{value:g}"
        return re.sub(r"e([+-])0+(\d+)", r"e\1\2", value)
    return str(value)


def classify_targets(network_args):
    if not network_args:
        return "all-linear"
    if isinstance(network_args, str):
        network_args = [network_args]
    if len(network_args) != 1 or "=" not in network_args[0]:
        return "custom"

    key, value = network_args[0].split("=", 1)
    if key.strip() != "exclude_patterns":
        return "custom"
    value = value.strip()
    if not value.startswith("[") or not value.endswith("]"):
        return "custom"
    patterns = []
    for item in value[1:-1].split(","):
        item = item.strip()
        if len(item) < 2 or item[0] not in {"'", '"'} or item[-1] != item[0]:
            return "custom"
        patterns.append(item[1:-1])
    return (
        "attention-only"
        if sorted(patterns) == sorted(canonical_attention_exclusions)
        else "custom"
    )

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
if not isinstance(args.dataset_config, str) or not args.dataset_config:
    print(
        "No --dataset_config path is configured after applying command-line overrides.",
        file=sys.stderr,
    )
    print(
        "Run init-krea2-character.sh or provide an existing dataset config.",
        file=sys.stderr,
    )
    raise SystemExit(2)
if not os.path.isfile(args.dataset_config):
    print(
        f"Missing required file for --dataset_config: {args.dataset_config}",
        file=sys.stderr,
    )
    print(
        "Run init-krea2-character.sh or provide an existing dataset config.",
        file=sys.stderr,
    )
    raise SystemExit(2)
try:
    inspection = inspect_dataset(args.dataset_config)
except (OSError, UnicodeError, TriggerWordsError) as error:
    print(error, file=sys.stderr)
    raise SystemExit(2)

max_train_steps = positive_int(args.max_train_steps)
save_every_n_steps = positive_int(args.save_every_n_steps)
sample_every_n_steps = positive_int(args.sample_every_n_steps)
state_window = positive_int(args.save_last_n_steps_state)
gradient_accumulation = positive_int(args.gradient_accumulation_steps)
per_device_batch = positive_int(inspection["per_device_batch_size"])
effective_batch = (
    per_device_batch * gradient_accumulation
    if per_device_batch is not None and gradient_accumulation is not None
    else None
)

periodic_candidates = None
final_checkpoint = ""
unique_candidate_states = None
if max_train_steps is not None:
    if save_every_n_steps is None:
        periodic_candidates = 0
        final_checkpoint = f"additional state at step {max_train_steps}"
        unique_candidate_states = 1
    else:
        periodic_candidates = max_train_steps // save_every_n_steps
        if max_train_steps % save_every_n_steps == 0:
            final_checkpoint = f"duplicates step {max_train_steps}"
            unique_candidate_states = periodic_candidates
        else:
            final_checkpoint = f"additional state at step {max_train_steps}"
            unique_candidate_states = periodic_candidates + 1

estimated_passes = None
primary_image_count = positive_int(inspection["primary_image_count"])
if (
    inspection["estimate_authoritative"]
    and max_train_steps is not None
    and effective_batch is not None
    and primary_image_count is not None
):
    estimated_passes = (
        max_train_steps * effective_batch / primary_image_count
    )

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
    max_train_steps,
    save_every_n_steps,
    sample_every_n_steps,
    state_window,
    gradient_accumulation,
    args.network_dim,
    args.network_alpha,
    args.learning_rate,
    classify_targets(args.network_args),
    inspection["layout"],
    inspection["primary_image_count"],
    inspection["additional_dataset_count"],
    per_device_batch,
    "true" if inspection["estimate_authoritative"] else "false",
    inspection["estimate_unavailable_reason"],
    None if estimated_passes is None else f"{estimated_passes:.1f}",
    periodic_candidates,
    final_checkpoint,
    unique_candidate_states,
    effective_batch,
    getattr(args, "huggingface_repo_id", None),
    getattr(args, "huggingface_repo_type", None),
    getattr(args, "huggingface_path_in_repo", None),
    "true" if getattr(args, "huggingface_token", None) else "false",
    getattr(args, "huggingface_repo_visibility", None),
    "true" if getattr(args, "save_state_to_huggingface", False) else "false",
    "true" if getattr(args, "async_upload", False) else "false",
):
    print(format_number(value))
PY

mapfile -t effective_paths <"$effective_paths_file"
rm -f -- "$effective_paths_file"
trap - EXIT

if [[ ${#effective_paths[@]} -ne 36 ]]; then
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
MAX_TRAIN_STEPS="${effective_paths[9]}"
SAVE_EVERY_N_STEPS="${effective_paths[10]}"
SAMPLE_EVERY_N_STEPS="${effective_paths[11]}"
STATE_WINDOW="${effective_paths[12]}"
GRADIENT_ACCUMULATION="${effective_paths[13]}"
NETWORK_DIM="${effective_paths[14]}"
NETWORK_ALPHA="${effective_paths[15]}"
LEARNING_RATE="${effective_paths[16]}"
TARGET_MODULES="${effective_paths[17]}"
DATASET_LAYOUT="${effective_paths[18]}"
PRIMARY_IMAGE_COUNT="${effective_paths[19]}"
ADDITIONAL_DATASET_COUNT="${effective_paths[20]}"
PER_DEVICE_BATCH="${effective_paths[21]}"
ESTIMATE_AUTHORITATIVE="${effective_paths[22]}"
ESTIMATE_UNAVAILABLE_REASON="${effective_paths[23]}"
ESTIMATED_PASSES="${effective_paths[24]}"
PERIODIC_CANDIDATES="${effective_paths[25]}"
FINAL_CHECKPOINT="${effective_paths[26]}"
UNIQUE_CANDIDATE_STATES="${effective_paths[27]}"
EFFECTIVE_BATCH="${effective_paths[28]}"
CONFIG_HF_REPO="${effective_paths[29]}"
CONFIG_HF_REPO_TYPE="${effective_paths[30]}"
CONFIG_HF_PATH="${effective_paths[31]}"
CONFIG_HF_TOKEN_SET="${effective_paths[32]}"
CONFIG_HF_VISIBILITY="${effective_paths[33]}"
CONFIG_HF_STATE_UPLOAD="${effective_paths[34]}"
CONFIG_HF_ASYNC_UPLOAD="${effective_paths[35]}"

if [[ -n "$HF_REPO" ]] && \
    { [[ -n "$CONFIG_HF_REPO" ]] || \
      [[ -n "$CONFIG_HF_REPO_TYPE" ]] || \
      [[ -n "$CONFIG_HF_PATH" ]] || \
      [[ "$CONFIG_HF_TOKEN_SET" == true ]] || \
      [[ -n "$CONFIG_HF_VISIBILITY" ]] || \
      [[ "$CONFIG_HF_STATE_UPLOAD" == true ]] || \
      [[ "$CONFIG_HF_ASYNC_UPLOAD" == true ]]; }; then
    echo "--hf-repo cannot be combined with Hugging Face options in the effective training config." >&2
    echo "Use the convenience options or the raw upstream options, not both." >&2
    exit 64
fi

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
EFFECTIVE_OUTPUT_NAME="$OUTPUT_NAME"
if [[ "$OUTPUT_NAME_EXPLICIT" != true ]]; then
    require_effective_file "$TRIGGER_FILE" "Krea2 trigger metadata" \
        "Run init-krea2-character.sh --trigger k2v9 or pass --output_name explicitly."
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
        echo "Run init-krea2-character.sh --trigger k2v9 or pass --output_name explicitly." >&2
        exit 2
    fi
    EFFECTIVE_OUTPUT_NAME="$GENERATED_OUTPUT_NAME"
    AUTO_OUTPUT_NAME_ARGS=(--output_name "$GENERATED_OUTPUT_NAME")
fi

HF_UPLOAD_ARGS=()
HF_STARTED_AT=""
if [[ -n "$HF_REPO" ]]; then
    HF_STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
    if [[ -z "$HF_PATH" ]]; then
        HF_PATH="krea2/$EFFECTIVE_OUTPUT_NAME/$HF_STARTED_AT"
    fi
    HF_UPLOAD_ARGS=(
        --huggingface_repo_id "$HF_REPO"
        --huggingface_repo_type model
        --huggingface_path_in_repo "$HF_PATH"
    )
fi

print_plan_field() {
    printf '%-32s %s\n' "$1" "$2"
}

echo
echo "Krea2 training search horizon"
echo
print_plan_field "Preset:" "$PRESET"
print_plan_field "Target modules:" "$TARGET_MODULES"
print_plan_field "Rank / alpha:" "${NETWORK_DIM:-unavailable} / ${NETWORK_ALPHA:-unavailable}"
print_plan_field "Learning rate:" "${LEARNING_RATE:-unavailable}"
echo

if [[ "$ESTIMATE_AUTHORITATIVE" == true ]]; then
    print_plan_field "Primary paired images:" "$PRIMARY_IMAGE_COUNT"
else
    print_plan_field "Primary images:" "${PRIMARY_IMAGE_COUNT:-unavailable}"
fi
if [[ "$ADDITIONAL_DATASET_COUNT" != 0 ]]; then
    print_plan_field "Additional datasets:" "$ADDITIONAL_DATASET_COUNT"
fi
print_plan_field "Per-device batch size:" "${PER_DEVICE_BATCH:-unavailable}"
print_plan_field "Gradient accumulation:" "${GRADIENT_ACCUMULATION:-unavailable}"
print_plan_field "Effective single-GPU batch:" "${EFFECTIVE_BATCH:-unavailable}"
echo

print_plan_field "Maximum optimizer steps:" "${MAX_TRAIN_STEPS:-unavailable}"
if [[ -n "$ESTIMATED_PASSES" ]]; then
    print_plan_field "Estimated maximum passes:" "$ESTIMATED_PASSES"
else
    print_plan_field "Estimated maximum passes:" "unavailable ($ESTIMATE_UNAVAILABLE_REASON)"
fi
print_plan_field "Checkpoint interval:" "${SAVE_EVERY_N_STEPS:-disabled}"
print_plan_field "Periodic checkpoint candidates:" "${PERIODIC_CANDIDATES:-0}"
print_plan_field "Final checkpoint:" "${FINAL_CHECKPOINT:-unavailable}"
print_plan_field "Unique candidate states:" "${UNIQUE_CANDIDATE_STATES:-unavailable}"
print_plan_field "Sample interval:" "${SAMPLE_EVERY_N_STEPS:-disabled}"
if [[ -n "$STATE_WINDOW" ]]; then
    print_plan_field "Resumable-state window:" "$STATE_WINDOW steps"
else
    print_plan_field "Resumable-state window:" "not configured"
fi
if [[ -n "$HF_REPO" ]]; then
    echo
    print_plan_field "Hugging Face repository:" "$HF_REPO"
    print_plan_field "Hugging Face path:" "$HF_PATH"
    print_plan_field "Hugging Face artifacts:" "LoRA checkpoints only (synchronous)"
fi

if [[ "$ESTIMATE_AUTHORITATIVE" == true ]] && \
    (( PRIMARY_IMAGE_COUNT < 20 || PRIMARY_IMAGE_COUNT > 40 )); then
    echo
    echo "Warning: the bundled character presets are intended for roughly 20-40 curated character images." >&2
    echo "Review the reported horizon and use an explicit --max_train_steps override when appropriate." >&2
fi

echo
echo "This maximum is a checkpoint-search horizon. The final checkpoint"
echo "is not expected to be the best checkpoint."

echo "LoRA output name: $EFFECTIVE_OUTPUT_NAME"

if [[ -n "$HF_REPO" ]]; then
    python "$HF_UPLOAD_SCRIPT" \
        --repo "$HF_REPO" \
        --path "$HF_PATH" \
        --preset "$PRESET" \
        --output-name "$EFFECTIVE_OUTPUT_NAME" \
        --started-at "$HF_STARTED_AT"
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
    "${AUTO_OUTPUT_NAME_ARGS[@]}" \
    "${HF_UPLOAD_ARGS[@]}"
