#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
SCRIPTS_DIR="${MUSUBI_SCRIPTS_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPTS_DIR/env.sh"

TEMPLATE_DIR="$SCRIPT_DIR/templates"
TRIGGER_WORDS_SCRIPT="$SCRIPTS_DIR/trigger_words.py"
WORKFLOW_DIR="$BASE_DIR/dataset/krea2"
OUTPUT_DIR="$BASE_DIR/output/krea2-character"
DEFAULT_TRIGGER="k2v9 person"
TRIGGER="$DEFAULT_TRIGGER"
TRIGGER_EXPLICIT=false

print_usage() {
    printf 'Usage: %s [--trigger "token class"]\n' "$(basename "$0")"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trigger)
            if [[ $# -lt 2 ]]; then
                echo "--trigger requires a value." >&2
                print_usage >&2
                exit 64
            fi
            TRIGGER="$2"
            TRIGGER_EXPLICIT=true
            shift 2
            ;;
        --trigger=*)
            TRIGGER="${1#*=}"
            TRIGGER_EXPLICIT=true
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

if [[ -z "${TRIGGER//[[:space:]]/}" || "$TRIGGER" == *$'\n'* || "$TRIGGER" == *$'\r'* ]]; then
    echo "The Krea2 trigger must contain visible text on one line." >&2
    exit 64
fi

mkdir -p \
    "$WORKFLOW_DIR/images" \
    "$WORKFLOW_DIR/cache" \
    "$WORKFLOW_DIR/cache-10gb" \
    "$OUTPUT_DIR/logs" \
    "$OUTPUT_DIR/quality/logs" \
    "$OUTPUT_DIR/attention/logs" \
    "$OUTPUT_DIR/10gb/logs"

escaped_base_dir="${BASE_DIR//\\/\\\\}"
escaped_base_dir="${escaped_base_dir//&/\\&}"
escaped_base_dir="${escaped_base_dir//|/\\|}"

render_template() {
    local source_path="$1"
    local target_path="$2"

    if [[ ! -f "$source_path" ]]; then
        echo "Missing bundled Krea2 template: $source_path" >&2
        exit 2
    fi

    if [[ -e "$target_path" || -L "$target_path" ]]; then
        echo "Preserving existing file: $target_path"
        return
    fi

    local temporary_path
    temporary_path="$(mktemp "${target_path}.tmp.XXXXXX")"
    sed "s|__MUSUBI_HOME__|${escaped_base_dir}|g" "$source_path" >"$temporary_path"
    chmod 0644 "$temporary_path"
    mv "$temporary_path" "$target_path"
    echo "Created: $target_path"
}

render_or_update_samples() {
    local source_path="$1"
    local target_path="$2"

    if [[ ! -f "$source_path" ]]; then
        echo "Missing bundled Krea2 template: $source_path" >&2
        exit 2
    fi

    if [[ -e "$target_path" || -L "$target_path" ]]; then
        if [[ "$TRIGGER_EXPLICIT" != true ]]; then
            echo "Preserving existing file: $target_path"
            return
        fi
        if [[ ! -f "$target_path" ]]; then
            echo "Cannot update Krea2 trigger in non-regular file: $target_path" >&2
            exit 2
        fi
    fi

    if ! command -v python >/dev/null 2>&1; then
        echo "Python is not available on PATH; the image virtual environment is not active." >&2
        exit 2
    fi

    if [[ ! -f "$TRIGGER_WORDS_SCRIPT" ]]; then
        echo "Missing trigger-word helper: $TRIGGER_WORDS_SCRIPT" >&2
        exit 2
    fi

    python "$TRIGGER_WORDS_SCRIPT" set \
        --template "$source_path" \
        --samples "$target_path" \
        --trigger "$TRIGGER" \
        --placeholder "__KREA2_TRIGGER__"
}

render_template "$TEMPLATE_DIR/dataset.toml" "$WORKFLOW_DIR/dataset.toml"
render_template "$TEMPLATE_DIR/train.toml" "$WORKFLOW_DIR/train.toml"
render_template "$TEMPLATE_DIR/train-quality.toml" "$WORKFLOW_DIR/train-quality.toml"
render_template "$TEMPLATE_DIR/train-attention.toml" "$WORKFLOW_DIR/train-attention.toml"
render_template "$TEMPLATE_DIR/dataset-10gb.toml" "$WORKFLOW_DIR/dataset-10gb.toml"
render_template "$TEMPLATE_DIR/train-10gb.toml" "$WORKFLOW_DIR/train-10gb.toml"
render_or_update_samples "$TEMPLATE_DIR/samples.txt" "$WORKFLOW_DIR/samples.txt"

echo
echo "Krea2 character workflow initialized in $WORKFLOW_DIR"
echo "Add paired images and .txt captions to $WORKFLOW_DIR/images."
echo "Set or update the preview trigger with: init-krea2-character.sh --trigger \"token class\""
echo "32 GB presets: --preset quality or --preset attention"
echo "Low-VRAM preset: --preset 10gb"
