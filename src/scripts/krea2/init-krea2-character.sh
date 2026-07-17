#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
SCRIPTS_DIR="${MUSUBI_SCRIPTS_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPTS_DIR/env.sh"

TEMPLATE_DIR="$SCRIPT_DIR/templates"
WORKFLOW_DIR="$BASE_DIR/dataset/krea2"
OUTPUT_DIR="$BASE_DIR/output/krea2-character"

mkdir -p \
    "$WORKFLOW_DIR/images" \
    "$WORKFLOW_DIR/cache" \
    "$WORKFLOW_DIR/cache-10gb" \
    "$OUTPUT_DIR/logs" \
    "$OUTPUT_DIR/32gb-quality/logs" \
    "$OUTPUT_DIR/32gb-attention/logs" \
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

render_template "$TEMPLATE_DIR/dataset.toml" "$WORKFLOW_DIR/dataset.toml"
render_template "$TEMPLATE_DIR/train.toml" "$WORKFLOW_DIR/train.toml"
render_template "$TEMPLATE_DIR/train-32gb-quality.toml" "$WORKFLOW_DIR/train-32gb-quality.toml"
render_template "$TEMPLATE_DIR/train-32gb-attention.toml" "$WORKFLOW_DIR/train-32gb-attention.toml"
render_template "$TEMPLATE_DIR/dataset-10gb.toml" "$WORKFLOW_DIR/dataset-10gb.toml"
render_template "$TEMPLATE_DIR/train-10gb.toml" "$WORKFLOW_DIR/train-10gb.toml"
render_template "$TEMPLATE_DIR/samples.txt" "$WORKFLOW_DIR/samples.txt"

echo
echo "Krea2 character workflow initialized in $WORKFLOW_DIR"
echo "Add paired images and .txt captions to $WORKFLOW_DIR/images, then edit $WORKFLOW_DIR/samples.txt."
echo "32 GB presets: --preset 32gb-quality or --preset 32gb-attention"
echo "Low-VRAM preset: --preset 10gb"
