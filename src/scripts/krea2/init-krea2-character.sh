#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
SCRIPTS_DIR="${MUSUBI_SCRIPTS_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
source "$SCRIPTS_DIR/env.sh"

TEMPLATE_DIR="$SCRIPT_DIR/templates"
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

    python - "$source_path" "$target_path" "$TRIGGER" <<'PY'
import os
from pathlib import Path
import re
import stat
import tempfile
import sys


source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
trigger = sys.argv[3]
placeholder = "__KREA2_TRIGGER__"

if target_path.exists():
    original = target_path.read_text(encoding="utf-8-sig")
    header = re.search(r"(?mi)^#\s*trigger:\s*([^\r\n]*?)\s*$", original)
    legacy = re.search(r'(?mi)^#\s*Replace every occurrence of\s+"([^"\r\n]+)"', original)
    match = header or legacy
    if match is None or not match.group(1).strip():
        print(
            f"Cannot detect the existing trigger in {target_path}. "
            "Add '# trigger: token class' or edit the file manually.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    old_trigger = match.group(1).strip()
    updated = original.replace(old_trigger, trigger)
    if header is None:
        updated = f"# trigger: {trigger}\n" + updated
    mode = stat.S_IMODE(target_path.stat().st_mode)
    action = "Trigger already set in" if updated == original else "Updated trigger in"
else:
    original = source_path.read_text(encoding="utf-8")
    if placeholder not in original:
        print(f"Missing {placeholder} in bundled template: {source_path}", file=sys.stderr)
        raise SystemExit(2)
    updated = original.replace(placeholder, trigger)
    mode = 0o644
    action = "Created"

if not target_path.exists() or updated != original:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", dir=target_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, target_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

print(f"{action}: {target_path}")
PY
}

render_template "$TEMPLATE_DIR/dataset.toml" "$WORKFLOW_DIR/dataset.toml"
render_template "$TEMPLATE_DIR/train.toml" "$WORKFLOW_DIR/train.toml"
render_template "$TEMPLATE_DIR/train-32gb-quality.toml" "$WORKFLOW_DIR/train-32gb-quality.toml"
render_template "$TEMPLATE_DIR/train-32gb-attention.toml" "$WORKFLOW_DIR/train-32gb-attention.toml"
render_template "$TEMPLATE_DIR/dataset-10gb.toml" "$WORKFLOW_DIR/dataset-10gb.toml"
render_template "$TEMPLATE_DIR/train-10gb.toml" "$WORKFLOW_DIR/train-10gb.toml"
render_or_update_samples "$TEMPLATE_DIR/samples.txt" "$WORKFLOW_DIR/samples.txt"

echo
echo "Krea2 character workflow initialized in $WORKFLOW_DIR"
echo "Add paired images and .txt captions to $WORKFLOW_DIR/images."
echo "Set or update the preview trigger with: init-krea2-character.sh --trigger \"token class\""
echo "32 GB presets: --preset 32gb-quality or --preset 32gb-attention"
echo "Low-VRAM preset: --preset 10gb"
