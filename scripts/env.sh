#!/bin/bash
# Shared environment bootstrap for musubi tuner scripts.
# Sets BASE_DIR to $MUSUBI_HOME if defined, else repository root (directory containing this file's parent).
set -euo pipefail

# Resolve repo root as the parent directory of this script's directory (scripts/ ..)
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
: "${MUSUBI_HOME:=}"  # ensure variable is defined (may be empty)
if [[ -n "$MUSUBI_HOME" ]]; then
  BASE_DIR="${MUSUBI_HOME}"
else
  BASE_DIR="$REPO_ROOT"
fi
export BASE_DIR

# Create a simple lock so repeated sourcing in one shell doesn't spam output
if [[ -z "${_MUSUBI_ENV_ECHOED:-}" ]]; then
  echo "Using base directory: $BASE_DIR" >&2
  export _MUSUBI_ENV_ECHOED=1
fi
