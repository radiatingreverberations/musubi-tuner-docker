#!/bin/bash
set -euo pipefail

if ! [ -t 0 ] || ! [ -t 1 ]; then
    echo "This container requires an interactive TTY. Run it with -it." >&2
    exit 1
fi

source /usr/local/lib/musubi/runtime-venv.sh

exec tmux new-session -A -s musubi bash -l
