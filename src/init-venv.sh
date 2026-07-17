#!/bin/sh

export MUSUBI_HOME="${MUSUBI_HOME:-/musubi}"
export MUSUBI_SCRIPTS_DIR="${MUSUBI_SCRIPTS_DIR:-/opt/musubi-scripts}"
export OFFLOADR_VENV="${OFFLOADR_VENV:-/opt/venv}"

expected_venv="${OFFLOADR_VENV}"
if [ "${VIRTUAL_ENV:-}" != "${expected_venv}" ] && [ -r /usr/local/lib/musubi/runtime-venv.sh ]; then
    . /usr/local/lib/musubi/runtime-venv.sh
fi
