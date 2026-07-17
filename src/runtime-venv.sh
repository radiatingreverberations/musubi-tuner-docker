#!/bin/sh

VIRTUAL_ENV="${OFFLOADR_VENV:-/opt/venv}"
export VIRTUAL_ENV

if [ ! -f "${VIRTUAL_ENV}/bin/activate" ]; then
    echo "Expected baked virtual environment at ${VIRTUAL_ENV}, but it was not found." >&2
    return 1 2>/dev/null || exit 1
fi

case ":${PATH}:" in
    *":${VIRTUAL_ENV}/bin:"*) ;;
    *) export PATH="${VIRTUAL_ENV}/bin:${PATH}" ;;
esac

. "${VIRTUAL_ENV}/bin/activate"
