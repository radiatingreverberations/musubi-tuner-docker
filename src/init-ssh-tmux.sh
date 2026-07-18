#!/bin/sh

# Replace only interactive SSH login shells with the persistent tmux client.
# Remote commands, file transfers, and forwarding-only connections bypass this.
case "$-" in
    *i*)
        if [ -n "${SSH_TTY:-}" ] && [ -z "${TMUX:-}" ] && command -v tmux >/dev/null 2>&1; then
            exec tmux new-session -A -s musubi bash -l
        fi
        ;;
esac
