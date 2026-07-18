#!/bin/sh

# Replace only interactive SSH login shells with the persistent tmux client.
# Remote commands, file transfers, and forwarding-only connections bypass this.
case "$-" in
    *i*)
        if [ -n "${SSH_TTY:-}" ] && [ -z "${TMUX:-}" ] && command -v tmux >/dev/null 2>&1; then
            # Print the welcome inside a newly created session, after tmux has
            # taken over the terminal. Reattaching leaves the session untouched.
            exec tmux new-session -A -s musubi \
                /bin/sh -c 'cat /etc/motd; exec /bin/bash -l'
        fi
        ;;
esac
