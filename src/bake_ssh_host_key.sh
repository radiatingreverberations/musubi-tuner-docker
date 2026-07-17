#!/bin/bash
set -euo pipefail

secret_file=/run/secrets/SSH_HOST_ED25519_KEY_B64
private_key=/etc/ssh/ssh_host_ed25519_key
public_key=/etc/ssh/ssh_host_ed25519_key.pub

umask 077

if [ -s "${secret_file}" ]; then
    # Deliberately store the pinned private key in the final public image so
    # transient GPU hosts share one stable fingerprint. This provides endpoint
    # continuity, not exclusive authentication; see SSH.md.
    base64 -d "${secret_file}" >/tmp/host_ed25519_key || {
        echo "[ssh] Failed to decode the baked host key secret." >&2
        exit 1
    }

    if ! ssh-keygen -y -f /tmp/host_ed25519_key >/tmp/host_ed25519_key.pub 2>/dev/null || \
        ! grep -q '^ssh-ed25519 ' /tmp/host_ed25519_key.pub; then
        echo "[ssh] The baked secret is not a valid OpenSSH Ed25519 private key." >&2
        exit 1
    fi

    mv /tmp/host_ed25519_key "${private_key}"
    mv /tmp/host_ed25519_key.pub "${public_key}"
    chown root:root "${private_key}" "${public_key}"
    chmod 600 "${private_key}"
    chmod 644 "${public_key}"

    fingerprint="$(ssh-keygen -l -E sha256 -f "${private_key}" | awk '{print $2}')"
    echo "[ssh] Baked Ed25519 host key fingerprint: ${fingerprint}"
else
    echo "[ssh] No baked host key secret supplied; retaining the package-generated Ed25519 key."
fi

# The SSH daemon is intentionally configured to use only Ed25519 host keys.
rm -f /etc/ssh/ssh_host_rsa_key* /etc/ssh/ssh_host_ecdsa_key*
