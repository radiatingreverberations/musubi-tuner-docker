#!/bin/bash
set -euo pipefail

source /usr/local/lib/musubi/runtime-venv.sh

SSH_USER="${SSH_USER:-u-$(tr -d '-' </proc/sys/kernel/random/uuid | cut -c1-30)}"

if [[ ! "${SSH_USER}" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]]; then
    echo "SSH_USER must be a valid lowercase Linux username of at most 32 characters." >&2
    exit 1
fi

get_public_ipv4() {
    local url ip
    for url in https://api.ipify.org https://ifconfig.me/ip https://icanhazip.com https://checkip.amazonaws.com; do
        ip="$(curl -4 -fsS --max-time 2 "${url}" 2>/dev/null | tr -d '\r\n' || true)"
        if [[ "${ip}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
            printf '%s\n' "${ip}"
            return 0
        fi
    done
    return 0
}

get_public_ipv6() {
    local url ip
    [ -r /proc/net/if_inet6 ] && [ -s /proc/net/if_inet6 ] || return 0
    for url in https://api.ipify.org https://ifconfig.me https://icanhazip.com; do
        ip="$(curl -6 -fsS --max-time 2 "${url}" 2>/dev/null | tr -d '\r\n' || true)"
        if [[ "${ip}" == *:* ]]; then
            printf '%s\n' "${ip}"
            return 0
        fi
    done
    return 0
}

public_ipv4="$(get_public_ipv4)"
public_ipv6="$(get_public_ipv6)"
host_hint="${public_ipv4:-${public_ipv6:-<host-ip>}}"

printf '%s\n' "${SSH_USER}" >/run/ssh-user

private_host_key=/etc/ssh/ssh_host_ed25519_key
public_host_key=/etc/ssh/ssh_host_ed25519_key.pub

# A runtime key explicitly overrides a key baked into the image.
if [ -n "${SSH_HOST_ED25519_KEY_B64:-}" ]; then
    umask 077
    printf '%s' "${SSH_HOST_ED25519_KEY_B64}" | base64 -d >/tmp/runtime_host_ed25519_key || {
        echo "Failed to decode SSH_HOST_ED25519_KEY_B64." >&2
        exit 1
    }
    if ! ssh-keygen -y -f /tmp/runtime_host_ed25519_key >/tmp/runtime_host_ed25519_key.pub 2>/dev/null || \
        ! grep -q '^ssh-ed25519 ' /tmp/runtime_host_ed25519_key.pub; then
        echo "SSH_HOST_ED25519_KEY_B64 is not a valid OpenSSH Ed25519 private key." >&2
        exit 1
    fi
    mv /tmp/runtime_host_ed25519_key "${private_host_key}"
    mv /tmp/runtime_host_ed25519_key.pub "${public_host_key}"
fi

if [ ! -f "${private_host_key}" ]; then
    ssh-keygen -q -t ed25519 -N '' -f "${private_host_key}"
elif [ ! -f "${public_host_key}" ]; then
    ssh-keygen -y -f "${private_host_key}" >"${public_host_key}"
fi

chown root:root "${private_host_key}" "${public_host_key}"
chmod 600 "${private_host_key}"
chmod 644 "${public_host_key}"

host_fingerprint="$(ssh-keygen -l -E sha256 -f "${private_host_key}" | awk '{print $2}')"

if ! id -u "${SSH_USER}" >/dev/null 2>&1; then
    useradd -M -d /musubi -s /bin/bash -u 0 -o "${SSH_USER}"
else
    usermod -o -u 0 -d /musubi -s /bin/bash "${SSH_USER}"
fi

if [ "$(id -u "${SSH_USER}")" -ne 0 ]; then
    echo "Failed to map SSH_USER=${SSH_USER} to UID 0." >&2
    exit 1
fi
passwd -d "${SSH_USER}" >/dev/null

if [ -n "${SSH_PASSWORD:-}" ]; then
    printf '%s:%s\n' "${SSH_USER}" "${SSH_PASSWORD}" | chpasswd
fi

if [ -n "${SSH_KEY:-}" ]; then
    install -d -m 0700 /musubi/.ssh
    printf '%s\n' "${SSH_KEY}" >/musubi/.ssh/authorized_keys
    chmod 0600 /musubi/.ssh/authorized_keys
    chown -R "${SSH_USER}:${SSH_USER}" /musubi/.ssh
    if ! ssh-keygen -l -f /musubi/.ssh/authorized_keys >/dev/null 2>&1; then
        echo "SSH_KEY does not contain a valid OpenSSH public key." >&2
        exit 1
    fi
fi

if command -v tput >/dev/null 2>&1 && [ -t 1 ]; then
    bold="$(tput bold)"
    reset="$(tput sgr0)"
else
    bold=""
    reset=""
fi

if [ -n "${SSH_KEY:-}" ]; then
    auth_method="SSH key"
elif [ -n "${SSH_PASSWORD:-}" ]; then
    auth_method="password"
else
    auth_method="empty password"
fi

echo
echo "================================================================================"
echo " ${bold}Musubi Tuner + SSH Access${reset}"
echo "================================================================================"
echo " User:        ${bold}${SSH_USER}${reset}"
echo " SSH Port:    ${bold}2222${reset}"
echo " Host key ID: ${bold}${host_fingerprint}${reset}"
echo " Auth method: ${bold}${auth_method}${reset}"
[ -n "${public_ipv4}" ] && echo " Public IPv4: ${bold}${public_ipv4}${reset}"
[ -n "${public_ipv6}" ] && echo " Public IPv6: ${bold}${public_ipv6}${reset}"
echo
echo " Connect with:"
echo "   ssh -p 2222 ${SSH_USER}@${host_hint}"
echo
echo " Provider-assigned addresses and ports may differ; check the provider dashboard."
echo "================================================================================"
echo

cat >/etc/motd <<'EOF'
You are connected to Musubi Tuner.

Working directory: /musubi
Python environment: /opt/venv

To forward a loopback service such as TensorBoard, reconnect with:
  ssh -p 2222 -L <local-port>:127.0.0.1:<service-port> <user>@<host>
EOF

cat >/etc/ssh/sshd_config <<EOF
Port 2222
HostKey ${private_host_key}
HostKeyAlgorithms ssh-ed25519
UsePAM no
PasswordAuthentication $([ -z "${SSH_KEY:-}" ] && echo yes || echo no)
KbdInteractiveAuthentication no
PermitEmptyPasswords $([ -z "${SSH_PASSWORD:-}" ] && [ -z "${SSH_KEY:-}" ] && echo yes || echo no)
PermitRootLogin yes
PubkeyAuthentication $([ -n "${SSH_KEY:-}" ] && echo yes || echo no)
AuthorizedKeysFile $([ -n "${SSH_KEY:-}" ] && echo .ssh/authorized_keys || echo none)
StrictModes yes

AllowUsers ${SSH_USER}
SetEnv MUSUBI_HOME=/musubi MUSUBI_SCRIPTS_DIR=/opt/musubi-scripts OFFLOADR_VENV=/opt/venv VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:/opt/musubi-scripts:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PermitTTY yes
AllowTcpForwarding local
PermitOpen 127.0.0.1:* [::1]:*
PermitListen none
AllowStreamLocalForwarding no
AllowAgentForwarding no
GatewayPorts no
X11Forwarding no
PermitTunnel no
PermitUserEnvironment no

ClientAliveInterval 30
ClientAliveCountMax 3
TCPKeepAlive yes

PrintMotd yes
Subsystem sftp internal-sftp
EOF

mkdir -p /run/sshd
chmod 0755 /run/sshd
sshd -t -f /etc/ssh/sshd_config

exec /usr/sbin/sshd -e -D -f /etc/ssh/sshd_config
