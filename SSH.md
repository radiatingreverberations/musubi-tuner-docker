# Musubi Tuner SSH Image

The SSH image is intended for cloud GPU instances where a provider console is
inconvenient or unavailable. It starts OpenSSH on port 2222 and gives the SSH
user root-equivalent file access inside the container so mounted training data
does not acquire a second container UID.

## Start the image

```shell
docker run --rm --gpus=all -p 2222:2222 \
  -v "$PWD/models:/musubi/models" \
  -v "$PWD/dataset:/musubi/dataset" \
  -v "$PWD/output:/musubi/output" \
  ghcr.io/radiatingreverberations/musubi-tuner-ssh:latest
```

At startup the image prints the username, detected public address, SSH host-key
fingerprint, and a connection command. Hosting providers may map port 2222 to a
different external port, so prefer the address and port shown in their
dashboard.

## Authentication

SSH key authentication is recommended:

```shell
docker run --rm --gpus=all -p 2222:2222 \
  -e SSH_USER=trainer \
  -e 'SSH_KEY=ssh-ed25519 AAAA...your-public-key...' \
  ghcr.io/radiatingreverberations/musubi-tuner-ssh:latest
```

Connect with the matching private key:

```shell
ssh -p 2222 -i ~/.ssh/id_ed25519 trainer@HOST
```

Password authentication is also supported:

```shell
docker run --rm --gpus=all -p 2222:2222 \
  -e SSH_USER=trainer \
  -e SSH_PASSWORD='a-long-unique-password' \
  ghcr.io/radiatingreverberations/musubi-tuner-ssh:latest
```

If no credentials are supplied, the image creates a random 32-character
username with an empty password. Treat that generated username as a secret and
use this mode only when the provider securely exposes the container log and SSH
port. The selected username is also written to `/run/ssh-user`.

When `SSH_KEY` is set, password authentication is disabled. The supported
runtime variables are:

| Variable | Purpose |
| --- | --- |
| `SSH_USER` | Optional fixed lowercase username; generated when omitted |
| `SSH_KEY` | OpenSSH public key placed in `authorized_keys` |
| `SSH_PASSWORD` | Optional password when key authentication is not used |
| `SSH_HOST_ED25519_KEY_B64` | Optional base64-encoded Ed25519 private host key |

## Host-key verification

Published SSH images intentionally contain a pinned Ed25519 private host key.
This gives transient GPU deployments, such as newly created RunPod instances,
the same SHA256 fingerprint across container replacements and image updates. It
avoids a new host identity and `REMOTE HOST IDENTIFICATION HAS CHANGED` warning
each time a short-lived instance is created.

This is a deliberate continuity tradeoff: because the image is public, anyone
who can pull it can extract the shared private host key. The printed fingerprint
helps detect accidental endpoint or image changes, but it does not provide
exclusive server authentication against an attacker who possesses the image.
Do not reuse this host key for unrelated infrastructure.

The CI-pinned host key used by published images has this fingerprint:

```text
SHA256:/5bQkL5SFEVPqTwgxNG0ElFxTxm8J7FGnQNnDndGDFM
```

Locally built images and deployments using `SSH_HOST_ED25519_KEY_B64` at
runtime may have a different fingerprint. Any other change should correspond
to a documented key rotation.

The release pipeline supplies the pinned key through the
`SSH_HOST_ED25519_KEY_B64` BuildKit secret. The same variable can be set at
runtime to override the image key with a deployment-specific identity. Reusing
one runtime key across deployments preserves a stable fingerprint without
sharing it through the public image. The value must be the base64 encoding of
an unencrypted OpenSSH Ed25519 private key.

## Forward a local service

Local TCP forwarding is permitted only to loopback addresses inside the
container. TensorBoard is included for viewing training metrics. Create another
tmux window with `Ctrl-b`, then `c`, and start it on loopback port 6006:

```shell
tensorboard --logdir /musubi/output --host 127.0.0.1 --port 6006
```

From your computer, reconnect with a local forward, substituting the external
SSH host and port assigned by the deployment:

```shell
ssh -p EXTERNAL_SSH_PORT -L 6006:127.0.0.1:6006 trainer@HOST
```

Then open `http://127.0.0.1:6006` locally. Remote forwarding, SSH-agent
forwarding, Unix-socket forwarding, tunneling, and X11 forwarding are disabled.

## Session environment

SSH shells start in `/musubi` with the baked Python environment at `/opt/venv`
activated. WAN helper scripts are available directly through `PATH`.

Interactive SSH logins automatically start or attach to the tmux session named
`musubi`, so jobs continue when the connection closes. Press `Ctrl-b`, then `d`
to detach without stopping them; the next interactive login reattaches. Press
`Ctrl-b`, then `c` to create another tmux window. When the session is first
created, the welcome and quickstart text is printed inside tmux so it remains
visible after tmux takes over the terminal.

Remote commands and SFTP, scp, rsync, and forwarding-only connections do not
enter tmux. A direct remote command therefore behaves normally:

```shell
ssh -p 2222 trainer@HOST nvidia-smi
```
