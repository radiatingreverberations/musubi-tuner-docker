# Musubi Tuner Docker Images

Prebuilt NVIDIA images for preparing datasets and training networks with
[Musubi Tuner](https://github.com/kohya-ss/musubi-tuner). The images include
turnkey Krea2 character-LoRA and WAN 2.2 i2v/t2v helper scripts, an interactive
tmux environment, and an optional SSH server for cloud GPU instances.

## Images and tags

| Image | Purpose |
| --- | --- |
| `ghcr.io/radiatingreverberations/musubi-tuner` | Interactive local or provider-console sessions |
| `ghcr.io/radiatingreverberations/musubi-tuner-ssh` | Remote shell access through OpenSSH on port 2222 |

Both images publish these tags:

| Tag | Musubi Tuner source |
| --- | --- |
| `latest` | Latest tagged Musubi Tuner release |
| `main` | Latest commit on the upstream `main` branch |
| `vX.Y.Z` | A specific upstream release |

The NVIDIA runtime is based on Python 3.12, PyTorch 2.11, and CUDA 13.0.3.
A compatible NVIDIA driver and the NVIDIA Container Toolkit are required.

## Interactive image

The standard image requires a TTY and starts or attaches to a tmux session:

```shell
docker run --rm --gpus=all -it \
  -v "$PWD/models:/musubi/models" \
  -v "$PWD/dataset:/musubi/dataset" \
  -v "$PWD/output:/musubi/output" \
  ghcr.io/radiatingreverberations/musubi-tuner:latest
```

The Musubi source lives in `/musubi`, its Python environment is `/opt/venv`,
and the helper scripts are installed in `/opt/musubi-scripts` and linked into
`PATH`.

For a remote cloud session, use the SSH image described in [SSH.md](SSH.md).

## Training workflows

- [Krea2 character LoRA](docs/KREA2.md) provides a turnkey 32 GB VRAM
  RAW-training and Turbo-preview workflow, higher-quality 32 GB comparison
  presets, and experimental 10 GB presets.
- [WAN 2.2](docs/WAN2.2.md) covers the existing i2v and t2v download, cache,
  and two-phase training helpers.

## Building locally

Build both images with Docker Buildx:

```shell
docker buildx bake
```

Build only one target or select an upstream ref:

```shell
docker buildx bake base
MUSUBI_VERSION=v0.3.4 IMAGE_LABEL=latest docker buildx bake ssh
```

The main Bake inputs are `DOCKER_REGISTRY_URL`, `NVIDIA_BASE_IMAGE`,
`MUSUBI_VERSION`, `REFRESH_MUSUBI`, and `IMAGE_LABEL`.
