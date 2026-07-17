# Musubi Tuner Docker Images

Prebuilt NVIDIA images for preparing datasets and training networks with
[Musubi Tuner](https://github.com/kohya-ss/musubi-tuner). The images include
WAN 2.2 i2v/t2v helper scripts, an interactive tmux environment, and an
optional SSH server for cloud GPU instances.

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

## WAN 2.2 workflows

The scripts expect the following writable directories:

```text
/musubi/
    dataset/dataset.toml
    models/
    output/
```

Scripts skip model files that already exist as regular files.

### Download models

Download the text encoder, VAE, and both diffusion models for the selected
workflow:

```shell
download-wan2.2-i2v.sh
# or
download-wan2.2-t2v.sh
```

Individual shared components can also be downloaded separately:

```shell
download-wan2.2-text-encoder.sh
download-wan2.2-vae.sh
```

The resulting layout is:

```text
models/
    hf-cache/
    text_encoders/models_t5_umt5-xxl-enc-bf16.pth
    vae/wan_2.1_vae.safetensors
    diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors
    diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors
    # or the corresponding wan2.2_t2v_* files
```

### Prepare caches

Cache latents for the selected workflow, then cache the shared text-encoder
outputs:

```shell
prepare-wan2.2-i2v-latents.sh
# or
prepare-wan2.2-t2v-latents.sh

prepare-wan2.2-text-encoder.sh
```

### Train

Run either or both noise phases:

```shell
train-wan2.2-i2v-low-noise.sh
train-wan2.2-i2v-high-noise.sh

# t2v equivalents
train-wan2.2-t2v-low-noise.sh
train-wan2.2-t2v-high-noise.sh
```

The bundled scripts contain example training parameters. Review their learning
rate, epoch, checkpoint, and network settings before starting a long run.

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

## Credits

- Initially based on the [WAN 2.2 LoRA training workflow](https://civitai.com/articles/17740) by [AI_Characters](https://civitai.com/user/AI_Characters).
- Parameters guided by [My Personally Training Experience of WAN: Starting with Data](https://civitai.com/articles/16936/my-personally-training-experience-of-wanstarting-with-data) by [_GhostInShell_](https://civitai.com/user/_GhostInShell_).
