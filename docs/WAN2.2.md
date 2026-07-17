# WAN 2.2 workflows

The scripts expect the following writable directories:

```text
/musubi/
    dataset/dataset.toml
    models/
    output/
```

Scripts skip model files that already exist as regular files.

## Download models

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

## Prepare caches

Cache latents for the selected workflow, then cache the shared text-encoder
outputs:

```shell
prepare-wan2.2-i2v-latents.sh
# or
prepare-wan2.2-t2v-latents.sh

prepare-wan2.2-text-encoder.sh
```

## Train

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

## Credits

- Initially based on the [WAN 2.2 LoRA training workflow](https://civitai.com/articles/17740) by [AI_Characters](https://civitai.com/user/AI_Characters).
- Parameters guided by [My Personally Training Experience of WAN: Starting with Data](https://civitai.com/articles/16936/my-personally-training-experience-of-wanstarting-with-data) by [_GhostInShell_](https://civitai.com/user/_GhostInShell_).
