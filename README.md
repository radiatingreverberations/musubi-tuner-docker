# Musubi Tuner Prebuilt Docker Images

Minimal scripts + container helpers for preparing and training WAN 2.2 i2v LoRA style networks.

## WAN 2.2 i2v Workflow

High-level sequence:

1. Download model components
2. Prepare (pre-cache) latents (VAE) and text encoder outputs
3. Train (low-noise and/or high-noise phases)

### 1. Download

Use the convenience scripts (they will skip existing regular files):

```bash
download-wan2.2-i2v.sh          # downloads text encoder, VAE, both diffusion models

# Or individual pieces:
download-wan2.2-text-encoder.sh
download-wan2.2-vae.sh
```

After completion you should have (flattened layout):

```text
models/
    hf-cache/                               # shared Hugging Face cache
    text_encoders/models_t5_umt5-xxl-enc-bf16.pth
    vae/wan_2.1_vae.safetensors
    diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors
    diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors
```

### 2. Prepare

Cache VAE latents:

```bash
prepare-wan2.2-latents.sh
```

Cache text encoder outputs:

```bash
prepare-wan2.2-text-encoder.sh
```

These wrappers call the underlying Python utilities inside the container / environment.

### 3. Train

Low-noise phase:

```bash
train-wan2.2-i2v-low-noise.sh
```

High-noise phase:

```bash
train-wan2.2-i2v-high-noise.sh
```

Adjust hyperparameters inside the scripts (learning rate, epochs, network_dim, etc.) as needed.

## Credits

* Initially based on [WAN2.2 LoRa training workflow](https://civitai.com/articles/17740) by [AI_Characters](https://civitai.com/user/AI_Characters).
* Parameters guided by [My Personally Training Experience of WAN:Starting with Data](https://civitai.com/articles/16936/my-personally-training-experience-of-wanstarting-with-data) by [_GhostInShell_](https://civitai.com/user/_GhostInShell_)
