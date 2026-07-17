# Krea2 character LoRA workflow

The default Krea2 helpers provide an opinionated character-likeness LoRA
workflow for an RTX 5090 with 32 GB VRAM. It trains on Krea-2-Raw and generates
fixed-seed validation previews with Krea-2-Turbo, following Musubi Tuner's
recommended RAW-training/Turbo-inference workflow. Experimental presets are
also included for attempting training with 10 GB VRAM. Krea2 support requires
Musubi Tuner v0.3.4 or newer.

The Krea model weights use the
[Krea 2 Community License](https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf).
Review that license before downloading or using the models.

## Download models

Download the Raw and Turbo DiTs, Qwen-Image VAE, and Qwen3-VL text encoder:

```shell
download-krea2.sh
```

The download is approximately 62 GB. Files already present as regular files
are skipped, and Hugging Face downloads are cached beneath `models/hf-cache`.
The resulting model layout is:

```text
models/
    hf-cache/
    krea2/raw.safetensors
    krea2/turbo.safetensors
    vae/qwen_image_vae.safetensors
    text_encoders/qwen3vl_4b_bf16.safetensors
```

Use `download-krea2.sh --dry-run` to inspect all four downloads without
fetching the model weights.

## Initialize the editable workflow

Create the character dataset directories and editable configuration files:

```shell
init-krea2-character.sh
```

Initialization is safe to rerun: existing files are reported and preserved.
The generated layout stays inside the already-mounted `dataset` and `output`
directories:

```text
dataset/krea2/
    dataset.toml
    train.toml
    dataset-10gb-smoke.toml
    train-10gb-smoke.toml
    dataset-10gb.toml
    train-10gb.toml
    samples.txt
    images/
    cache/
    cache-10gb-smoke/
    cache-10gb/
output/krea2-character/
    logs/
    10gb-smoke/logs/
    10gb/logs/
```

Add 20-30 curated images to `dataset/krea2/images`, each beside a UTF-8 `.txt`
caption with the same basename:

```text
images/001.png
images/001.txt
images/002.jpg
images/002.txt
```

Choose a unique trigger plus generic class, such as `k2v9 person`, and use it
consistently in every training caption. Edit `dataset/krea2/samples.txt` to use
the same trigger. Caption visible variable attributes such as clothing,
hairstyle, pose, lighting, framing, and background so they do not silently
bind to the identity.

## Experimental 10 GB presets

The low-VRAM presets are intentionally aggressive and are not a guarantee that
Krea2 will fit every 10 GB card or host. They need substantial system RAM for
CPU block swapping; 64 GB is preferable, while 32 GB plus a large swap file may
work very slowly.

Initialize or rerun initialization first so the additional non-destructive
templates are present:

```shell
init-krea2-character.sh
```

Start with the 512-pixel, rank-8, 30-step smoke test:

```shell
prepare-krea2-character.sh --preset 10gb-smoke
train-krea2-character.sh --preset 10gb-smoke
```

If it completes, try the potentially useful 640-pixel, rank-16, 800-step run:

```shell
prepare-krea2-character.sh --preset 10gb
train-krea2-character.sh --preset 10gb
```

Both presets use BF16, scaled FP8 base weights, gradient checkpointing, SDPA,
26 of 28 main blocks swapped to CPU, H2D-only swapping with a one-block ring,
batch size 1, AdamW8bit, and attention-only LoRA targets. Text-encoder caching
runs on CPU. Training-time sampling and Turbo weight loading are disabled
because Turbo previews cannot be combined with block swapping.

The presets share `dataset/krea2/images` but use separate resolution-specific
caches. Re-run preparation for each preset after changing images, captions, or
the configured resolution.
Outputs and TensorBoard logs are written under
`output/krea2-character/10gb-smoke` and `output/krea2-character/10gb`.

If the smoke test still runs out of VRAM, close other GPU applications and edit
`dataset-10gb-smoke.toml` to try 448 or 384 pixels, then reduce rank and alpha
in `train-10gb-smoke.toml` from 8 to 4. Rebuild the smoke cache with
`prepare-krea2-character.sh --preset 10gb-smoke` before retrying training.
Expect block swapping to make training considerably slower than the default
32 GB workflow.

## Prepare caches and train

Rebuild both caches whenever an image or caption changes:

```shell
prepare-krea2-character.sh
```

Preparation validates every image dataset declared in `dataset.toml`, then
removes only generated Krea2 cache files (`*_kr2.safetensors` and
`*_kr2_te.safetensors`) from each configured cache directory before rebuilding
them. Other files are preserved, while deleted or renamed images cannot remain
silently present through stale caches.

Then start training:

```shell
train-krea2-character.sh
```

Additional arguments other than the launcher's `--preset` option are forwarded
to Musubi and override matching values in the selected training TOML. For
example, resume the default workflow from a saved state with:

```shell
train-krea2-character.sh --resume /musubi/output/krea2-character/krea2-character-lora-state
```

The default configuration uses 1024-pixel buckets, BF16, scaled FP8 base
weights, gradient checkpointing, SDPA, Musubi's resolution-aware
`krea2_shift`, AdamW8bit at `1e-4`, and rank/alpha 32. It trains for 1,200
steps and writes checkpoints, resumable state, Turbo previews, and TensorBoard
logs beneath `output/krea2-character`. Checkpoints and previews are produced
every 200 steps. Block swapping is intentionally disabled because Musubi does
not allow it together with training-time Turbo previews.

Start TensorBoard in another shell or tmux pane with:

```shell
tensorboard --logdir /musubi/output/krea2-character/logs
```
