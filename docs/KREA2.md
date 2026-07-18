# Krea2 character LoRA workflow

The default Krea2 helpers provide an opinionated character-likeness LoRA
workflow for a GPU with 32 GB VRAM. It trains on Krea-2-Raw and generates
fixed-seed validation previews with Krea-2-Turbo, following Musubi Tuner's
recommended RAW-training/Turbo-inference workflow. Experimental presets are
included for higher-capacity 32 GB comparisons and for attempting training
with 10 GB VRAM. Krea2 support requires Musubi Tuner v0.3.4 or newer.

The Krea model weights use the
[Krea 2 Community License](https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf).
Review that license before downloading or using the models.

For guidance on curating images, writing captions, comparing previews, and
choosing a checkpoint, see
[Krea2 character LoRA best practices](KREA2-BEST-PRACTICES.md).

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
init-krea2-character.sh --trigger "k2v9 person"
```

Choose a unique trigger plus generic class and pass them together, for example
`mira7 person`. Without `--trigger`, initialization uses `k2v9 person` for a
new `samples.txt`. Initialization is safe to rerun: existing files are reported
and preserved unless `--trigger` is explicitly supplied, in which case the
detected trigger in `samples.txt` is updated atomically. The generated layout
stays inside the already-mounted `dataset` and `output` directories:

```text
dataset/krea2/
    dataset.toml
    train.toml
    train-quality.toml
    train-attention.toml
    dataset-10gb.toml
    train-10gb.toml
    samples.txt
    images/
    cache/
    cache-10gb/
output/krea2-character/
    logs/
    quality/logs/
    attention/logs/
    10gb/logs/
```

If the workflow was initialized before the 32 GB comparison presets were
added, the existing `dataset.toml` is deliberately preserved. Set
`bucket_no_upscale = true` in that file to match the current 1024-pixel
template, then rerun `prepare-krea2-character.sh` to rebuild the shared cache.

Add 20-30 curated images to `dataset/krea2/images`, each beside a UTF-8 `.txt`
caption with the same basename:

```text
images/001.png
images/001.txt
images/002.jpg
images/002.txt
```

Use the same trigger and class in every primary training caption. Caption
visible variable attributes such as clothing, hairstyle, pose, lighting,
framing, and background so they do not silently bind to the identity.

During preparation, the `# trigger:` header in `samples.txt` is treated as the
source of truth. Every active sample prompt and every caption in the first
configured image dataset must contain that exact phrase. A sample immediately
following `# trigger-check: allow-next` is exempt, which preserves the bundled
trigger-free leakage-control preview. Optional regularization datasets are
still checked for missing files, but their captions are deliberately excluded
from trigger validation. Legacy sample templates are recognized through their
original `Replace every occurrence of ...` comment.

The bundled preview prompts are medium-neutral: they vary framing, viewpoint,
pose, clothing, and setting without requiring photography, realism, or a
specific illustration style. They can therefore be used for photographic,
anime, and other character LoRAs, or replaced with prompts tailored to the
dataset's intended style.

The training launcher also uses the first token in the trigger phrase as a
filesystem-safe LoRA name component. For example, `mira7 person` produces
`krea2-mira7-character-lora`, with the selected preset suffix retained. Pass an
explicit Musubi output name to opt out or choose a completely custom name:

```shell
train-krea2-character.sh --output_name my-custom-lora
```

For an unusual workflow, override only the validation value or disable the
trigger check explicitly:

```shell
prepare-krea2-character.sh --trigger "mira7 person"
prepare-krea2-character.sh --skip-trigger-check
```

Thirty curated 1024x1024 images are already a strong dataset shape for these
presets. The shared dataset config keeps native 1024 buckets and disables
upscaling; spend additional GPU time on LoRA capacity and checkpoint comparison
rather than enlarging 1024-pixel sources to 1280.

## 32 GB comparison presets

The default configuration is the reference run. Two additional presets provide
a controlled higher-capacity comparison while keeping the data, 1024-pixel
resolution, seed, precision, and preview cadence constant:

| Run | Launcher preset | Targets | Rank/alpha | Learning rate | Steps |
| --- | --- | --- | --- | --- | --- |
| A - reference | `default` | All Linear layers | 32/32 | `1e-4` | 1,800 |
| B - higher capacity | `quality` | All Linear layers | 64/64 | `7e-5` | 2,700 |
| C - longer attention-only | `attention` | Attention projections | 64/64 | `5e-5` | 3,600 |

All three use `dataset/krea2/dataset.toml` and its shared cache. Prepare once
after adding or changing images or captions:

```shell
prepare-krea2-character.sh
```

Then run whichever comparisons you want:

```shell
train-krea2-character.sh
train-krea2-character.sh --preset quality
train-krea2-character.sh --preset attention
```

With 30 images and batch size 1, the runs are approximately 60, 90, and 120
dataset passes. Each saves checkpoints, resumable state, and fixed-seed Turbo
previews every 200 steps. The default intentionally continues beyond step 1,200
so that 1,400, 1,600, and 1,800 can be compared; this does not imply that the
last checkpoint is best. Likeness can peak before clothing, pose, or background
bias becomes excessive. The preset outputs are isolated beneath
`output/krea2-character`, `output/krea2-character/quality`, and
`output/krea2-character/attention`.

## Experimental 10 GB preset

The low-VRAM preset has completed on a 10 GB RTX 3080, but other cards and hosts
may behave differently. It needs substantial system RAM for CPU block swapping;
64 GB is preferable, while 32 GB plus a large swap file may work very slowly.

Initialize or rerun initialization first so the additional non-destructive
templates are present:

```shell
init-krea2-character.sh
```

Prepare and train the 640-pixel, rank-16, 800-step run:

```shell
prepare-krea2-character.sh --preset 10gb
train-krea2-character.sh --preset 10gb
```

The preset uses BF16, scaled FP8 base weights, gradient checkpointing, SDPA,
26 of 28 main blocks swapped to CPU, H2D-only swapping with a one-block ring,
batch size 1, AdamW8bit, and attention-only LoRA targets. Text-encoder caching
runs on CPU. Training-time sampling and Turbo weight loading are disabled
because Turbo previews cannot be combined with block swapping.

It shares `dataset/krea2/images` with the 32 GB workflows but uses its own
resolution-specific cache. Re-run preparation after changing images, captions,
or the configured resolution. Outputs and TensorBoard logs are written under
`output/krea2-character/10gb`.

If it runs out of VRAM, close other GPU applications and edit
`dataset-10gb.toml` to try 512, 448, or 384 pixels, then reduce rank and alpha
in `train-10gb.toml`. Rebuild the cache with
`prepare-krea2-character.sh --preset 10gb` before retrying training. Expect
block swapping to make training considerably slower than the default 32 GB
workflow.

## Default preparation and training

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
train-krea2-character.sh --resume /musubi/output/krea2-character/krea2-mira7-character-lora-state
```

The default configuration uses 1024-pixel buckets without upscaling, BF16,
scaled FP8 base weights, gradient checkpointing, SDPA, Musubi's resolution-aware
`krea2_shift`, AdamW8bit at `1e-4`, and rank/alpha 32. It trains for 1,800
steps and writes checkpoints, resumable state, Turbo previews, and TensorBoard
logs beneath `output/krea2-character`. Checkpoints and previews are produced
every 200 steps. Block swapping is intentionally disabled because Musubi does
not allow it together with training-time Turbo previews.

Initialization preserves an existing `train.toml`. To extend a workflow created
with the previous 1,200-step template, either set `max_train_steps = 1800` and
`save_last_n_steps = 2000` in that file, or override the duration for a new run:

```shell
train-krea2-character.sh --max_train_steps 1800
```

To continue an existing 1,200-step run, also resume its saved state rather than
starting over:

```shell
train-krea2-character.sh --max_train_steps 1800 \
  --resume /musubi/output/krea2-character/krea2-mira7-character-lora-state
```

Start TensorBoard in another shell or tmux pane with:

```shell
tensorboard --logdir /musubi/output/krea2-character
```
