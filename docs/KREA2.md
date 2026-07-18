# Krea2 character LoRA workflow

This image includes a turnkey character-LoRA workflow for Krea2. Training uses
Krea-2-Raw; previews and final inference use Krea-2-Turbo. The default workflow
targets a GPU with 32 GB VRAM, and a slower 10 GB preset is also included.

The commands to run are collected first. Configuration rationale, generated
paths, cache behavior, and recovery details are in the [appendix](#appendix).
For dataset and checkpoint-selection guidance, see
[Krea2 character LoRA best practices](KREA2-BEST-PRACTICES.md).

## Commands at a glance

Default 32 GB workflow:

```shell
download-krea2.sh
init-krea2-character.sh --trigger k2v9
# Add paired images and captions under /musubi/dataset/krea2/images
prepare-krea2-character.sh
train-krea2-character.sh
```

Alternative 32 GB training runs reuse the same prepared cache:

```shell
train-krea2-character.sh --preset quality
train-krea2-character.sh --preset attention
```

For a 10 GB card, use its separate preparation command:

```shell
prepare-krea2-character.sh --preset 10gb
train-krea2-character.sh --preset 10gb
```

Inspect every run with:

```shell
tensorboard --logdir /musubi/output/krea2-character
```

## Step-by-step: default 32 GB run

### 1. Download the models

```shell
download-krea2.sh
```

This downloads approximately 62 GB. The Krea weights use the
[Krea 2 Community License](https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf),
which should be reviewed before downloading or using them. To inspect the four
downloads without fetching anything, run:

```shell
download-krea2.sh --dry-run
```

### 2. Initialize the character workflow

Choose a distinctive identity trigger and pass only that trigger—not a class
noun. This guide uses `k2v9`:

```shell
init-krea2-character.sh --trigger k2v9
```

Initialization creates editable files under `/musubi/dataset/krea2` and output
directories under `/musubi/output/krea2-character`. It is safe to rerun:
existing files are preserved unless `--trigger` is explicitly supplied, which
updates the detected trigger in `samples.txt` atomically.

### 3. Add paired images and captions

Put 20-30 curated images under `/musubi/dataset/krea2/images`. Every image must
have a UTF-8 `.txt` caption with the same basename:

```text
images/001.png
images/001.txt
images/002.jpg
images/002.txt
```

Describe the subject with an appropriate class noun in ordinary prose, then
append the unique trigger. For example:

```text
A woman is shown in a thigh-up front three-quarter view on a bright futuristic ship bridge, holding an open translucent cyan command fan beside her with a calm slight smile. k2v9
```

Here, `k2v9` is the learned identity and `woman` is an existing base-model
category. Use the most accurate class for the character; do not combine them
into an indivisible trigger phrase such as `k2v9 woman`.

### 4. Review the preview prompts

Open `/musubi/dataset/krea2/samples.txt`. Confirm that its `# trigger:` header
contains `k2v9`, then adjust the generic `person` wording to the character's
appropriate class if useful. Keep the fixed seeds and core prompt pack stable
when comparing checkpoints.

The bundled prompts are medium-neutral and do not require photography, realism,
anime, or another rendering style.

One trigger-free leakage control is active by default. It repeats the first
portrait's wording and seed without `k2v9`, making the two images a direct A/B
comparison. If the control starts resembling the trained identity, the LoRA is
beginning to affect the generic class without being invoked.

If `samples.txt` came from the older combined `trigger class` convention,
rerunning initialization with `--trigger k2v9` updates the metadata and old
trigger occurrences but preserves the surrounding prompt prose. Review those
prompts and add the class noun back as ordinary descriptive language where
needed.

### 5. Prepare the dataset

```shell
prepare-krea2-character.sh
```

Preparation checks images, captions, sample prompts, and trigger consistency,
then builds latent and text-encoder caches. Run it again whenever an image or
caption changes.

### 6. Train

```shell
train-krea2-character.sh
```

The default run trains for 1,800 steps and generates a checkpoint plus fixed-seed
Turbo previews every 200 steps.

### 7. Inspect the results

Default artifacts are written beneath:

```text
/musubi/output/krea2-character/
```

This includes LoRA checkpoints, previews, resumable state, and TensorBoard
logs. Start TensorBoard in another shell or tmux pane:

```shell
tensorboard --logdir /musubi/output/krea2-character
```

Do not assume the final checkpoint is best. Compare the saved previews and
prefer the earliest checkpoint that has reliable identity without excessive
clothing, pose, or background bias.

## Other presets

The default, quality, and attention presets all require the shared 1024-pixel
cache. Prepare once, then run any of them:

```shell
prepare-krea2-character.sh

train-krea2-character.sh
train-krea2-character.sh --preset quality
train-krea2-character.sh --preset attention
```

| Preset | Intended hardware | Resolution | Targets | Rank/alpha | Steps | Output directory |
| --- | --- | --- | --- | --- | --- | --- |
| `default` | 32 GB VRAM | 1024 | All Linear layers | 32/32 | 1,800 | `output/krea2-character` |
| `quality` | 32 GB VRAM | 1024 | All Linear layers | 64/64 | 2,700 | `output/krea2-character/quality` |
| `attention` | 32 GB VRAM | 1024 | Attention projections | 64/64 | 3,600 | `output/krea2-character/attention` |
| `10gb` | 10 GB VRAM | 640 | Attention projections | 16/16 | 800 | `output/krea2-character/10gb` |

The 10 GB preset needs its own resolution-specific cache:

```shell
prepare-krea2-character.sh --preset 10gb
train-krea2-character.sh --preset 10gb
```

It has completed on a 10 GB RTX 3080. Other cards and hosts may behave
differently, and substantial system RAM is required for CPU block swapping.

## Common follow-up commands

After changing an image or caption:

```shell
prepare-krea2-character.sh
train-krea2-character.sh
```

Resume from a saved state by forwarding Musubi's `--resume` option:

```shell
train-krea2-character.sh \
  --resume /path/to/saved-state-directory
```

Use the same preset that created the state, for example:

```shell
train-krea2-character.sh --preset quality \
  --resume /path/to/quality-saved-state-directory
```

Choose a completely custom LoRA filename instead of the automatic
trigger-derived name:

```shell
train-krea2-character.sh --output_name my-custom-lora
```

Override a TOML value for one run by passing the corresponding Musubi option:

```shell
train-krea2-character.sh --max_train_steps 2000
```

## Quick troubleshooting

| Problem | What to do |
| --- | --- |
| Trigger validation fails | Put the exact identity token from `# trigger:` in every primary caption and active sample prompt |
| Images or captions changed | Run `prepare-krea2-character.sh` again before training |
| 32 GB run is out of memory | Close other GPU applications; use the `10gb` preset if necessary |
| 10 GB run is out of memory | Lower `resolution` in `dataset-10gb.toml`, then rerun 10 GB preparation |
| Training was interrupted | Resume the saved state with the same preset and `--resume` |
| LoRA copies clothing or backgrounds | Improve captions, compare an earlier checkpoint, or consider regularization |

## Appendix

### A. Generated layout

Initialization scaffolds the following persisted structure:

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

The model download layout is:

```text
models/
    hf-cache/
    krea2/raw.safetensors
    krea2/turbo.safetensors
    vae/qwen_image_vae.safetensors
    text_encoders/qwen3vl_4b_bf16.safetensors
```

Existing regular files are skipped during downloads. Hugging Face download
cache data stays beneath `models/hf-cache`.

### B. Trigger and caption rules

Keep the unique identity trigger separate from the class noun:

```text
k2v9   = learned identity
woman  = existing base-model category
```

Recommended caption pattern:

```text
<literal description using the class noun>. k2v9
```

Describe only what is visible in that image. Mention variable details such as
framing, viewpoint, expression, clothing, accessories, props, lighting, and
background when they should remain prompt-controllable. Avoid repeatedly
describing stable facial identity traits that should be absorbed into `k2v9`.

Use one spelling of the identity trigger everywhere. A readable, distinctive
fictional name is acceptable and does not need to be one tokenizer token. Do
not use multiple aliases for the same identity.

At inference, both of these are reasonable:

```text
A portrait of k2v9
A portrait of k2v9, a woman wearing a naval-style uniform
```

The second supplies more explicit semantic structure, while the trigger itself
should carry the identity.

`samples.txt` is the trigger source of truth. Preparation requires its exact
trigger in every primary caption and active sample prompt. A prompt immediately
following `# trigger-check: allow-next` is exempt so the bundled trigger-free
leakage control can remain active. Regularization captions are not
required to contain the identity trigger.

For an unusual workflow, override the validation trigger or explicitly skip
only the trigger check:

```shell
prepare-krea2-character.sh --trigger k2v9
prepare-krea2-character.sh --skip-trigger-check
```

Image and caption existence checks still run when trigger checking is skipped.

### C. Preset details

The three 32 GB presets keep the dataset, 1024-pixel resolution, seed, BF16
precision, scaled FP8 base weights, gradient checkpointing, SDPA, constant
scheduler, AdamW8bit, checkpoint cadence, and Turbo preview pack fixed.

The differences are deliberate:

- `default` is the rank-32 reference run at `1e-4` for 1,800 steps.
- `quality` doubles all-linear rank to 64, lowers LR to `7e-5`, and runs for
  2,700 steps.
- `attention` uses rank 64 at `5e-5`, narrows training to 140 attention
  projections, and runs for 3,600 steps to preserve prompt adherence during a
  longer experiment.

With 30 images and batch size 1, those are approximately 60, 90, and 120
dataset passes. All save checkpoints, state, and fixed-seed Turbo previews
every 200 steps. Frequent saves are intentional because later checkpoints may
strengthen likeness while weakening prompt control.

All 32 GB presets use Musubi's resolution-aware `krea2_shift` schedule. The
shared dataset keeps 1024 buckets enabled with `bucket_no_upscale = true`.
Clean 1024x1024 sources therefore enter the square bucket without resizing or
cropping, while future non-square additions remain supported.

### D. Preparation and cache behavior

Preparation validates every dataset declared in the selected dataset TOML. It
then removes only generated Krea2 cache files (`*_kr2.safetensors` and
`*_kr2_te.safetensors`) from each configured cache directory before rebuilding
them. Other files are preserved, and deleted or renamed images cannot survive
through stale cache entries.

The three 32 GB presets share `dataset/krea2/cache`. The 10 GB preset uses
`dataset/krea2/cache-10gb` because its 640-pixel latents are incompatible with
the 1024-pixel cache. Run the matching preparation command whenever images,
captions, or resolution change.

### E. 10 GB implementation details

The `10gb` preset uses BF16, scaled FP8 base weights, gradient checkpointing,
SDPA, AdamW8bit, batch size 1, rank/alpha 16, attention-only targets, and 26 of
28 main blocks swapped to CPU. Text-encoder caching runs on CPU.

Training-time sampling and Turbo weight loading are disabled because Musubi
does not allow Turbo previews together with block swapping. Expect training to
be much slower than the 32 GB workflow. Prefer 64 GB of system RAM; 32 GB plus
a large swap file may work very slowly.

If 640 pixels still runs out of VRAM, edit `dataset-10gb.toml` and try 512,
448, or 384 pixels. Rebuild the 10 GB cache after changing resolution. Reducing
rank and alpha in `train-10gb.toml` is another fallback.

### F. Persistence, overrides, and automatic names

Initialization deliberately preserves existing TOMLs. Rerunning it installs
missing templates but does not overwrite edits or automatically apply newer
template defaults to an existing workflow. Compare an old persisted TOML with
the bundled template when adopting changed defaults.

The same preservation rule applies to `samples.txt`. Supplying `--trigger`
updates its detected trigger without replacing customized prompts, so migrations
from a combined trigger/class phrase may need a one-time manual prompt edit.

The launcher forwards all arguments except its own `--preset` selector to
Musubi, so command-line values override the selected TOML. This includes model
paths, output paths, step counts, configuration files, and `--resume`.

Unless `--output_name` is supplied, the launcher reads the trigger from
`samples.txt` and inserts it into the configured name. With `k2v9`, the default
names are:

```text
krea2-k2v9-character-lora
krea2-k2v9-character-lora-quality
krea2-k2v9-character-lora-attention
krea2-k2v9-character-lora-10gb
```

### G. Why RAW training and Turbo previews

Krea2 provides two DiT checkpoints. Krea-2-Raw is the undistilled training
checkpoint; Krea-2-Turbo is the distilled few-step inference checkpoint. The
recommended workflow is to train the LoRA on Raw and evaluate it on Turbo.

The bundled Turbo preview pack uses 1024x1024 images, 8 inference steps, CFG
off (`--l 1` in Musubi's sample syntax), and fixed seeds. Resident Turbo caching
is disabled to avoid holding a second DiT in system RAM for the entire run.
Block swapping is disabled in 32 GB presets because Musubi cannot combine it
with Turbo sampling.

## Further reading

- [Krea2 character LoRA best practices](KREA2-BEST-PRACTICES.md)
- [Musubi Tuner Krea2 documentation](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md)
- [Diffusers Krea2 LoRA guide](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_krea2.md)
- [Krea training documentation](https://www.krea.ai/docs/user-guide/features/training)
- [Krea 2 Community License](https://www.krea.ai/krea-2-licensing)
- [Krea Acceptable Use Policy](https://www.krea.ai/krea-2-use-policy)
