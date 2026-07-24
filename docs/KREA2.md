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
train-krea2-character.sh --preset baseline
train-krea2-character.sh --preset quality
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

## Upload checkpoints to Hugging Face

Create a private Hugging Face model repository before training, then create a
fine-grained token with write access to that repository. Enter the token inside
the training shell or tmux session without putting it in shell history:

```shell
read -rsp "Hugging Face write token: " HF_TOKEN
echo
export HF_TOKEN
```

Add the repository to any Krea2 training command:

```shell
train-krea2-character.sh --hf-repo OWNER/REPO
```

The launcher verifies that the existing repository is accessible and writable,
then creates a token-free `run.json` before starting Accelerate. It prints the
exact destination, which defaults to a new UTC-stamped path:

```text
Hugging Face repository:        OWNER/REPO
Hugging Face path:              krea2/krea2-k2v9-character-lora/20260724T120000Z
Hugging Face artifacts:         LoRA checkpoints only (synchronous)
```

Supply a stable or descriptive path when preferred:

```shell
train-krea2-character.sh \
  --hf-repo OWNER/REPO \
  --hf-path krea2/k2v9/default-search
```

Musubi uploads every periodic `.safetensors` checkpoint and the final
checkpoint synchronously. Training pauses while each upload is attempted.
Generated previews, resumable optimizer state, TensorBoard logs, source images,
and captions remain local.

The preflight stops before training if authentication, repository access, or
the initial write fails. Later checkpoint uploads use Musubi's native
best-effort behavior: an upload error is logged and training continues without
automatic retry.

Download all checkpoints for a run on another machine with the Hugging Face
CLI:

```shell
hf download OWNER/REPO \
  --include 'krea2/krea2-k2v9-character-lora/20260724T120000Z/*.safetensors' \
  --local-dir ./krea2-checkpoints
```

The token can be removed from the shell after training:

```shell
unset HF_TOKEN
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

The default run uses rank-64 attention-only adaptation for an 8,000-step search
horizon. It generates a step-numbered checkpoint plus fixed-seed Turbo previews
every 400 steps, giving 20 periodic candidates.

Before Accelerate starts, the launcher reports the effective target modules,
rank, learning rate, dataset shape, batch settings, horizon, estimated dataset
passes, and checkpoint counts. For the standard 30-image layout, the default
summary includes:

```text
Krea2 training search horizon

Preset:                         default
Target modules:                 attention-only
Rank / alpha:                   64 / 64
Learning rate:                  5e-5

Primary paired images:          30
Per-device batch size:          1
Gradient accumulation:          1
Effective single-GPU batch:     1

Maximum optimizer steps:        8000
Estimated maximum passes:       266.7
Checkpoint interval:            400
Periodic checkpoint candidates: 20
Final checkpoint:               duplicates step 8000
Unique candidate states:        20
Sample interval:                400
Resumable-state window:         800 steps
```

The estimate is authoritative only for one directory-backed dataset with a
resolved batch size and complete same-basename caption pairs. A JSONL primary,
an enabled regularization dataset, or another custom multi-dataset layout is
valid, but the launcher reports its simple pass estimate as unavailable.
For an authoritative standard layout, it warns without blocking when the
primary count is outside the presets' intended 20-40 image range.

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
clothing, pose, or background bias. A run is comfortably long enough when at
least four later periodic checkpoints fail to improve the overall
identity/control trade-off.

## Other presets

The default, baseline, and quality presets all require the shared 1024-pixel
cache. Prepare once, then run any of them:

```shell
prepare-krea2-character.sh

train-krea2-character.sh
train-krea2-character.sh --preset baseline
train-krea2-character.sh --preset quality
```

| Preset | Hardware | Resolution | Targets | Rank/alpha | Horizon | Checkpoint / sample | State window | Output directory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `default` | 32 GB | 1024 | Attention projections | 64/64 | 8,000 | 400 / 400 | 800 | `output/krea2-character` |
| `baseline` | 32 GB | 1024 | All Linear layers | 32/32 | 4,000 | 200 / 200 | 400 | `output/krea2-character/baseline` |
| `quality` | 32 GB | 1024 | All Linear layers | 64/64 | 6,000 | 300 / 300 | 600 | `output/krea2-character/quality` |
| `10gb` | 10 GB | 640 | Attention projections | 16/16 | 800 | 100 / disabled | None | `output/krea2-character/10gb` |

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
| Pass estimate is unavailable | Custom JSONL and multi-dataset layouts cannot be represented by the simple primary-directory estimate |
| 32 GB run is out of memory | Close other GPU applications; use the `10gb` preset if necessary |
| 10 GB run is out of memory | Lower `resolution` in `dataset-10gb.toml`, then rerun 10 GB preparation |
| Training was interrupted | Resume the saved state with the same preset and `--resume` |
| Hugging Face preflight fails | Confirm `HF_TOKEN` is set, the model repository already exists, and the token has write access |
| LoRA copies clothing or backgrounds | Improve captions, compare an earlier checkpoint, or consider regularization |

## Appendix

### A. Generated layout

Initialization scaffolds the following persisted structure:

```text
dataset/krea2/
    dataset.toml
    train.toml
    train-baseline.toml
    train-quality.toml
    dataset-10gb.toml
    train-10gb.toml
    samples.txt
    images/
    cache/
    cache-10gb/
output/krea2-character/
    logs/
    baseline/logs/
    quality/logs/
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
scheduler, AdamW8bit, and Turbo preview pack fixed.

The differences are deliberate:

- `default` uses rank 64 at `5e-5` and narrows training to 140 attention
  projections for an 8,000-step, prompt-control-oriented search.
- `baseline` is the compact rank-32 all-linear reference at `1e-4` for 4,000
  steps.
- `quality` uses rank-64 all-linear capacity at `7e-5` for 6,000 steps.

With 30 images, per-device batch size 1, and gradient accumulation 1, those
horizons are approximately 266.7, 133.3, and 200.0 passes for `default`,
`baseline`, and `quality`. They save and sample every 400, 200, and 300 steps
respectively, giving exactly 20 step-numbered candidate checkpoints each.
Their similar interval-times-learning-rate values are incidental engineering
symmetry for checkpoint spacing, not evidence for the horizon lengths.

The three 32 GB templates intentionally omit `save_last_n_steps`, so every
periodic LoRA checkpoint remains available. Musubi also writes one unsuffixed
model at normal completion. Because each bundled horizon is divisible by its
checkpoint interval, that file duplicates the twentieth periodic training
state: there are 21 model files but 20 unique trained states.

Resumable state is separate and much larger. The `save_last_n_steps_state`
values are rolling windows measured in steps, not counts of state directories;
the default, baseline, and quality windows are 800, 400, and 600 steps
respectively. They retain only recent recovery points while model checkpoint
deletion remains disabled.

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
missing templates but does not overwrite edits.

The same preservation rule applies to `samples.txt`. Supplying `--trigger`
updates its detected trigger without replacing customized prompts, so migrations
from a combined trigger/class phrase may need a one-time manual prompt edit.

The launcher forwards all arguments except its own `--preset` selector to
Musubi, so command-line values override the selected TOML. This includes model
paths, output paths, step counts, save/sample cadence, accumulation, network
settings, configuration files, and `--resume`. The printed planning summary
uses those effective values. If the horizon is not divisible by its checkpoint
interval, the unsuffixed final model is reported as one additional unique
state.

Unless `--output_name` is supplied, the launcher reads the trigger from
`samples.txt` and inserts it into the configured name. With `k2v9`, the default
names are:

```text
krea2-k2v9-character-lora
krea2-k2v9-character-lora-baseline
krea2-k2v9-character-lora-quality
krea2-k2v9-character-lora-10gb
```

The output tree must outlive an ephemeral training VM unless checkpoint upload
is enabled with `--hf-repo`. That option copies LoRA checkpoints to an existing
Hugging Face model repository, but resumable state remains local. Mount the
output tree from persistent storage when remote resume is required.

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
