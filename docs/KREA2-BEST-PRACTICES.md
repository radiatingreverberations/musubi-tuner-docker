# Krea2 character LoRA best practices

This is the practical companion to the command-focused
[Krea2 workflow](KREA2.md). It focuses on decisions that most affect character
quality: dataset curation, captions, validation, and checkpoint selection. The
bundled TOML presets remain the source of truth for exact Musubi settings.

## The short version

- Train the LoRA on Krea-2-Raw; preview and use it on Krea-2-Turbo.
- Prefer 20-30 strong, varied images over a larger repetitive dataset.
- Put one unique identity trigger, such as `k2v9`, in every primary caption.
- Describe the subject with an appropriate class noun separately from the
  trigger.
- Caption only visible attributes, especially details that should remain
  prompt-controllable.
- Reuse the same prompts and seeds at every checkpoint.
- Choose the earliest checkpoint with reliable identity and good prompt
  control, not automatically the final checkpoint.

## Curate for identity and control

The goal is not merely to make the subject recognizable. A useful character
LoRA should retain identity while still accepting changes to pose, clothing,
setting, viewpoint, and rendering style.

For a 20-30 image dataset, aim for a deliberate mixture:

| Include | Why |
| --- | --- |
| Tight face and head-and-shoulders views | Establish facial identity and small details |
| Three-quarter and profile views | Reduce front-view dependence |
| Half-body and seated views | Teach posture and intermediate framing |
| A few full-body views with a clearly visible subject | Improve wider compositions without weakening the face signal |
| Different expressions, clothing, lighting, and backgrounds | Separate identity from incidental attributes |

Remove images that are blurry, heavily compressed, near-duplicates, strongly
beautified, or so wide that the character contributes little identity signal.
Correct rotation and obvious exposure outliers, but avoid synthetic face
enhancement, aggressive warps, heavy filters, and other preprocessing that
changes defining geometry.

For the bundled 32 GB presets, clean 1024x1024 sources are ideal. They enter the
1024 square bucket without resizing or cropping. Do not upscale them to 1280;
that increases work without adding identity information. The 10 GB preset uses
its own 640-pixel cache.

If enough images are available, keep 3-5 representative images out of training.
Use them only as a reference when comparing generated identity across
checkpoints.

## Write captions that disentangle the character

Use one short, distinctive identity token. Keep the class noun separate: the
token represents the learned identity, while a word such as `woman`, `man`, or
`person` supplies a category the base model already understands. The launcher
also uses the trigger in checkpoint names:

```text
k2v9 = learned identity
woman = existing base-model category
```

Use the exact same trigger in every primary caption and in `samples.txt`, but
write the class noun naturally in the description. Choose the most accurate
stable class: for an adult woman, prefer `woman` over the less specific
`person`. Avoid aliases and spelling variations. A readable, distinctive
fictional name can work; it does not need to be one tokenizer token. A legal
name is usually a worse identifier than a neutral invented token.

Describe what is visible in each image—no more and no less. A close portrait
should not describe shoes or an unseen outfit; a full-body image can describe
the visible clothing and props.

A useful caption answers three questions:

1. What is visibly happening in this frame?
2. Which visible attributes should remain changeable at inference time?
3. Which incidental detail could bind to the identity if it goes unnamed?

Example captions:

```text
A woman is shown in a close portrait, looking slightly left with loose dark hair against a plain background. k2v9
A woman is seated at an outdoor cafe wearing sunglasses and a light jacket, with a city street behind her. k2v9
A woman stands in a relaxed full-body pose wearing a red coat in a modern interior. k2v9
```

The recommended pattern is:

```text
<literal description using the class noun>. k2v9
```

Do not treat `k2v9 woman` as one indivisible trigger phrase. At inference,
either `A portrait of k2v9` or `A portrait of k2v9, a woman wearing a blue
uniform` can work; the second form gives the base model more explicit semantic
structure.

Mention recurring clothing, glasses, hairstyles, props, lighting, and settings
when they should remain prompt-controllable. Do not add invisible details,
biography, personality claims, or a separate negative-caption convention.
Positive descriptions are the mechanism for separating incidental attributes
from identity.

`prepare-krea2-character.sh` checks the trigger in all primary captions and
active preview prompts before it deletes or rebuilds caches. Regularization
captions are intentionally exempt. See [KREA2.md](KREA2.md) for trigger setup
and the explicit validation escape hatch.

## Treat the bundled presets as separate experiments

Training steps and learning rates are not portable between Musubi, Diffusers,
hosted trainers, and other toolkits. A step in one trainer may represent
different data repetition, batching, target modules, or scheduling in another.
Do not copy a number from another Krea2 recipe into these TOMLs without
accounting for those differences.

For this repository:

- `default` is the rank-32, 1,800-step reference run, with post-1,200
  checkpoints available for comparison.
- `quality` increases all-linear capacity and runs for 2,700 steps.
- `attention` narrows the targets and runs longer for 3,600 steps.
- `10gb` is the block-swapped 640-pixel low-VRAM run.

Change one major variable at a time. Reusing the same dataset, captions,
preview prompts, and seeds makes comparisons meaningful.

## Validate on Turbo with a fixed prompt pack

Krea-2-Raw is the training checkpoint. Judge the LoRA through Krea-2-Turbo,
using the bundled preview convention:

- 8 steps
- CFG off (`--l 1` in the bundled sample syntax)
- fixed seeds
- consistent resolution and prompts

The medium-neutral `samples.txt` varies framing, viewpoint, clothing, pose, and
setting without forcing photography, anime, or another rendering style. Keep
those prompts stable across a comparison. Add prompts only when they test a
specific concern.

A compact evaluation pack should cover:

- neutral close portrait
- three-quarter or profile view
- half-body composition
- full-body composition
- clothing and background changes
- a different rendering style, when style portability matters
- the paired trigger-free leakage control

The trigger-free prompt answers an important question: does enabling the LoRA
alter a generic character even when the trigger is absent? The bundled control
uses the same wording and seed as the first triggered portrait, isolating the
effect of the identity token. If the control starts resembling the trained
character, the LoRA may be too strong or overfit.

For a more rigorous comparison, generate the same grid at each checkpoint and
compare it with the held-out reference images. Face-embedding similarity can be
an additional signal for real-person likeness, but it should not replace visual
checks for prompt adherence, anatomy, and unwanted attribute binding.

## Pick the earliest clean checkpoint

Use this checklist rather than selecting the last file:

| Check | Healthy result | Warning sign |
| --- | --- | --- |
| Identity | Stable across seeds and viewpoints | Resembles the subject only in one pose or seed |
| Prompt adherence | Pose, clothing, setting, and framing respond normally | Trigger overwhelms the rest of the prompt |
| Attribute control | Hair, glasses, clothing, and accessories can change | One-off attributes appear unrequested |
| Background control | New scenes work cleanly | A training room, chair, or backdrop keeps returning |
| Style behavior | Intended styles remain available | Character becomes locked to one unintended aesthetic |
| Image quality | Eyes, silhouette, anatomy, and fine details remain coherent | Waxy, warped, oversharpened, or repetitive details |

Later checkpoints often strengthen likeness while also increasing dataset
leakage. Prefer the earliest checkpoint that is reliably recognizable and still
easy to direct. The bundled 200-step save and preview cadence is designed for
this comparison.

## Respond to common failure modes

| Symptom | First response |
| --- | --- |
| Weak identity everywhere | Check curation and trigger consistency before increasing steps |
| Strong only on close portraits | Add or improve a few half/full-body images with a salient subject |
| Same clothing or background keeps appearing | Caption those attributes explicitly; consider regularization |
| Prompt changes stop working | Try an earlier checkpoint or the attention-only preset |
| Generic prompts change without the trigger | Use an earlier checkpoint or reduce LoRA strength at inference |
| Artifacts increase while likeness improves | Stop earlier; more steps are not the remedy |

If background, wardrobe, or generic-person behavior remains over-bound after
caption improvements, add a modest regularization dataset. Keep its captions
generic and omit the character trigger. Start with roughly comparable subject
and regularization weighting, then reevaluate the same fixed preview pack.

## Final inference

Apply the selected LoRA to Krea-2-Turbo. Start with 8 steps, CFG off, a fixed
seed, and a clear positive prompt. Negative prompts are not a primary control
surface in the canonical Turbo setup. First establish stable identity at a
moderate LoRA strength, then explore composition and style.

## Consent and responsible use

For a real person's likeness, obtain informed consent unless the subject is
yourself, keep source images private, and define allowed uses before sharing a
LoRA. Do not use it for deception, false endorsement, harassment, fraud, or
non-consensual sexual content. Public or commercial use may also engage privacy,
publicity, biometric-data, and platform-policy obligations.

## Before starting a run

- [ ] One identity or character concept, without conflicting redesigns
- [ ] Weak images and near-duplicates removed
- [ ] Useful spread of framing, viewpoints, expressions, and settings
- [ ] Exact identity trigger in every primary caption
- [ ] Appropriate class noun used naturally and separately from the trigger
- [ ] Only visible attributes described
- [ ] `samples.txt` trigger and prompts reviewed
- [ ] Caches rebuilt after the last image or caption change
- [ ] Fixed preview seeds retained
- [ ] Enough disk space for frequent checkpoints and saved state
- [ ] Consent and intended use are clear for real-person likenesses

## Further reading

- [Musubi Tuner Krea2 documentation](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md)
- [Diffusers Krea2 LoRA guide](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_krea2.md)
- [Krea2 LoRA trainer captioning reference](https://huggingface.co/spaces/multimodalart/krea2-lora-trainer/blob/560e932f0e5d81ed9a09b7ea0038e92da4f0ba01/caption.py)
- [Krea training documentation](https://www.krea.ai/docs/user-guide/features/training)
- [Krea 2 Community License](https://www.krea.ai/krea-2-licensing)
- [Krea Acceptable Use Policy](https://www.krea.ai/krea-2-use-policy)
