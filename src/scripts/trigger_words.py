#!/usr/bin/env python3
"""Manage and validate trigger words for Musubi training workflows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import unicodedata


TRIGGER_HEADER = re.compile(r"(?mi)^#\s*trigger:\s*([^\r\n]*?)\s*$")
LEGACY_TRIGGER_HEADER = re.compile(
    r'(?mi)^#\s*Replace every occurrence of\s+"([^"\r\n]+)"'
)
ALLOW_NEXT_PROMPT = "# trigger-check: allow-next"


class TriggerWordsError(ValueError):
    """Raised for an actionable trigger-word configuration error."""


def extract_trigger(text: str) -> str | None:
    """Return trigger metadata from a samples file, including the legacy format."""
    match = TRIGGER_HEADER.search(text) or LEGACY_TRIGGER_HEADER.search(text)
    if match is None or not match.group(1).strip():
        return None
    return match.group(1).strip()


def validate_trigger(trigger: str) -> None:
    if not trigger.strip() or "\n" in trigger or "\r" in trigger:
        raise TriggerWordsError("The trigger must contain visible text on one line.")


def read_trigger(samples_path: Path) -> str:
    trigger = extract_trigger(samples_path.read_text(encoding="utf-8-sig"))
    if trigger is None:
        raise TriggerWordsError(
            f"No trigger metadata was found in {samples_path}. "
            "Add '# trigger: unique-token'."
        )
    return trigger


def trigger_slug(trigger: str) -> str:
    validate_trigger(trigger)
    trigger_token = trigger.split()[0]
    slug = unicodedata.normalize("NFKD", trigger_token)
    slug = slug.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug).strip("-._")
    if not slug:
        slug = re.sub(r"[^\w.-]+", "-", trigger_token.lower()).strip("-._")
    if not slug:
        raise TriggerWordsError(
            f'Unable to create a safe output-name slug from trigger token "{trigger_token}".'
        )
    return slug


def output_name_for_trigger(
    trigger: str, base_name: str, insert_after_prefix: str | None = None
) -> str:
    """Add the trigger token to an output name unless it is already present."""
    if not base_name:
        raise TriggerWordsError("The base output name must not be empty.")

    slug = trigger_slug(trigger)
    name_segments = [part for part in re.split(r"[-_.]+", base_name.lower()) if part]
    if slug in name_segments:
        return base_name

    if insert_after_prefix:
        prefix = f"{insert_after_prefix}-"
        if base_name.lower().startswith(prefix.lower()):
            return f"{base_name[:len(prefix)]}{slug}-{base_name[len(prefix):]}"

    return f"{slug}-{base_name}"


def write_atomically(path: Path, contents: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(contents)
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def set_samples_trigger(
    template_path: Path, samples_path: Path, trigger: str, placeholder: str
) -> str:
    """Create a samples file or atomically replace its currently declared trigger."""
    validate_trigger(trigger)

    if samples_path.exists():
        if not samples_path.is_file():
            raise TriggerWordsError(f"Samples path is not a regular file: {samples_path}")
        original = samples_path.read_text(encoding="utf-8-sig")
        old_trigger = extract_trigger(original)
        if old_trigger is None:
            raise TriggerWordsError(
                f"Cannot detect the existing trigger in {samples_path}. "
                "Add '# trigger: unique-token' or edit the file manually."
            )

        updated = original.replace(old_trigger, trigger)
        if TRIGGER_HEADER.search(original) is None:
            updated = f"# trigger: {trigger}\n{updated}"
        mode = stat.S_IMODE(samples_path.stat().st_mode)
        action = "Trigger already set in" if updated == original else "Updated trigger in"
    else:
        if not template_path.is_file():
            raise TriggerWordsError(f"Missing samples template: {template_path}")
        original = template_path.read_text(encoding="utf-8")
        if placeholder not in original:
            raise TriggerWordsError(
                f"Missing {placeholder} in bundled template: {template_path}"
            )
        updated = original.replace(placeholder, trigger)
        mode = 0o644
        action = "Created"

    if not samples_path.exists() or updated != original:
        write_atomically(samples_path, updated, mode)

    return f"{action}: {samples_path}"


def active_sample_prompts(samples_text: str) -> list[tuple[int, str, bool]]:
    """Return active prompts and whether each prompt may omit the trigger."""
    prompts: list[tuple[int, str, bool]] = []
    allow_next_without_trigger = False
    for line_number, line in enumerate(samples_text.splitlines(), start=1):
        prompt = line.strip()
        if not prompt:
            continue
        if prompt.lower() == ALLOW_NEXT_PROMPT:
            allow_next_without_trigger = True
            continue
        if prompt.startswith("#"):
            allow_next_without_trigger = False
            continue
        prompts.append((line_number, prompt, allow_next_without_trigger))
        allow_next_without_trigger = False
    return prompts


def load_dataset_config(dataset_config_path: Path):
    """Load a Musubi dataset config and its canonical image discovery helper."""
    try:
        import toml
        from musubi_tuner.dataset.media_utils import glob_images
    except ImportError as error:
        raise TriggerWordsError(
            "Dataset operations require Musubi Tuner and its Python dependencies."
        ) from error

    return toml.load(str(dataset_config_path)), glob_images


def resolved_dataset_value(
    dataset: dict[str, object], general: dict[str, object], key: str
):
    """Resolve a dataset value using Musubi's dataset-over-general precedence."""
    value = dataset.get(key)
    return general.get(key) if value is None else value


def supported_images(dataset: dict[str, object], glob_images) -> list[str]:
    """Return the Musubi-supported images for an image-directory dataset."""
    image_directory = dataset.get("image_directory")
    if not isinstance(image_directory, str) or not image_directory:
        return []
    return glob_images(os.path.abspath(image_directory))


def inspect_dataset(dataset_config_path: Path) -> dict[str, object]:
    """Describe whether a simple primary-dataset pass estimate is authoritative."""
    config, glob_images = load_dataset_config(dataset_config_path)
    general = config.get("general", {})
    datasets = config.get("datasets", [])
    if not isinstance(general, dict):
        general = {}
    if not isinstance(datasets, list) or not datasets:
        raise TriggerWordsError(f"No datasets are configured in {dataset_config_path}")

    primary = datasets[0]
    if not isinstance(primary, dict):
        raise TriggerWordsError(
            f"Dataset 1 in {dataset_config_path} is not a TOML table."
        )

    additional_dataset_count = len(datasets) - 1
    batch_size_value = resolved_dataset_value(primary, general, "batch_size")
    per_device_batch_size = (
        batch_size_value
        if isinstance(batch_size_value, int)
        and not isinstance(batch_size_value, bool)
        and batch_size_value > 0
        else None
    )
    result: dict[str, object] = {
        "layout": "custom",
        "primary_image_count": None,
        "additional_dataset_count": additional_dataset_count,
        "per_device_batch_size": per_device_batch_size,
        "caption_pairs_complete": None,
        "estimate_authoritative": False,
        "estimate_unavailable_reason": "custom dataset configuration",
    }

    image_directory = primary.get("image_directory")
    image_jsonl_file = primary.get("image_jsonl_file")
    if isinstance(image_jsonl_file, str) and image_jsonl_file:
        result["layout"] = "jsonl"
        result["estimate_unavailable_reason"] = (
            "primary dataset uses image_jsonl_file"
        )
        return result

    if not isinstance(image_directory, str) or not image_directory:
        result["estimate_unavailable_reason"] = (
            "primary dataset does not use image_directory"
        )
        return result

    image_directory = os.path.abspath(image_directory)
    if not os.path.isdir(image_directory):
        raise TriggerWordsError(
            f"Dataset 1 image directory does not exist: {image_directory}"
        )

    images = supported_images(primary, glob_images)
    if not images:
        raise TriggerWordsError(
            f"Dataset 1 has no supported training images: {image_directory}"
        )
    result["primary_image_count"] = len(images)

    if additional_dataset_count:
        result["layout"] = "multi-dataset"
        result["estimate_unavailable_reason"] = "multi-dataset configuration"
        return result

    caption_extension = resolved_dataset_value(
        primary, general, "caption_extension"
    )
    if not isinstance(caption_extension, str) or not caption_extension:
        result["estimate_unavailable_reason"] = (
            "caption extension is not configured"
        )
        return result
    if per_device_batch_size is None:
        result["estimate_unavailable_reason"] = (
            "per-device batch size is not a positive integer"
        )
        return result

    missing_captions = [
        os.path.splitext(image_path)[0] + caption_extension
        for image_path in images
        if not os.path.isfile(os.path.splitext(image_path)[0] + caption_extension)
    ]
    if missing_captions:
        details = "\n".join(
            f"Missing caption for dataset 1 training image: {caption_path}"
            for caption_path in missing_captions
        )
        raise TriggerWordsError(details)

    result.update(
        {
            "layout": "standard-directory",
            "caption_pairs_complete": True,
            "estimate_authoritative": True,
            "estimate_unavailable_reason": None,
        }
    )
    return result


def validate_dataset(
    dataset_config_path: Path,
    samples_path: Path | None,
    trigger_override: str | None,
    skip_trigger_check: bool,
) -> None:
    """Validate Musubi image/caption datasets and their trigger-word usage."""
    config, glob_images = load_dataset_config(dataset_config_path)
    general = config.get("general", {})
    datasets = config.get("datasets", [])
    errors: list[str] = []
    trigger: str | None = None
    prompts: list[tuple[int, str, bool]] = []
    validated_captions = 0
    primary_uses_jsonl = False

    if not skip_trigger_check:
        if samples_path is None:
            errors.append("A samples file is required unless trigger validation is skipped.")
            samples_text = ""
        else:
            samples_text = samples_path.read_text(encoding="utf-8-sig")

        if trigger_override is not None:
            validate_trigger(trigger_override)
            trigger = trigger_override
        else:
            trigger = extract_trigger(samples_text)

        if trigger is None:
            errors.append(
                f"No trigger metadata was found in {samples_path}. "
                "Add '# trigger: unique-token' or pass --trigger."
            )

        prompts = active_sample_prompts(samples_text)
        if not prompts:
            errors.append(f"No active sample prompts were found in {samples_path}")
        elif trigger:
            for line_number, prompt, trigger_optional in prompts:
                if not trigger_optional and trigger not in prompt:
                    errors.append(
                        f'Trigger "{trigger}" is missing from active sample prompt '
                        f"{samples_path}:{line_number}"
                    )

    if not datasets:
        errors.append(f"No datasets are configured in {dataset_config_path}")

    for index, dataset in enumerate(datasets, start=1):
        image_directory = dataset.get("image_directory")
        image_jsonl_file = dataset.get("image_jsonl_file")

        if image_directory:
            image_directory = os.path.abspath(image_directory)
            if not os.path.isdir(image_directory):
                errors.append(
                    f"Dataset {index} image directory does not exist: {image_directory}"
                )
                continue

            images = supported_images(dataset, glob_images)
            if not images:
                errors.append(
                    f"Dataset {index} has no supported training images: {image_directory}"
                )
                continue

            caption_extension = resolved_dataset_value(
                dataset, general, "caption_extension"
            )
            if caption_extension:
                for image_path in images:
                    caption_path = os.path.splitext(image_path)[0] + caption_extension
                    if not os.path.isfile(caption_path):
                        errors.append(
                            f"Missing caption for dataset {index} training image: "
                            f"{caption_path}"
                        )
                    elif not skip_trigger_check and index == 1 and trigger:
                        try:
                            caption = Path(caption_path).read_text(encoding="utf-8-sig")
                        except UnicodeDecodeError:
                            errors.append(f"Caption is not valid UTF-8: {caption_path}")
                            continue
                        if trigger not in caption:
                            errors.append(
                                f'Trigger "{trigger}" is missing from primary training '
                                f"caption: {caption_path}"
                            )
                        else:
                            validated_captions += 1
            elif not skip_trigger_check and index == 1:
                errors.append(
                    "Dataset 1 has no caption_extension, so its trigger cannot be validated."
                )
        elif image_jsonl_file:
            image_jsonl_file = os.path.abspath(image_jsonl_file)
            if not os.path.isfile(image_jsonl_file):
                errors.append(
                    f"Dataset {index} image JSONL file does not exist: {image_jsonl_file}"
                )
            elif not skip_trigger_check and index == 1:
                primary_uses_jsonl = True
        else:
            errors.append(
                f"Dataset {index} has neither image_directory nor image_jsonl_file configured"
            )

    if errors:
        raise TriggerWordsError("\n".join(errors))

    if skip_trigger_check:
        print("Trigger validation skipped by request.")
    elif primary_uses_jsonl:
        print(
            f"Validated trigger rules for {len(prompts)} active sample prompts; "
            "primary JSONL captions were not inspected."
        )
    else:
        print(
            f'Validated trigger "{trigger}" in {validated_captions} primary captions '
            f"and trigger rules for {len(prompts)} active sample prompts."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage and validate trigger words for Musubi training workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    get_parser = subparsers.add_parser("get", help="print trigger metadata")
    get_parser.add_argument("--samples", required=True, type=Path)

    set_parser = subparsers.add_parser(
        "set", help="create a samples file or replace its declared trigger"
    )
    set_parser.add_argument("--template", required=True, type=Path)
    set_parser.add_argument("--samples", required=True, type=Path)
    set_parser.add_argument("--trigger", required=True)
    set_parser.add_argument("--placeholder", required=True)

    name_parser = subparsers.add_parser(
        "output-name", help="add the samples trigger token to an output name"
    )
    name_parser.add_argument("--samples", required=True, type=Path)
    name_parser.add_argument("--base-name", required=True)
    name_parser.add_argument("--insert-after-prefix")

    validate_parser = subparsers.add_parser(
        "validate", help="validate a Musubi dataset and its trigger usage"
    )
    validate_parser.add_argument("--dataset-config", required=True, type=Path)
    validate_parser.add_argument("--samples", type=Path)
    validate_parser.add_argument("--trigger")
    validate_parser.add_argument("--skip-trigger-check", action="store_true")

    inspect_parser = subparsers.add_parser(
        "inspect", help="print structured dataset planning information as JSON"
    )
    inspect_parser.add_argument("--dataset-config", required=True, type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "get":
            print(read_trigger(args.samples))
        elif args.command == "set":
            print(
                set_samples_trigger(
                    args.template, args.samples, args.trigger, args.placeholder
                )
            )
        elif args.command == "output-name":
            trigger = read_trigger(args.samples)
            print(
                output_name_for_trigger(
                    trigger, args.base_name, args.insert_after_prefix
                )
            )
        elif args.command == "validate":
            validate_dataset(
                args.dataset_config,
                args.samples,
                args.trigger,
                args.skip_trigger_check,
            )
        elif args.command == "inspect":
            print(json.dumps(inspect_dataset(args.dataset_config), sort_keys=True))
        else:  # pragma: no cover - argparse enforces a known subcommand.
            raise AssertionError(f"Unhandled command: {args.command}")
    except (OSError, UnicodeError, TriggerWordsError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
