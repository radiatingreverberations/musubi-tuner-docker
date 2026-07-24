#!/usr/bin/env python3
"""Validate a Krea2 Hugging Face checkpoint destination before training."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import PurePosixPath
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import HFValidationError, validate_repo_id


def repository_id(value: str) -> str:
    try:
        validate_repo_id(value)
    except HFValidationError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if "/" not in value:
        raise argparse.ArgumentTypeError(
            "repository must include its owner or organization (OWNER/REPO)"
        )
    return value


def repository_path(value: str) -> str:
    if not value or value != value.strip() or value.startswith("/") or "\\" in value:
        raise argparse.ArgumentTypeError(
            "path must be a non-empty relative Hugging Face repository path"
        )
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise argparse.ArgumentTypeError(
            "path must not contain empty, current-directory, or parent-directory segments"
        )
    path = PurePosixPath(value)
    return path.as_posix()


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm a Hugging Face model repository is writable and create a "
            "token-free Krea2 run manifest."
        )
    )
    parser.add_argument("--repo", required=True, type=repository_id)
    parser.add_argument("--path", required=True, type=repository_path)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--started-at", required=True)
    return parser


def preflight(args: argparse.Namespace, token: str) -> None:
    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=args.repo, repo_type="model")
    except Exception:
        raise RuntimeError(
            f"Unable to access Hugging Face model repository {args.repo} with HF_TOKEN."
        ) from None

    manifest = {
        "schema_version": 1,
        "workflow": "krea2-character",
        "preset": args.preset,
        "output_name": args.output_name,
        "started_at": args.started_at,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "artifacts": ["lora-checkpoints"],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    manifest_path = f"{args.path}/run.json"
    try:
        api.upload_file(
            repo_id=args.repo,
            repo_type="model",
            path_or_fileobj=manifest_bytes,
            path_in_repo=manifest_path,
            commit_message=f"Start Krea2 checkpoint upload for {args.output_name}",
        )
    except Exception:
        raise RuntimeError(
            f"Unable to write {manifest_path} in Hugging Face model repository "
            f"{args.repo}. Ensure HF_TOKEN has write access."
        ) from None


def main(argv: list[str] | None = None) -> int:
    args = setup_parser().parse_args(argv)
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print(
            "HF_TOKEN is required when --hf-repo is used. "
            "Set a fine-grained token with write access to the existing model repository.",
            file=sys.stderr,
        )
        return 2

    try:
        preflight(args, token)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
