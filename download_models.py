#!/usr/bin/env python3
"""Minimal single-file model downloader.

Purpose: replace earlier workflow JSON + multi-spec logic with a simple
invocation that fetches ONE file (Hugging Face or direct URL) and writes it
into a flat output directory, discarding any intermediate path components.

Examples:
    # Hugging Face repo file (keep only basename in output dir)
    python download_models.py \
        --hf Comfy-Org/Wan_2.1_ComfyUI_repackaged \
        --file split_files/vae/wan_2.1_vae.safetensors \
        --output-dir models/vae

    # Specific revision
    python download_models.py \
        --hf Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
        --rev main \
        --file split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
        --output-dir models/diffusion_models

    # Direct URL
    python download_models.py \
        --url https://example.com/path/model.bin \
        --output-dir models/extras

Destination name rules:
    - By default the basename of the source file/URL path is used.
    - Override with --dest-name NAME.

Behavior:
    - Creates <output-dir>/hf-cache for HF caching.
    - Symlink to cached HF file when possible, else copies.
    - If a regular file already exists at destination, skip (idempotent).
    - If a symlink exists, it is replaced.
"""

from __future__ import annotations

import os
import shutil
import argparse
from urllib.parse import urlparse
import requests
from huggingface_hub import hf_hub_download


def download_hf(repo_id: str, file_path: str, revision: str, dest_abs: str, cache_dir: str, dry_run: bool):
    if dry_run:
        print(f"[dry-run] hf:{repo_id}@{revision}:{file_path} -> {dest_abs}")
        return
    print(f"➡️  Fetch hf:{repo_id}@{revision}:{file_path} -> {dest_abs}")
    cached = hf_hub_download(repo_id=repo_id, filename=file_path, revision=revision, cache_dir=cache_dir)
    cached_abs = os.path.abspath(cached)
    try:
        if os.path.islink(dest_abs):
            os.remove(dest_abs)
        os.symlink(cached_abs, dest_abs)
        print(f"🔗  HF symlink: {repo_id}@{revision}:{file_path} -> {dest_abs}")
    except OSError:
        shutil.copy(cached_abs, dest_abs)
        print(f"✔️  HF copy: {repo_id}@{revision}:{file_path} -> {dest_abs}")
    print(f"✅  Done: {dest_abs}")


def download_url(url: str, dest_abs: str, dry_run: bool):
    if dry_run:
        print(f"[dry-run] url:{url} -> {dest_abs}")
        return
    print(f"➡️  Fetch {url} -> {dest_abs}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest_abs, 'wb') as out:
        for chunk in resp.iter_content(chunk_size=8192):
            out.write(chunk)
    print(f"✔️  HTTP download: {url} -> {dest_abs}")
    print(f"✅  Done: {dest_abs}")


def main():
    p = argparse.ArgumentParser(description="Download a single model/file (HF or direct URL) into a flat directory.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--hf', metavar='REPO_ID', help='Hugging Face repo id, e.g. owner/name')
    src.add_argument('--url', metavar='URL', help='Direct HTTP(S) URL')
    p.add_argument('--file', metavar='PATH', help='File path inside HF repo (required with --hf)')
    p.add_argument('--rev', default='main', help='HF revision (tag/branch/commit), default: main')
    p.add_argument('-o', '--output-dir', required=True, help='Destination directory (absolute, or relative to --base-dir if that is provided)')
    p.add_argument('--base-dir', help='Base directory for a shared hf-cache (defaults to --output-dir if omitted)')
    p.add_argument('--dest-name', help='Override output file name (defaults to basename of source path)')
    p.add_argument('-n', '--dry-run', action='store_true', help='Show actions without downloading')
    args = p.parse_args()

    base_dir = os.path.abspath(args.base_dir) if args.base_dir else None
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
        # Resolve output directory relative to base_dir if not absolute
        if not os.path.isabs(args.output_dir):
            resolved_output_dir = os.path.join(base_dir, args.output_dir)
        else:
            resolved_output_dir = args.output_dir
    else:
        resolved_output_dir = os.path.abspath(args.output_dir)
        base_dir = resolved_output_dir  # cache colocated when no explicit base

    os.makedirs(resolved_output_dir, exist_ok=True)

    if args.hf:
        if not args.file:
            p.error('--file is required with --hf')
        base_name = args.dest_name or os.path.basename(args.file)
        dest_abs = os.path.join(resolved_output_dir, base_name)
        if os.path.exists(dest_abs) and not os.path.islink(dest_abs):
            print(f"⚠️  Skip existing: {dest_abs}")
            return
        cache_dir = os.path.join(base_dir, 'hf-cache')
        os.makedirs(cache_dir, exist_ok=True)
        download_hf(args.hf, args.file, args.rev, dest_abs, cache_dir, args.dry_run)
    else:
        parsed = urlparse(args.url)
        name_from_url = os.path.basename(parsed.path)
        if not name_from_url:
            raise SystemExit('URL must end with a file name')
        base_name = args.dest_name or name_from_url
        dest_abs = os.path.join(resolved_output_dir, base_name)
        if os.path.exists(dest_abs) and not os.path.islink(dest_abs):
            print(f"⚠️  Skip existing: {dest_abs}")
            return
        download_url(args.url, dest_abs, args.dry_run)


if __name__ == '__main__':
    main()
