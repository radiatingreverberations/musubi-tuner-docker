#!/bin/bash
set -e

# Activate virtual environment
source /comfyui/venv/bin/activate

# Drop into a shell
bash "$@"