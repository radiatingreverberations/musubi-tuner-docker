#!/bin/bash
set -e

# Activate virtual environment
source venv/bin/activate

# Drop into a shell
bash "$@"