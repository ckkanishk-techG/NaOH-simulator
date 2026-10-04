#!/usr/bin/env sh
# One-click laptop install (macOS/Linux). Creates a venv and installs HYDRA offline-capable.
set -e
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
echo "Done. Run: .venv/bin/hydra-sim"
