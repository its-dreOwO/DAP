#!/usr/bin/env bash
# DAP391m Project 8 — environment setup
# Usage: bash install.sh
# Creates app/.venv and installs all project dependencies.

set -e

VENV_DIR="app/.venv"
PYTHON="python3"

# ── 1. python check ──────────────────────────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found Python $PY_VERSION"

# ── 2. create venv ───────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "Virtualenv already exists at $VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"

# ── 3. upgrade pip ───────────────────────────────────────────────────────────
"$PIP" install --upgrade pip --quiet

# ── 4. install dependencies ──────────────────────────────────────────────────
echo "Installing dependencies from requirements.txt ..."
"$PIP" install -r requirements.txt

# ── 5. done ──────────────────────────────────────────────────────────────────
echo ""
echo "Setup complete."
echo ""
echo "To activate the environment:"
echo "  bash/zsh:  source $VENV_DIR/bin/activate"
echo "  fish:      source $VENV_DIR/bin/activate.fish"
echo ""
echo "To run the pipeline:"
echo "  $VENV_DIR/bin/python3 src-code/01_ingestion_cleaning.py"
echo ""
echo "To launch the Streamlit app:"
echo "  $VENV_DIR/bin/streamlit run app/app.py"
