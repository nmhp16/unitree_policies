#!/usr/bin/env bash
# Convenience wrapper for play.py — see train_rl.sh for environment notes.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ISAACLAB=${ISAACLAB:-$HOME/IsaacLab}

cd "$PROJECT_ROOT"
exec env -u VIRTUAL_ENV -u CONDA_PREFIX \
    PYTHONPATH="$PROJECT_ROOT/..:${PYTHONPATH:-}" \
    "$ISAACLAB/isaaclab.sh" -p scripts/play.py "$@"
