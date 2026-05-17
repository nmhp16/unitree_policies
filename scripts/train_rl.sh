#!/usr/bin/env bash
# Convenience wrapper for train_rl.py — uses Isaac Lab's bundled python so the
# sim bindings resolve correctly. Any extra args are forwarded.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ISAACLAB=${ISAACLAB:-$HOME/IsaacLab}

cd "$PROJECT_ROOT"
# Unset the host venv/conda env so isaaclab.sh picks up Isaac Sim's bundled
# python rather than whatever shell environment we were called from.
exec env -u VIRTUAL_ENV -u CONDA_PREFIX \
    PYTHONPATH="$PROJECT_ROOT/..:${PYTHONPATH:-}" \
    "$ISAACLAB/isaaclab.sh" -p scripts/train_rl.py "$@"
