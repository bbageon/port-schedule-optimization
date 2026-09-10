#!/usr/bin/env bash
# A clean frozen worktree is required. WSL callers supply its GIT_DIR/GIT_WORK_TREE.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
if [[ -f "$HOME/.venvs/yard-rl/bin/activate" ]]; then
    source "$HOME/.venvs/yard-rl/bin/activate"
fi
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u -m yard_rl.v5.ppo.continuous "$@"
