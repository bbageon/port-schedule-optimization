#!/usr/bin/env bash
# Bounded local debugging only. Arguments override the short defaults.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
if [[ -f "$HOME/.venvs/yard-rl/bin/activate" ]]; then
    source "$HOME/.venvs/yard-rl/bin/activate"
fi
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m yard_rl.v5.ppo "$@"
