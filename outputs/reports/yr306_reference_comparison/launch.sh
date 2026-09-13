#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT='/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매'
TASK_SOURCE="$TASK_ROOT/outputs/v5/yr306-reference-source"
export GIT_DIR='/mnt/c/Users/geonu/Desktop/port_reinforcement/.git/worktrees/yr306-reference-source'
export GIT_WORK_TREE="$TASK_SOURCE"
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$TASK_SOURCE"
test "$(git rev-parse HEAD)" = 'd38b2ecb1f2d7aa19f8a3b19c3b37c4c4552eb6e'
test ! -e "$TASK_ROOT/outputs/v5/yr306-reference-comparison"
/home/geonu/.venvs/yard-rl/bin/python -c 'from yard_rl.v5.ppo.provenance import code_stamp; print(code_stamp())'
exec taskset -c 23 /home/geonu/.venvs/yard-rl/bin/python -u scripts/v5/queue_reference_comparison.py \
  --workspace "$TASK_ROOT" --output "$TASK_ROOT/outputs/v5/yr306-reference-comparison"
