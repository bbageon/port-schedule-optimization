#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT='/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매'
TASK_SOURCE="$TASK_ROOT/outputs/v5/yr306-wait-source-4549ae9"
TASK_OUTPUT="$TASK_ROOT/outputs/v5/yr306-wait-smoke-4549ae9"
export GIT_DIR='/mnt/c/Users/geonu/Desktop/port_reinforcement/.git/worktrees/yr306-wait-source-4549ae9'
export GIT_WORK_TREE="$TASK_SOURCE"
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$TASK_SOURCE"
test "$(git rev-parse HEAD)" = '4549ae97db4aad834cad02698fad8c8b8c880f99'
test ! -e "$TASK_OUTPUT"
exec taskset -c 23 /home/geonu/.venvs/yard-rl/bin/python -u -m yard_rl.v5.ppo.continuous \
  --seed 9900306 --days 3 --debug-load 60 \
  --seed-bundle "$TASK_ROOT/outputs/v5/yr306-cargo-smoke-seed-83de18d/seed-data.json.gz" \
  --seed-sha256 2c8e68db5aad0d2dc49d8c4b503f7c014d598046328421679885a3af73ef1f6b \
  --output "$TASK_OUTPUT"
