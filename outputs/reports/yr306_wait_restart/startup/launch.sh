#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT='/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매'
TASK_SOURCE="$TASK_ROOT/outputs/v5/yr306-wait-source-4549ae9"
TASK_OUTPUT="$TASK_ROOT/outputs/v5/yr306-wait-30d-4549ae9"
export GIT_DIR='/mnt/c/Users/geonu/Desktop/port_reinforcement/.git/worktrees/yr306-wait-source-4549ae9'
export GIT_WORK_TREE="$TASK_SOURCE"
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$TASK_SOURCE"
test "$(git rev-parse HEAD)" = '4549ae97db4aad834cad02698fad8c8b8c880f99'
test ! -e "$TASK_OUTPUT"
exec taskset -c 23 /home/geonu/.venvs/yard-rl/bin/python -u -m yard_rl.v5.ppo.continuous \
  --seed 9900306 --days 30 \
  --seed-bundle "$TASK_ROOT/outputs/reports/yr306_seed_regeneration/seed-9900306/seed-data.json.gz" \
  --seed-sha256 efdb45e2ec8feafed6987cf48dfe022ca2959dad8c4f1d56bee680456081af7a \
  --output "$TASK_OUTPUT"
