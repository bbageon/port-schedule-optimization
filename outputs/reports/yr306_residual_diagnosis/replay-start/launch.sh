#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT='/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매'
TASK_SOURCE="$TASK_ROOT/outputs/v5/yr306-residual-source"
TASK_OUTPUT="$TASK_ROOT/outputs/v5/yr306-residual-30d-5e3cdce"
export GIT_DIR='/mnt/c/Users/geonu/Desktop/port_reinforcement/.git/worktrees/yr306-residual-source'
export GIT_WORK_TREE="$TASK_SOURCE"
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$TASK_SOURCE"
test "$(git rev-parse HEAD)" = '5e3cdce606f0c1fa24077df2361516482320b466'
test ! -e "$TASK_OUTPUT"
exec taskset -c 23 /home/geonu/.venvs/yard-rl/bin/python -u scripts/v5/replay_residual_diagnosis.py \
  --original-run "$TASK_ROOT/outputs/v5/yr306-wait-30d-4549ae9" \
  --seed-bundle "$TASK_ROOT/outputs/reports/yr306_seed_regeneration/seed-9900306/seed-data.json.gz" \
  --output "$TASK_OUTPUT"
