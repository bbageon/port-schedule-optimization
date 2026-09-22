#!/usr/bin/env bash
# YR-319 — 가중치는 얼린 채 행동만 추첨으로 고르는 30일 1회.
#   YR-306 의 final 팔과 **같은** 체크포인트·시드·30일 명단·화물·측정창을 쓰고
#   고르는 방식만 argmax → 추첨으로 되돌린다. 사전 중단규칙은 스크립트에 박혀 있다.
set -euo pipefail
TASK_ROOT='/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매'
TASK_SOURCE="$TASK_ROOT/outputs/v5/yr319-source-adaa5f2"
TASK_OUTPUT="$TASK_ROOT/outputs/v5/yr319-sampling-final"
ORIGINAL="$TASK_ROOT/outputs/v5/yr306-wait-30d-4549ae9"
RULE="$TASK_ROOT/outputs/v5/yr306-reference-comparison/rule/month_result.json"
export GIT_DIR='/mnt/c/Users/geonu/Desktop/port_reinforcement/.git/worktrees/yr319-source-adaa5f2'
export GIT_WORK_TREE="$TASK_SOURCE"
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$TASK_SOURCE"
test "$(git rev-parse HEAD)" = 'adaa5f25fbb3149c184007356010be2e54a51e5a'
test ! -e "$TASK_OUTPUT"
exec taskset -c 23 /home/geonu/.venvs/yard-rl/bin/python -u scripts/v5/eval_sampling.py \
  --original-run "$ORIGINAL" \
  --seed-bundle "$TASK_ROOT/outputs/reports/yr306_seed_regeneration/seed-9900306/seed-data.json.gz" \
  --rule-month-result "$RULE" \
  --checkpoint final \
  --output "$TASK_OUTPUT"
