#!/usr/bin/env bash
# YR-324 단계 0 — 2×2 의 빈 칸을 채운다: **학습 전 가중치 + 추첨**.
#   이미 채워진 칸: 학습전+최고점(무너짐) · 학습후+최고점(무너짐) · 학습후+추첨(7일).
#   이 칸이 채워지면 "학습이 정책을 나쁘게 만들었나" 가 직접 갈린다.
#   YR-319 와 **같은 코드·같은 시드·같은 30일·같은 중단규칙**, 체크포인트만 initial.pt.
set -euo pipefail
TASK_ROOT='/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매'
TASK_SOURCE="$TASK_ROOT/outputs/v5/yr319-source-adaa5f2"
TASK_OUTPUT="$TASK_ROOT/outputs/v5/yr324-sampling-initial"
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
  --checkpoint initial \
  --output "$TASK_OUTPUT"
