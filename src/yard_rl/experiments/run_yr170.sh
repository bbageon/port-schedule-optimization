#!/usr/bin/env bash
# YR-170 — 전건 KEEP 기준선 3시드 + 축소 예산 학습 3시드 (병렬).
# 에피소드 1회 약 10분(채택 실행 헤드 rollout) → 15 iter × 2 eps = 30 에피소드 ≈ 5시간/시드.
set -u
ROOT=/mnt/c/Users/geonu/Desktop/port_reinforcement
DIR="$ROOT/outputs/reports/yr170_sell_ppo_diurnal"
mkdir -p "$DIR"
LOG="$DIR/run.log"
cd "$ROOT"
N_ITER=${N_ITER:-15}
EPS=${EPS:-2}
echo "YR170_START $(date -u '+%Y-%m-%d %H:%M:%S') n_iter=$N_ITER eps=$EPS" > "$LOG"
# ① 기준선 (참조값 — 판정 아님)
seq 0 2 | xargs -P 3 -I{} sh -c \
  "cd $ROOT && \$HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr170_sell_ppo_diurnal --baseline --seed-idx {} >> $LOG 2>&1"
echo "BASELINE_DONE $(ls $DIR/baseline_s*.json 2>/dev/null | wc -l)/3" >> "$LOG"
# ② 학습
seq 0 2 | xargs -P 3 -I{} sh -c \
  "cd $ROOT && \$HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr170_sell_ppo_diurnal --seed-idx {} --n-iter $N_ITER --eps-per-iter $EPS >> $LOG 2>&1"
echo "TRAIN_DONE $(ls -d $DIR/ppo_s* 2>/dev/null | wc -l)/3" >> "$LOG"
echo "YR170_END $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
