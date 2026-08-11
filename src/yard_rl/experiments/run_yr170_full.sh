#!/usr/bin/env bash
# YR-170 전체 예산 학습 — 40 iter x 4 eps x 3 시드, iteration 내 에피소드 병렬.
# 진행 상황은 각 ppo_s*/train.json 이 iteration 마다 갱신된다.
set -u
ROOT=/mnt/c/Users/geonu/Desktop/port_reinforcement
DIR="$ROOT/outputs/reports/yr170_sell_ppo_diurnal"
mkdir -p "$DIR"
LOG="$DIR/train_full.log"
cd "$ROOT"
N_ITER=${N_ITER:-40}
EPS=${EPS:-4}
echo "FULL_START $(date -u '+%Y-%m-%d %H:%M:%S') n_iter=$N_ITER eps=$EPS seeds=3" > "$LOG"
seq 0 2 | xargs -P 3 -I{} sh -c \
  "cd $ROOT && \$HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr170_sell_ppo_diurnal --seed-idx {} --n-iter $N_ITER --eps-per-iter $EPS --parallel >> $LOG 2>&1"
echo "FULL_DONE $(ls -d $DIR/ppo_s* 2>/dev/null | wc -l)/3" >> "$LOG"
echo "FULL_END $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
