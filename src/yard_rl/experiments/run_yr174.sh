#!/usr/bin/env bash
# YR-174 — 거래별 실현 손익(차이 보상) 학습. 시드 N × 에피소드 EPS 동시(24코어 = 6×4).
# 대조군은 YR-170 의 같은 시드·같은 회차(전역 보상) 결과다.
set -u
ROOT=/mnt/c/Users/geonu/Desktop/port_reinforcement
DIR="$ROOT/outputs/reports/yr174_txn_reward"
mkdir -p "$DIR"
cd "$ROOT"
N_ITER=${N_ITER:-10}
EPS=${EPS:-4}
N_SEEDS=${N_SEEDS:-6}
LOG="$DIR/train.log"
LAST=$((N_SEEDS - 1))
echo "YR174_START $(date -u '+%Y-%m-%d %H:%M:%S') n_iter=$N_ITER eps=$EPS seeds=$N_SEEDS concurrent=$((N_SEEDS * EPS))" > "$LOG"
seq 0 "$LAST" | xargs -P "$N_SEEDS" -I{} sh -c \
  "cd $ROOT && \$HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr174_train --seed-idx {} --n-iter $N_ITER --eps-per-iter $EPS >> $LOG 2>&1"
echo "YR174_DONE $(ls -d $DIR/ppo_s* 2>/dev/null | wc -l)/$N_SEEDS" >> "$LOG"
echo "YR174_END $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
