#!/usr/bin/env bash
# YR-170 학습 드라이버 — iteration 내 에피소드 병렬 + 시드 병렬.
#
# ★코어 활용 (2026-08-13 측정): 에피소드 1개 = 1 코어(이산사건 시뮬레이션은 블록을
#   한 번에 하나씩 순차 전진시킨다 — multiblock.py:192). 24코어 환경에서
#   시드 N × 에피소드 EPS = N*EPS 동시 실행이 되므로 N*EPS <= 24 로 맞춘다.
#   기본 N=3·EPS=4 는 12코어만 쓴다 → N_SEEDS=6 으로 24코어 전량 사용.
#
# 환경변수: N_ITER(기본 40) EPS(기본 4) N_SEEDS(기본 3) OUT_TAG(기본 없음)
set -u
ROOT=/mnt/c/Users/geonu/Desktop/port_reinforcement
DIR="$ROOT/outputs/reports/yr170_sell_ppo_diurnal"
mkdir -p "$DIR"
cd "$ROOT"
N_ITER=${N_ITER:-40}
EPS=${EPS:-4}
N_SEEDS=${N_SEEDS:-3}
LOG="$DIR/train_full.log"
LAST=$((N_SEEDS - 1))
echo "FULL_START $(date -u '+%Y-%m-%d %H:%M:%S') n_iter=$N_ITER eps=$EPS seeds=$N_SEEDS concurrent=$((N_SEEDS * EPS))" > "$LOG"
seq 0 "$LAST" | xargs -P "$N_SEEDS" -I{} sh -c \
  "cd $ROOT && \$HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr170_sell_ppo_diurnal --seed-idx {} --n-iter $N_ITER --eps-per-iter $EPS --parallel >> $LOG 2>&1"
echo "FULL_DONE $(ls -d $DIR/ppo_s* 2>/dev/null | wc -l)/$N_SEEDS" >> "$LOG"
echo "FULL_END $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
