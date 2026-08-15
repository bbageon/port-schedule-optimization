#!/usr/bin/env bash
# YR-171-B/C 전체 체인 — 정답지 수집 → BUY 견적망 학습·검증 → 3팔 학습 → 평가.
# 무인 실행. 앞 단계가 실패하면 뒤를 돌리지 않는다(잘못된 입력으로 학습 금지).
set -u
ROOT=/mnt/c/Users/geonu/Desktop/port_reinforcement
cd "$ROOT"
UV=$HOME/.local/bin/uv
DIR="$ROOT/outputs/reports/yr171c_slots"
mkdir -p "$DIR" "$ROOT/outputs/reports/yr171b_estimator"
LOG="$DIR/chain.log"
N_ITER=${N_ITER:-10}
EPS=${EPS:-4}
N_SEEDS=${N_SEEDS:-2}
N_DAYS=${N_DAYS:-8}
say() { echo "$(date -u '+%H:%M:%S') $*" >> "$LOG"; }

echo "CHAIN_START $(date -u '+%Y-%m-%d %H:%M:%S') iter=$N_ITER eps=$EPS seeds=$N_SEEDS" > "$LOG"

say "STEP1 정답지 수집 ($N_DAYS 일)"
$UV run --extra rl python -m yard_rl.experiments.yr171b_collect --n-days "$N_DAYS" >> "$LOG" 2>&1
if [ ! -f "$ROOT/outputs/reports/yr171b_estimator/dataset_meta.json" ]; then
  say "CHAIN_ABORT — 정답지 수집 실패"; exit 1
fi

say "STEP2 BUY 견적망 학습 + 3종 검증"
$UV run --extra rl python -m yard_rl.experiments.yr171b_train_buy >> "$LOG" 2>&1
if [ ! -f "$ROOT/outputs/reports/yr171b_estimator/buy_net.pt" ]; then
  say "CHAIN_ABORT — BUY 견적망 학습 실패"; exit 1
fi

say "STEP3 3팔 학습 (fixed15 / slots48 / slots48_buy) × $N_SEEDS 시드"
LAST=$((N_SEEDS - 1))
for ARM in fixed15 slots48 slots48_buy; do
  seq 0 "$LAST" | xargs -P "$N_SEEDS" -I{} sh -c \
    "cd $ROOT && $UV run --extra rl python -m yard_rl.experiments.yr171c_train \
     --arm $ARM --seed-idx {} --n-iter $N_ITER --eps-per-iter $EPS >> $LOG 2>&1"
  say "  $ARM 완료 ($(ls -d $DIR/${ARM}_s* 2>/dev/null | wc -l)/$N_SEEDS)"
done

say "STEP4 평가 (같은 날 짝지어, 전건 KEEP 기준)"
$UV run --extra rl python -m yard_rl.experiments.yr171c_eval >> "$LOG" 2>&1
say "CHAIN_END"
