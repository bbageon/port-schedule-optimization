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

DATA="$ROOT/outputs/reports/yr171b_estimator"
if [ -f "$DATA/dataset_meta.json" ] && [ "${FORCE_DATA:-0}" != "1" ]; then
  say "STEP1 건너뜀 — 정답지가 이미 있다"
else
  say "STEP1 정답지 수집 ($N_DAYS 일)"
  $UV run --extra rl python -m yard_rl.experiments.yr171b_collect --n-days "$N_DAYS" >> "$LOG" 2>&1
  [ -f "$DATA/dataset_meta.json" ] || { say "CHAIN_ABORT — 정답지 수집 실패"; exit 1; }
fi

if [ -f "$DATA/buy_net.pt" ] && [ "${FORCE_BUY:-0}" != "1" ]; then
  say "STEP2 건너뜀 — BUY 견적망이 이미 있다"
else
  say "STEP2 BUY 견적망 학습 + 3종 검증"
  $UV run --extra rl python -m yard_rl.experiments.yr171b_train_buy >> "$LOG" 2>&1
  [ -f "$DATA/buy_net.pt" ] || { say "CHAIN_ABORT — BUY 견적망 학습 실패"; exit 1; }
fi

# ★3팔을 **동시에** 돌린다 — 3팔 × N_SEEDS 시드 × EPS 에피소드 = 24 동시(24코어).
# 팔을 하나씩 돌리면 코어의 1/3만 쓰고 벽시계가 3배가 된다(실측 후 정정).
say "STEP3 3팔 동시 학습 × $N_SEEDS 시드 (동시 에피소드 $((3 * N_SEEDS * EPS)))"
LAST=$((N_SEEDS - 1))
for ARM in fixed15 slots48 slots48_buy; do
  for S in $(seq 0 "$LAST"); do
    echo "$ARM $S"
  done
done | xargs -P $((3 * N_SEEDS)) -n 2 sh -c \
  'cd '"$ROOT"' && '"$UV"' run --extra rl python -m yard_rl.experiments.yr171c_train \
   --arm "$0" --seed-idx "$1" --n-iter '"$N_ITER"' --eps-per-iter '"$EPS"' >> '"$LOG"' 2>&1'
say "  학습 완료 ($(ls -d $DIR/*_s* 2>/dev/null | wc -l)/$((3 * N_SEEDS)))"

say "STEP4 평가 (같은 날 짝지어, 전건 KEEP 기준)"
$UV run --extra rl python -m yard_rl.experiments.yr171c_eval >> "$LOG" 2>&1
say "CHAIN_END"
