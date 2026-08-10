#!/usr/bin/env bash
# YR-151 0B 반사실 짝 스윕 드라이버 — 40쌍(80런) + 합산 판정.
# 로그: pairs.log (절대경로). 결과 파일에는 코드 커밋·표본 해시 스탬프가 박힌다.
set -u
ROOT=/mnt/c/Users/geonu/Desktop/port_reinforcement
LOG="$ROOT/outputs/reports/yr151_0b_counterfactual/pairs.log"
cd "$ROOT"
echo "SWEEP_START $(date -u '+%Y-%m-%d %H:%M:%S')" > "$LOG"
seq 0 39 | xargs -P 3 -I{} sh -c \
  "cd $ROOT && \$HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr151_0b_counterfactual --pair {} >> $LOG 2>&1"
"$HOME/.local/bin/uv" run --extra rl python -m yard_rl.experiments.yr151_0b_counterfactual --verdict >> "$LOG" 2>&1
echo "SWEEP_END $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
