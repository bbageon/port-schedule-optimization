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
# 36차 감사 가드: 40쌍 전부 산출된 경우에만 최종 판정을 실행한다 — 오류로 빈
# 상태에서 판정이 자동 생성되는 것(코드 오류가 STOP 으로 둔갑) 방지.
N_PAIRS=$(ls "$ROOT"/outputs/reports/yr151_0b_counterfactual/pair_*.json 2>/dev/null | wc -l)
if [ "$N_PAIRS" -eq 40 ]; then
  "$HOME/.local/bin/uv" run --extra rl python -m yard_rl.experiments.yr151_0b_counterfactual --verdict >> "$LOG" 2>&1
  echo "SWEEP_END $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
else
  echo "SWEEP_ABORTED pairs=$N_PAIRS/40 — 판정 미실행 (오류 조사 필요)" >> "$LOG"
fi
