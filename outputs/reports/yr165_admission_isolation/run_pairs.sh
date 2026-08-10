#!/usr/bin/env bash
# YR-165 분리검증 — 8쌍(16런) + 판정(전건 산출 시에만).
set -u
ROOT=/mnt/c/Users/geonu/Desktop/port_reinforcement
LOG="$ROOT/outputs/reports/yr165_admission_isolation/pairs.log"
cd "$ROOT"
echo "ISO_START $(date -u '+%Y-%m-%d %H:%M:%S')" > "$LOG"
seq 0 7 | xargs -P 3 -I{} sh -c \
  "cd $ROOT && \$HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr165_admission_isolation --pair {} >> $LOG 2>&1"
N=$(ls "$ROOT"/outputs/reports/yr165_admission_isolation/iso_*.json 2>/dev/null | wc -l)
if [ "$N" -eq 8 ]; then
  "$HOME/.local/bin/uv" run --extra rl python -m yard_rl.experiments.yr165_admission_isolation --verdict >> "$LOG" 2>&1
  echo "ISO_END $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
else
  echo "ISO_ABORTED pairs=$N/8 — 판정 미실행(오류 조사 필요)" >> "$LOG"
fi
