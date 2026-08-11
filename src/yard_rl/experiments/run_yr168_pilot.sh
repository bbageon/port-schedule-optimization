#!/usr/bin/env bash
# YR-168 눈가림 파일럿 — 12쌍 병렬 + 합산(전건 산출 시에만).
set -u
ROOT=/mnt/c/Users/geonu/Desktop/port_reinforcement
DIR="$ROOT/outputs/reports/yr168_pilot"
mkdir -p "$DIR"
LOG="$DIR/pilot.log"
cd "$ROOT"
echo "PILOT_START $(date -u '+%Y-%m-%d %H:%M:%S')" > "$LOG"
seq 0 11 | xargs -P 6 -I{} sh -c \
  "cd $ROOT && \$HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr168_pilot --pair {} >> $LOG 2>&1"
N=$(ls "$DIR"/pair*.json 2>/dev/null | wc -l)
if [ "$N" -eq 12 ]; then
  echo "PAIRS_DONE 12/12" >> "$LOG"
  $HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr168_pilot \
    --summarize >> "$LOG" 2>&1
else
  echo "PILOT_ABORTED pairs=$N/12 — 합산 미실행(오류 조사 필요)" >> "$LOG"
fi
echo "PILOT_END $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
