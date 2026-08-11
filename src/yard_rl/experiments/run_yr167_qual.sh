#!/usr/bin/env bash
# YR-167 24시간 자격 — 3셀 병렬 + 합산(전건 산출 시에만).
set -u
ROOT=/mnt/c/Users/geonu/Desktop/port_reinforcement
DIR="$ROOT/outputs/reports/yr167_diurnal_qual"
mkdir -p "$DIR"
LOG="$DIR/qual.log"
cd "$ROOT"
echo "QUAL_START $(date -u '+%Y-%m-%d %H:%M:%S')" > "$LOG"
seq 0 2 | xargs -P 3 -I{} sh -c \
  "cd $ROOT && \$HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr167_diurnal_qual --rep {} >> $LOG 2>&1"
N=$(ls "$DIR"/cell_rep*.json 2>/dev/null | wc -l)
if [ "$N" -eq 3 ]; then
  echo "CELLS_DONE 3/3" >> "$LOG"
else
  echo "QUAL_ABORTED cells=$N/3 — 합산 미실행(오류 조사 필요)" >> "$LOG"
fi
echo "QUAL_END $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
