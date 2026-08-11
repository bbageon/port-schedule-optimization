#!/usr/bin/env bash
# YR-167 24시간 재자격 — 시드 3셀 + 시드0 반복셀(런 결정론) 병렬 + 합산(전건 산출 시에만).
set -u
ROOT=/mnt/c/Users/geonu/Desktop/port_reinforcement
DIR="$ROOT/outputs/reports/yr167_diurnal_qual"
mkdir -p "$DIR"
LOG="$DIR/qual.log"
cd "$ROOT"
echo "QUAL_START $(date -u '+%Y-%m-%d %H:%M:%S')" > "$LOG"
printf '0\n1\n2\nR\n' | xargs -P 4 -I{} sh -c \
  "cd $ROOT && case {} in R) A='--rep 0 --tag repeat';; *) A='--rep {}';; esac; \
   \$HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr167_diurnal_qual \$A >> $LOG 2>&1"
N=$(ls "$DIR"/cell_rep0.json "$DIR"/cell_rep1.json "$DIR"/cell_rep2.json \
       "$DIR"/cell_rep0_repeat.json 2>/dev/null | wc -l)
if [ "$N" -eq 4 ]; then
  echo "CELLS_DONE 4/4" >> "$LOG"
  $HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr167_diurnal_qual \
    --summarize >> "$LOG" 2>&1
else
  echo "QUAL_ABORTED cells=$N/4 — 합산 미실행(오류 조사 필요)" >> "$LOG"
fi
echo "QUAL_END $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
