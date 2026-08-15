#!/usr/bin/env bash
# YR-171-A 재자격 — 하루 공개 예약 장부(day_plan_public) 계약으로 YR-167 자격 재취득.
# 산출은 outputs/reports/yr167_diurnal_qual_public/ (구 계약 증거를 덮지 않는다).
# 시드 3셀 + 시드0 반복셀(런 결정론) 병렬 → 전건 산출 시에만 합산.
set -u
ROOT=/mnt/c/Users/geonu/Desktop/port_reinforcement
DIR="$ROOT/outputs/reports/yr167_diurnal_qual_public"
mkdir -p "$DIR"
LOG="$DIR/requal.log"
cd "$ROOT"
echo "REQUAL_START $(date -u '+%Y-%m-%d %H:%M:%S') contract=day_plan_public" > "$LOG"
printf '0\n1\n2\nR\n' | xargs -P 4 -I{} sh -c \
  "cd $ROOT && case {} in R) A='--rep 0 --tag repeat';; *) A='--rep {}';; esac; \
   \$HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr167_diurnal_qual \
   \$A --day-plan-public >> $LOG 2>&1"
N=$(ls "$DIR"/cell_rep0.json "$DIR"/cell_rep1.json "$DIR"/cell_rep2.json \
       "$DIR"/cell_rep0_repeat.json 2>/dev/null | wc -l)
if [ "$N" -eq 4 ]; then
  echo "CELLS_DONE 4/4" >> "$LOG"
  $HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr167_diurnal_qual \
    --summarize --day-plan-public >> "$LOG" 2>&1
else
  echo "REQUAL_ABORTED cells=$N/4 — 합산 미실행(오류 조사 필요)" >> "$LOG"
fi
echo "REQUAL_END $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
