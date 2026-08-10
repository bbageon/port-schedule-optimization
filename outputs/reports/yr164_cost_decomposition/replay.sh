#!/usr/bin/env bash
# YR-164 ② 진단 재생 드라이버 — 40건 세부 장부 수집 + 요약(전건 산출 시에만).
set -u
ROOT=/mnt/c/Users/geonu/Desktop/port_reinforcement
LOG="$ROOT/outputs/reports/yr164_cost_decomposition/replay.log"
cd "$ROOT"
echo "REPLAY_START $(date -u '+%Y-%m-%d %H:%M:%S')" > "$LOG"
seq 0 39 | xargs -P 3 -I{} sh -c \
  "cd $ROOT && \$HOME/.local/bin/uv run --extra rl python -m yard_rl.experiments.yr164_cost_decomposition --replay {} >> $LOG 2>&1"
N=$(ls "$ROOT"/outputs/reports/yr164_cost_decomposition/decomp_*.json 2>/dev/null | wc -l)
if [ "$N" -eq 40 ]; then
  "$HOME/.local/bin/uv" run --extra rl python -m yard_rl.experiments.yr164_cost_decomposition --summarize >> "$LOG" 2>&1
  echo "REPLAY_END $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
else
  echo "REPLAY_ABORTED decomp=$N/40 — 요약 미실행(오류 조사 필요)" >> "$LOG"
fi
