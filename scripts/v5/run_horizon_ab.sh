#!/usr/bin/env bash
# YR-299 B — 반사실 창 3시간 vs 6시간. **병렬**로 굴려 벽시계를 아낀다.
#
# 창은 학습 라벨을 몇 시간 어치로 채점하느냐만 정한다. 두 실행의 무대·수요·시드·정책
# 구조가 전부 같고, 다른 것은 **채점 길이 하나**다.
#
# ⚠️ 사전등록은 실행 전에 커밋했다 (de12b15) — 결과를 보고 판정 기준을 고르지 않기 위해서다.
set -u
ROOT="/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매"
PY="$HOME/.venvs/yard-rl/bin/python"
SEED=9900996
cd "$ROOT" || exit 1
mkdir -p outputs/v5/horizon

run () {   # $1 = 창(시간)
  local h="$1" tag="h$1"
  echo "[$(date '+%H:%M')] ■ 창 ${h}시간 시작"
  PYTHONPATH=src "$PY" -m yard_rl.v5.train \
      --seed "$SEED" --days 30 --labels 64 --workers 5 \
      --horizon-h "$h" --out "outputs/v5/horizon/$tag" \
      > "outputs/v5/horizon/$tag.log" 2>&1
  echo "[$(date '+%H:%M')] ■ 창 ${h}시간 끝 (exit $?)"
}

echo "■ YR-299 B — 반사실 창 3h vs 6h · 시드 $SEED · 병렬"
run 3 &
run 6 &
wait
echo "[$(date '+%H:%M')] ■ 둘 다 완료"
for t in h3 h6; do
  echo "--- $t ---"
  grep -E "총 |검증|손실" "outputs/v5/horizon/$t.log" 2>/dev/null | tail -4
done
