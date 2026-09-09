#!/usr/bin/env bash
# YR-299 B 판정 — 창 3h/6h 로 배운 두 정책을 **같은 달**에서 비용으로 겨룬다.
#
# ■ 왜 이게 필요한가 (사용자 지적 2026-09-09)
#   검증 손실은 *"라벨을 잘 맞추나"* 이고, 우리가 알고 싶은 것은 *"비용이 줄었나"* 다.
#   학습 로그의 Φ 는 탐색(ε 0.50→0.05)이 섞여 있어 성능 비교로 못 쓴다.
#   여기서는 **탐색 0** 으로 둘을 나란히 굴려 날 단위 짝비교한다.
#
# ⚠️ 진단 대역이다. 판정 대역은 [[YR-233]]·[[YR-296]] 이 이미 썼고, 이건 v4 내부 비교다.
set -u
ROOT="/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매"
PY="$HOME/.venvs/yard-rl/bin/python"
SEED=9900997          # 학습 시드(9,900,996)와 다른 달 — 배운 달에서 재면 안 된다
cd "$ROOT" || exit 1
mkdir -p outputs/v5/horizon-judge

run () {   # $1 = h3|h6
  local tag="$1"
  echo "[$(date '+%H:%M')] ■ $tag 판정 시작"
  PYTHONPATH=src "$PY" -m yard_rl.v5.eval \
      --seed "$SEED" --ckpt "outputs/v5/horizon/$tag/ckpt_029.pt" \
      --arms NO_REALLOC --workers 2 \
      --out "outputs/v5/horizon-judge/$tag" \
      > "outputs/v5/horizon-judge/$tag.log" 2>&1
  echo "[$(date '+%H:%M')] ■ $tag 끝 (exit $?)"
}

echo "■ YR-299 B 판정 — 창 3h vs 6h · 시드 $SEED (학습과 다른 달) · 탐색 0"
run h3 &
run h6 &
wait
echo "[$(date '+%H:%M')] ■ 둘 다 완료"
