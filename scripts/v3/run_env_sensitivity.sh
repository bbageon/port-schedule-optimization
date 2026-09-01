#!/usr/bin/env bash
# 혼잡 빈도 민감도 — 세 환경에서 공간·시간의 몫이 어떻게 갈리는지 잰다.
#
# ■ 왜
#   §5.4 의 분해가 두 대역 사이에서 뒤집혔다 (시각 91.3% ↔ 53.8% · 블록 −1.31억 ↔ +3.94억).
#   원인 후보는 **혼잡일 빈도**다 — 진단 대역은 4일, 판정 대역은 6일이었다.
#   그렇다면 빈도를 직접 움직여 재면 된다.
#
# ■ 설계
#   시드를 **셋 다 9,900,980 으로 같게** 둔다. `plan_days` 가 날짜 시드를
#   `seed + 1000*(i+1)` 로 만들므로, 같은 자리의 날은 세 환경에서 같은 난수를 쓴다.
#   달라지는 것은 **그 날의 부하 하나뿐**이다.
#
#   혼잡은 어느 환경에서나 `12,500 → 15,000` 이틀 파도로 통일했다. 파도의 모양은
#   같고 **빈도만** 2 / 6 / 14 일로 바뀐다. 어느 수준도 연달아 오지 않게 균등 위상으로
#   흩뿌렸다 — 30일은 세계가 하나라 월 초·말로 몰리면 그 자체가 처치가 된다.
#
# ■ 팔 다섯
#   RL(둘 다) · RL_SPACE(공간만) · RL_TIME(시간만) · RL_EARLY(1회차) · NO_REALLOC(기준)
#   앞 셋으로 분해가 나오고, RL_EARLY 로 학습의 몫이 환경별로 나온다.
#
# ⚠️ 진단 대역이다 — 논문 대표 수치가 아니라 **민감도 근거**로만 쓴다.
set -u

ROOT="/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매"
PY="$HOME/.venvs/yard-rl/bin/python"
SEED=9900980
CKPT="outputs/v3/month-02/ckpt_029.pt"
EARLY="outputs/v3/month-02/ckpt_000.pt"
ARMS="NO_REALLOC,RL_SPACE,RL_TIME"

QUIET="3500,3500,5000,7500,3500,5000,3500,5000,3500,7500,3500,5000,3500,5000,7500,12500,15000,3500,5000,3500,7500,3500,5000,3500,5000,3500,7500,5000,3500,3500"
MIXED="3500,3500,5000,7500,12500,15000,3500,5000,7500,3500,5000,7500,3500,5000,7500,12500,15000,3500,5000,7500,3500,5000,7500,3500,12500,15000,5000,7500,3500,3500"
HEAVY="3500,12500,15000,5000,7500,3500,12500,15000,5000,7500,12500,15000,3500,5000,7500,12500,15000,3500,12500,15000,5000,7500,12500,15000,3500,5000,7500,12500,15000,3500"

cd "$ROOT" || exit 1

run_env () {
  local key="$1" loads="$2"
  mkdir -p "outputs/v3/env-$key"
  PYTHONPATH=src "$PY" -m yard_rl.v3.eval --seed "$SEED" --ckpt "$CKPT" --ckpt-early "$EARLY" --arms "$ARMS" --loads "$loads" --workers 5 --out "outputs/v3/env-$key" > "outputs/v3/env-$key/run.log" 2>&1
  echo "[$key] 끝 (exit $?)"
}

echo "■ 세 환경 동시 실행 — 팔 5 × 환경 3 = 15 프로세스"
run_env quiet "$QUIET" &
run_env mixed "$MIXED" &
run_env heavy "$HEAVY" &
wait
echo "■ 셋 다 완료"
for k in quiet mixed heavy; do
  echo "--- $k ---"
  tail -3 "outputs/v3/env-$k/run.log" 2>/dev/null
done
