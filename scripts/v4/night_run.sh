#!/usr/bin/env bash
# YR-248 1단계 야간 실행 — 무대 검증 게이트 → 통과 시에만 학습.
#
# ■ 왜 게이트를 두나
#   무대(야드트랙터 왕복 180→470초)를 바꾼 직후다. 무대가 망가진 채로 18시간짜리
#   학습을 걸면 밤새 헛돈다. 그래서 **먼저 4일 프로브로 확인하고, 통과했을 때만**
#   학습으로 넘어간다. 실패하면 그 자리에서 멈추고 이유를 로그에 남긴다.
#
# ■ 게이트 (실행 전 고정 — 결과 보고 기준을 고르지 않는다)
#   ① 본선 유휴가 살아났나   c_vessel 비중이 v3(1.0%)보다 **뚜렷이 커야** 한다.
#                            문헌의 안벽 성능 손실 10~20% 를 목표로 보되, 비용 비중과
#                            시간 비중은 다른 양이므로 **하한 3%** 를 통과선으로 둔다.
#   ② 트럭이 안 무너졌나     90분위 체류시간이 v3 대비 **2배 미만**.
#                            트랙터가 느려지면 트럭도 느려지는데, 그게 감당 못 할
#                            수준이면 무대가 트럭 축까지 망가뜨린 것이다.
#
# ■ 통과 시 학습
#   재배정층(제안망·수락망)을 **새 무대에서 처음부터** 다시 배운다. 무대가 바뀌었으니
#   v3 체크포인트는 못 쓴다 — 배운 세계가 다르다.
#   ⚠️ 크레인 RL 은 아직 구현이 없다. 이번 학습은 재배정층이며, 크레인층은 규칙 그대로다.
set -u

ROOT="/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매"
PY="$HOME/.venvs/yard-rl/bin/python"
OUT="$ROOT/outputs/v4"
cd "$ROOT" || exit 1
mkdir -p "$OUT"

LOG="$OUT/night.log"
: > "$LOG"
say () { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

say "■ 1/2 무대 검증 — 야드트랙터 왕복 470초"
PYTHONPATH=src "$PY" scripts/v4/gate_stage_check.py > "$OUT/gate.log" 2>&1
GATE=$?
grep -v "cpu = \|UserWarning\|functional_tensor\|Failed to init" "$OUT/gate.log" | tee -a "$LOG"

if [ "$GATE" -ne 0 ]; then
  say "■ ✗ 게이트 실패 — 학습을 걸지 않는다. 위 사유를 보고 무대를 다시 정한다."
  exit 1
fi

# 시드 9,900,994 — 부하 5수준이 전부 있고 초혼잡 2일이 든 달이다(--dry 로 골랐다).
# 혼잡일이 얇으면 학습이 "재배치가 값어치 있는 지형" 을 한 번도 못 본다.
say "■ 2/2 게이트 통과 — 재배정층 30일 학습 시작 (시드 9,900,994 · 약 18시간)"
PYTHONPATH=src "$PY" -m yard_rl.v4.train \
    --seed 9900994 --days 30 --labels 64 --workers 6 \
    --out "$OUT/train-400s" >> "$OUT/train.log" 2>&1
TR=$?
tail -20 "$OUT/train.log" | grep -v "cpu = \|UserWarning\|functional_tensor" | tee -a "$LOG"
say "■ 학습 종료 (exit $TR) · 산출물 $OUT/train-400s"
