#!/usr/bin/env bash
# YR-299 A — 재검토 창 W 민감도. *"도착 30분 전 재배정은 촉박하지 않나"* 에 답한다.
#
# ■ 물음의 정당성
#   공간 재배치는 반입 전용이라 옮길 물건이 없고 게이트 안내만 바뀌므로 안 촉박하다.
#   **시간 이연은 정당한 지적**이다 — 기사에게 30분 전에 "2시간 뒤에 오세요" 는 현실
#   예약제에서 무리다. 논문은 §5.4 에서 *"실행 가능한 일정이 아니라 잠재 범위"* 라고
#   한정했지만 **창 자체의 민감도는 잰 적이 없다.**
#
# ■ 1단계 = 평가만 (재학습 없음)
#   망은 30분 조건에서 배웠으므로 큰 W 에 **불리한 편향**이 있다. 그런데도 이득이
#   유지되면 그 자체로 강한 결과다. 급감하면 2단계(W 별 재학습)로 원인을 가른다.
#
# ■ 어느 결과든 논문에 보탬
#   유지 → *"1~2시간 여유를 주고도 실행 가능"* (실행성 주장 획득)
#   급감 → *"가치가 막판 정보에 있다"* — 실시간 정보 공유(스마트항만)의 가치를
#          정량화한 발견이 되고 서론의 서사와 직결된다.
#
# ⚠️ 진단 대역이다. 논문 본문에 실을 값이면 새 판정 대역에서 확증한다([[YR-296]] 절차).
set -u
ROOT="/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매"
PY="$HOME/.venvs/yard-rl/bin/python"
cd "$ROOT" || exit 1
mkdir -p outputs/v3/window-sweep

for W in 1800 3600 7200; do
  M=$((W / 60))
  echo "[$(date '+%H:%M')] ■ W=${M}분 시작"
  PYTHONPATH=src "$PY" - "$W" <<'PYEOF' >> "outputs/v3/window-sweep/w.log" 2>&1
import json, pathlib, sys
sys.path.insert(0, "src")
from yard_rl.v3.eval.month_judge import judge_month
from yard_rl.v3.stage.month import plan_month
from yard_rl.v3.actors import BuyerNet, SellerNet
import torch

W = float(sys.argv[1])
ck = torch.load("outputs/v3/month-02/ckpt_029.pt", map_location="cpu", weights_only=True)
s, b = SellerNet(), BuyerNet()
s.load_state_dict(ck["seller"]); b.load_state_dict(ck["buyer"])
out = pathlib.Path(f"outputs/v3/window-sweep/w{int(W)}")
out.mkdir(parents=True, exist_ok=True)
res = judge_month(seed=9_900_995, seller_net=s, buyer_net=b,
                  arms=("NO_REALLOC",), window_s=W,
                  days=plan_month(9_900_995, n_days=30),
                  workers=2, ckpt_dir=out / "arms")
(out / "judge.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                encoding="utf-8")
print(f"■ W={W/60:.0f}분 완료 → {out}")
PYEOF
  echo "[$(date '+%H:%M')] ■ W=${M}분 끝 (exit $?)"
done
echo "[$(date '+%H:%M')] ■ 세 창 모두 완료"
