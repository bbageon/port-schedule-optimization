#!/usr/bin/env bash
# YR-299 A — 재검토 창 W 하나만 돌린다 (병렬 실행용).
#
#     bash scripts/v3/run_window_one.sh 7200
#
# `run_window_sweep.sh` 는 창을 **순차로** 돈다. 24코어에서 창 하나가 팔 2개만
# 쓰므로 나머지가 논다. 남은 코어에 다른 창을 겹쳐 띄우려고 쪼갠 것이다.
# 같은 시드·같은 달·같은 정책이라 결과는 순차 실행과 동일하다.
set -u
W="${1:?창 길이를 초로 주세요 (예: 7200)}"
ROOT="/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매"
cd "$ROOT" || exit 1
mkdir -p outputs/v3/window-sweep

echo "[$(date '+%H:%M')] ■ W=$((W/60))분 시작 (단독)"
PYTHONPATH=src "$HOME/.venvs/yard-rl/bin/python" - "$W" <<'PYEOF' >> "outputs/v3/window-sweep/w${W}.log" 2>&1
import json, pathlib, sys
sys.path.insert(0, "src")
import torch
from yard_rl.v3.actors import BuyerNet, SellerNet
from yard_rl.v3.eval.month_judge import judge_month
from yard_rl.v3.stage.month import plan_month

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
(out / "judge.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(f"■ W={W/60:.0f}분 완료 → {out}")
PYEOF
echo "[$(date '+%H:%M')] ■ W=$((W/60))분 끝 (exit $?)"
