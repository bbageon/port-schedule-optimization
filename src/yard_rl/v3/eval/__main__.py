"""30일 무대 판정 실행 진입점 ([[YR-239]]).

    PYTHONPATH=src python -m yard_rl.v3.eval --seed 9400000 \\
        --ckpt outputs/v3/month/ckpt_028.pt --workers 7

■ 같은 달을 **팔만 바꿔** 다시 굴린다
  30일은 세계가 하나라 에피소드 안에서 기준선을 못 만든다. 그래서 달 전체를 팔마다
  한 번씩 굴리고 **날 단위로 짝비교**한다(같은 시드·같은 도착 명단·같은 본선).

■ 판정 대역 계약
  학습은 진단 대역(9,900,0xx)에서 돌고 **판정은 새 대역**에서 한 번만 한다
  ([[YR-210]] · 06 §3). 여기서 `--seed` 를 진단 대역으로 주면 경고한다 — 막지는
  않는다(진단 목적의 예행이 있을 수 있다). 논문 수치는 새 대역에서만 쓴다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..stage.month import SHORT_LOADS, plan_days
from .guards import DIAGNOSTIC_BAND
from .month_judge import JUDGE_ARMS, judge_month


def _load_nets(path: str | None):
    """체크포인트에서 학생 두 망을 되살린다. 없으면 무작위 초기화(진단용)."""
    from ..actors import BuyerNet, SellerNet

    s, b = SellerNet(), BuyerNet()
    if not path:
        return s, b, "무작위 초기화 (학습 전 — 진단용)"
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=True)
    s.load_state_dict(ck["seller"])
    b.load_state_dict(ck["buyer"])
    return s, b, f"{path} (회차 {ck.get('it', '?')})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="yard_rl.v3.eval",
                                 description="30일 무대 판정 (부하별 부호검정)")
    ap.add_argument("--seed", type=int, required=True, help="판정할 달의 시드")
    ap.add_argument("--ckpt", default=None, help="학생 체크포인트 .pt")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--arms", default=",".join(JUDGE_ARMS),
                    help="RL 과 겨룰 팔 (쉼표 구분)")
    ap.add_argument("--workers", type=int, default=0,
                    help="팔을 나눌 프로세스 수 (0 = 팔 수만큼)")
    ap.add_argument("--ckpt-early", default=None,
                    help="★학습 전 체크포인트 — 이것과 --ckpt 의 차이가 **순수 학습 효과**다 "
                         "(둘 다 ε=0 이라 탐색이 안 섞인다)")
    ap.add_argument("--loads", default=None,
                    help="부하를 직접 지정 — `short`(9일) 또는 쉼표 목록")
    ap.add_argument("--out", default="outputs/v3/month-judge")
    a = ap.parse_args(argv)

    if (a.seed // 100_000) * 100_000 == DIAGNOSTIC_BAND:
        print("⚠️ 진단 대역 시드다 — 여기 나온 수치는 **논문 주장이 아니다**")
    s_net, b_net, tag = _load_nets(a.ckpt)
    print(f"■ 정책: {tag}")
    extra = None
    if a.ckpt_early:
        es, eb, etag = _load_nets(a.ckpt_early)
        extra = {"RL_EARLY": (es, eb)}
        print(f"■ 학습 전 대조: {etag}")

    days = None
    if a.loads:
        ld = (SHORT_LOADS if a.loads == "short"
              else tuple(int(x) for x in a.loads.split(",") if x))
        days, a.days = plan_days(a.seed, ld), len(ld)
        print(f"■ 부하 지정 {a.days}일: " + " ".join(str(v // 1000) for v in ld))
    out = Path(a.out)
    t0 = time.time()
    res = judge_month(seed=a.seed, seller_net=s_net, buyer_net=b_net,
                      arms=tuple(x for x in a.arms.split(",") if x),
                      n_days=a.days, days=days, workers=a.workers,
                      extra_policies=extra, ckpt_dir=out / "arms")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"judge_{a.seed}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print(f"■ {(time.time() - t0) / 60:.1f}분 · 결과 {out}/judge_{a.seed}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
