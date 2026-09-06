"""30일 무대 학습 실행 진입점 ([[YR-239]]).

    PYTHONPATH=src python -m yard_rl.v4.train --seed 9900700 --workers 20

■ 왜 스크립트로 두나
  전에는 `python -c` 로 띄웠다. 그러면 **무엇으로 돌렸는지가 남지 않는다** — 시드·
  작업자 수·라벨 예산이 셸 히스토리에만 있고 원자료에는 없다. 여기 두면 실행 설정이
  코드로 박제되고 `--dry` 로 계획만 먼저 볼 수 있다.

■ 판정 대역은 못 쓴다
  `run_month_training` 이 진단 대역(9,900,0xx)이 아니면 즉시 거절한다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..stage.month import (N_DAYS, SHORT_LOADS, plan_days, plan_month,
                          summarize)
from .loop import DIAGNOSTIC_BASE, LABELS_PER_ITER
from .month_loop import explore_of, run_month_training


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="yard_rl.v4.train",
                                 description="30일 무대 학습 (하루 = 한 회차)")
    ap.add_argument("--seed", type=int, default=DIAGNOSTIC_BASE + 700,
                    help="달 시드 — 진단 대역(9,900,0xx)만 허용")
    ap.add_argument("--days", type=int, default=N_DAYS, help="달 길이 (기본 30)")
    ap.add_argument("--labels", type=int, default=LABELS_PER_ITER,
                    help="하루에 만들 반사실 라벨 수")
    ap.add_argument("--workers", type=int, default=-1,
                    help="반사실 세계를 나눌 프로세스 수 (-1 = 코어−2)")
    ap.add_argument("--out", default="outputs/v4/month", help="결과 폴더")
    ap.add_argument("--dry", action="store_true",
                    help="굴리지 않고 **달 계획만** 보여준다")
    ap.add_argument("--loads", default=None,
                    help="부하를 **직접 지정** — `short`(바닥 비교용 9일) 또는 "
                         "쉼표 목록(예 3500,5000,7500). 주면 --days 는 무시한다")
    a = ap.parse_args(argv)

    if a.loads:
        loads = (SHORT_LOADS if a.loads == "short"
                 else tuple(int(x) for x in a.loads.split(",") if x))
        days, a.days = plan_days(a.seed, loads), len(loads)
    else:
        days = plan_month(a.seed, n_days=a.days)
    s = summarize(days)
    print(f"■ 달 시드 {a.seed:,} · {a.days}일 (학습 {s['n_train']}일)")
    print(f"  {' · '.join(f'{k} {v}일' for k, v in s['by_label'].items())}")
    print(f"  학습분 트럭 {s['trucks_train']:,}대 · 평균 부하 {s['mean_load_train']:,.0f}")
    print("  날별 부하: " + " ".join(
        (f"[{d.load // 1000}]" if not d.is_train else str(d.load // 1000))
        for d in days) + "   ([]=연결용)")
    print(f"  ε {explore_of(days[0], n_days=len(days)):.2f} → "
          f"{explore_of(days[-1], n_days=len(days)):.2f} · 라벨 {a.labels}/일")
    if a.dry:
        return 0

    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_month_training(seed=a.seed, n_days=a.days, labels_per_day=a.labels,
                       out_dir=a.out, workers=a.workers, days=days)
    print(f"■ 총 {(time.time() - t0) / 3600:.2f}시간 · 결과 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
