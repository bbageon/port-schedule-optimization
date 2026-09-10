"""크레인 정책 학습 진입점 ([[YR-308]]).

    PYTHONPATH=src python -m yard_rl.v4.crane --iters 20 --workers 20

■ 왜 스크립트로 두나
  `python -c` 로 띄우면 **무엇으로 돌렸는지가 남지 않는다** — 시드·작업자 수·
  라벨 예산이 셸 히스토리에만 있고 원자료에는 없다. 여기 두면 실행 설정이 코드로
  박제되고 `--dry` 로 계획만 먼저 볼 수 있다 (재배정층 `train/__main__.py` 와 같다).

■ 판정 대역은 못 쓴다
  학습은 진단 대역(9,900,0xx)에서만 돈다 — 여기 Φ 는 진단이지 주장이 아니다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..eval import TRAIN_LOADS
from .train import DIAGNOSTIC_BASE, LABELS_PER_ITER, run_crane_training


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="yard_rl.v4.crane",
        description="크레인 정책 학습 — 규칙 바닥(SF_SPT) 대비")
    ap.add_argument("--iters", type=int, default=20, help="회차 수 (하루 = 한 회차)")
    ap.add_argument("--seed", type=int, default=DIAGNOSTIC_BASE + 700,
                    help="시드 바닥 — 진단 대역(9,900,0xx)만 쓴다")
    ap.add_argument("--labels", type=int, default=LABELS_PER_ITER,
                    help="회차당 반사실 라벨 수 (재배정층과 같은 64)")
    ap.add_argument("--workers", type=int, default=-1,
                    help="반사실 세계를 나눌 프로세스 수 (-1 = 코어−2)")
    ap.add_argument("--horizon-h", type=float, default=3.0,
                    help="반사실 창(시간) — 라벨을 채점하는 길이")
    ap.add_argument("--loads", default=None,
                    help="부하 목록 (예 3500,5000,7500). 안 주면 학습 기본 부하")
    ap.add_argument("--hours", type=float, default=None,
                    help="시간 예산(시간) — 이 시간을 넘기면 회차를 더 안 연다")
    ap.add_argument("--eval-every", type=int, default=5,
                    help="몇 회차마다 **고정 평가일**로 짝비교할까 (0 = 안 함). "
                         "학습 전에도 한 번 재서 기준점을 남긴다")
    ap.add_argument("--out", default="outputs/v4/crane-train", help="결과 폴더")
    ap.add_argument("--dry", action="store_true", help="굴리지 않고 계획만 본다")
    a = ap.parse_args(argv)

    if a.seed < DIAGNOSTIC_BASE:
        ap.error(f"판정 대역 시드({a.seed:,})로는 학습을 못 돌린다 — "
                 f"진단 대역은 {DIAGNOSTIC_BASE:,} 이상이다")
    loads = (tuple(int(x) for x in a.loads.split(",") if x) if a.loads
             else TRAIN_LOADS)
    workers = a.workers
    print(f"■ 크레인 학습 — {a.iters}회차 · 시드 바닥 {a.seed:,}")
    print(f"  부하 {loads} · 라벨 {a.labels}/회차 · 반사실 창 {a.horizon_h:.0f}시간")
    print(f"  작업자 {workers} · 결과 {a.out}")
    print(f"  기준선: 같은 시드 SF_SPT (규칙 크레인) · 재배정은 끈다(NO_REALLOC)")
    if a.eval_every > 0:
        from .train import EVAL_LOADS, EVAL_SEED_BASE
        print(f"  고정 평가일: 부하 {EVAL_LOADS} · 시드 {EVAL_SEED_BASE:,}대 · "
              f"{a.eval_every}회차마다 (학습 전 1회 포함) — **같은 날 짝비교**")
    if a.dry:
        return 0

    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    st = run_crane_training(
        iters=a.iters, out_dir=a.out, labels_per_iter=a.labels,
        seed_base=a.seed, loads=loads, workers=workers,
        horizon_s=a.horizon_h * 3600.0, eval_every=a.eval_every,
        time_budget_s=(a.hours * 3600.0 if a.hours else None))
    secs = time.time() - t0
    print(f"■ 끝 — {len(st.history)}회차 · {secs/3600:.2f}시간")
    if st.history:
        last = st.history[-1]
        print(f"  마지막 회차 격차 {last.gap:+,.0f}원 ({last.gap_ratio:+.2%}) "
              f"· 검증손실 {last.val_loss:.5f}(정규 {last.val_loss_norm:.3f}) "
              f"· 눈금 {last.scale}")
    if st.evals:
        first, last_e = st.evals[0], st.evals[-1]
        print(f"  ★고정 평가일 (같은 날 짝비교 · 음수 = RL 이 싸다)")
        print(f"    학습 전 중앙 {first['median_gap_ratio']:+.2%} "
              f"(이긴 날 {first['n_win']}/{len(first['rows'])})")
        print(f"    학습 후 중앙 {last_e['median_gap_ratio']:+.2%} "
              f"(이긴 날 {last_e['n_win']}/{len(last_e['rows'])})")
    (Path(a.out) / "run.json").write_text(
        json.dumps({"iters": a.iters, "seed": a.seed, "labels": a.labels,
                    "loads": list(loads), "horizon_h": a.horizon_h,
                    "workers": workers, "secs": secs}, ensure_ascii=False,
                   indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
