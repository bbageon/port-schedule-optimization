"""크레인 한 결정의 후보 목록에 본선 작업과 트럭 작업이 **같이** 들어오는가 — 실측.

    PYTHONPATH=src python scripts/v5/probe_crane_candidate_mix.py [--decisions N] [--loads 3500,12500]

배경: "크레인층을 간소화해 본선/트럭 중 하나를 고르게 하자" 는 제안의 가치는
**둘이 동시에 고를 수 있는 결정의 비율**에 달려 있다. 하나뿐이면 고를 게 없다.

재는 것 (크레인 1대의 1결정 = 표본 1):
  · 후보 개수(WAIT 포함) 분포
  · 실행가능 실작업(SERVE) 후보 수 — 0 / 1 / 2개 이상
  · 그 안에 본선(VESSEL_*)·트럭(GATE_*)이 **둘 다** 있는가
측정 지점은 `CandidateGenerator.generate` 반환값 = 정책이 실제로 보는 목록이다
(형제 크레인 공동 제약 이전 값이라 **상한**이다 — 보고 시 명시).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

sys.path.insert(0, "src")

import torch                                                       # noqa: E402

from yard_rl.v5.stage.month import plan_days                       # noqa: E402
from yard_rl.v5.stage.month_run import run_month                   # noqa: E402
from yard_rl.v5.world.contract.schema import CandidateKind         # noqa: E402
from yard_rl.v5.world.integrated.candidates import CandidateGenerator  # noqa: E402


class Enough(Exception):
    """표본을 다 모았다 — 엔진 오류가 아니다."""


class Tally:
    def __init__(self, limit: int, wall_s: float):
        self.limit, self.wall_s, self.t0 = limit, wall_s, time.perf_counter()
        self.n = 0
        self.n_items = Counter()        # 후보 개수(WAIT 포함)
        self.n_serve = Counter()        # 실행가능 SERVE 개수
        self.mix = Counter()            # NONE / VESSEL_ONLY / TRUCK_ONLY / BOTH
        self.kinds = Counter()          # 실행가능 후보의 종류
        self.serve_vessel = 0
        self.serve_truck = 0
        self.by_day = {}                # 날짜 → Counter(mix)  (야드가 찬 뒤를 따로 보려고)
        self.raw_mix = Counter()        # 가지치기(k_max=12) **전** 원본 SERVE 목록 기준
        self.raw = Counter()            # 원본 합계 (본선·트럭·필수·실행가능필수)

    def add_raw(self, gcs):
        v = t = 0
        for gc in gcs:
            if not gc.feasible or gc.job_ref is None:
                continue
            if gc.job_ref.is_vessel:
                v += 1
            elif gc.job_ref.is_external:
                t += 1
        self.raw_mix[("NONE", "TRUCK_ONLY", "VESSEL_ONLY", "BOTH")[(v > 0) * 2 + (t > 0)]] += 1
        self.raw["serve_vessel"] += v
        self.raw["serve_truck"] += t
        m = [g for g in gcs if g.mandatory]
        self.raw["mandatory"] += len(m)
        self.raw["mandatory_feasible"] += sum(1 for g in m if g.feasible)
        self.raw["decisions_mandatory_over_budget"] += len(m) >= 11
        self.raw["decisions_with_feasible_vessel"] += v > 0

    def add(self, gcs, now=0.0):
        self.n += 1
        day = int(now // 86400)
        self.n_items[len(gcs)] += 1
        v = t = 0
        for gc in gcs:
            if not gc.feasible:
                continue
            self.kinds[gc.kind.name] += 1
            if gc.kind is not CandidateKind.SERVE or gc.job_ref is None:
                continue
            if gc.job_ref.is_vessel:
                v += 1
            elif gc.job_ref.is_external:
                t += 1
        self.serve_vessel += v
        self.serve_truck += t
        self.n_serve[min(v + t, 10)] += 1
        label = ("NONE", "TRUCK_ONLY", "VESSEL_ONLY", "BOTH")[(v > 0) * 2 + (t > 0)]
        self.mix[label] += 1
        d = self.by_day.setdefault(day, Counter())
        d[label] += 1
        d["serve_ge2"] += (v + t) >= 2
        d["n"] += 1
        if self.n >= self.limit or time.perf_counter() - self.t0 > self.wall_s:
            raise Enough()

    def report(self) -> dict:
        counts = sorted(self.n_items.elements())
        return dict(decisions=self.n,
                    n_items_median=counts[len(counts) // 2] if counts else 0,
                    n_items_max=max(self.n_items) if self.n_items else 0,
                    n_items_hist=dict(sorted(self.n_items.items())),
                    feasible_serve_hist=dict(sorted(self.n_serve.items())),
                    serve_ge2_pct=100.0 * sum(c for k, c in self.n_serve.items() if k >= 2) / max(self.n, 1),
                    mix={k: dict(n=c, pct=100.0 * c / max(self.n, 1))
                         for k, c in sorted(self.mix.items())},
                    feasible_kinds=dict(self.kinds),
                    feasible_serve_vessel=self.serve_vessel,
                    feasible_serve_truck=self.serve_truck,
                    by_day={k: dict(v) for k, v in sorted(self.by_day.items())},
                    raw_totals=dict(self.raw),
                    raw_mix={k: dict(n=c, pct=100.0 * c / max(sum(self.raw_mix.values()), 1))
                             for k, c in sorted(self.raw_mix.items())},
                    wall_s=round(time.perf_counter() - self.t0, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=int, default=60_000)
    ap.add_argument("--wall", type=float, default=240.0)
    ap.add_argument("--loads", default="3500,12500")
    ap.add_argument("--seed", type=int, default=9_900_306)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_num_threads(1)

    loads = tuple(int(x) for x in args.loads.split(","))
    tally = Tally(args.decisions, args.wall)
    original = CandidateGenerator.generate

    def generate(self, sim, crane_id, level):
        out = original(self, sim, crane_id, level)
        tally.add(out.items, sim.now)
        return out

    raw_serve = CandidateGenerator._serve

    def _serve(self, sim, cid, now):
        out = raw_serve(self, sim, cid, now)
        tally.add_raw(out)
        return out

    CandidateGenerator.generate = generate
    CandidateGenerator._serve = _serve
    try:
        run_month(seed=args.seed, arm="NO_REALLOC", dispatcher="SF_SPT",
                  n_days=len(loads), days=plan_days(args.seed, loads))
        done = "달을 끝까지 굴렸다"
    except Enough:
        done = "표본 한도에서 중단"
    finally:
        CandidateGenerator.generate = original
        CandidateGenerator._serve = raw_serve

    rep = {"loads": list(loads), "seed": args.seed, "stop": done, **tally.report()}
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
