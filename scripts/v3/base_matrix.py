"""크레인 바닥 × 부하 매트릭스 ([[YR-243]]) — 바닥 순위가 부하에 따라 뒤집히나.

    PYTHONPATH=src python scripts/v3/base_matrix.py [작업자수]

재배치를 끄고(`NO_REALLOC`) 바닥만 갈아 끼운다. 도착 명단이 여섯 바닥에서
**똑같으므로** 차이는 전부 크레인 순서 탓이다.

★미완료 대수를 같이 본다 — Φ 는 창 끝에서 검열되므로(완료차 O−A · 미완료차 T−A),
  일을 덜 끝낸 바닥은 **싸 보인다**. 미완료가 많은데 Φ 가 낮으면 그 순위는 못 쓴다.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

LOADS = (3_500, 5_000, 7_500, 12_500, 15_000)
SEED = 9_900_777
OUT = Path("outputs/v3/base-matrix")


def one(kw):
    from yard_rl.v3.stage.episode import run_episode
    r = run_episode(load=kw["load"], arm="NO_REALLOC", seed=SEED,
                    dispatcher=kw["base"])
    b = r.breakdown
    return {"base": kw["base"], "load": kw["load"], "phi": r.phi_krw,
            "c_wait": b["c_wait"], "c_move": b["c_move"],
            "c_rehandle": b["c_rehandle"], "n_trucks": b["n_trucks"],
            "n_censored": b["n_censored"],
            "mean_tt_s": b["mean_turn_time_s"], "p90_tt_s": b["p90_turn_time_s"],
            "over_ratio": b["over_ratio"]}


def main() -> int:
    from yard_rl.v3.stage.episode import DISPATCHERS_READY
    bases = [b for b in DISPATCHERS_READY if b != "SPT"]   # SPT ≡ SF_SPT (실측)
    jobs = [{"base": b, "load": v} for v in LOADS for b in bases]
    n_w = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"■ 바닥 {len(bases)}종 × 부하 {len(LOADS)}종 = {len(jobs)}판 "
          f"· 작업자 {n_w}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=n_w) as ex:
        for r in ex.map(one, jobs):
            rows.append(r)
            print(f"  · {r['base']:<8} 부하 {r['load']:>6,} "
                  f"Φ {r['phi']:>15,.0f} 미완료 {r['n_censored']:>4,}", flush=True)
            (OUT / "rows.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'바닥':<9}" + "".join(f"{v//1000:>7}천" for v in LOADS) + "   ← Φ (최저=1.000)")
    by = {(r["base"], r["load"]): r for r in rows}
    for b in bases:
        line = f"{b:<9}"
        for v in LOADS:
            lo = min(by[(x, v)]["phi"] for x in bases)
            line += f"{by[(b, v)]['phi'] / lo:>8.3f}"
        print(line)
    print(f"\n{'바닥':<9}" + "".join(f"{v//1000:>7}천" for v in LOADS) + "   ← 미완료 대수")
    for b in bases:
        print(f"{b:<9}" + "".join(f"{by[(b, v)]['n_censored']:>8,}" for v in LOADS))
    print(f"\n{'바닥':<9}" + "".join(f"{v//1000:>7}천" for v in LOADS) + "   ← P90 회전(분)")
    for b in bases:
        print(f"{b:<9}" + "".join(f"{by[(b, v)]['p90_tt_s']/60:>8.0f}" for v in LOADS))
    print(f"\n■ 원자료 {OUT}/rows.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
