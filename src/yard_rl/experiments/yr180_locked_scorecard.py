"""YR-180 — 현 무대 SF-SPT 대비 잠금 성적표.

연구 최초로 **현행 배차 규칙 대비** 비용을 잰다. 지금까지 모든 성적은
"안 팔기" 대비였고 크레인 집행은 채택 정책 하나로 고정돼 있었다 — 그 정책이
이 무대에서 규칙을 이기는지는 측정된 적이 없다(YR-184).

■ 사전등록 (실행 전 동결)
  `.claude/docs/strategy-history/2026-08-17-YR-180-SF대비-잠금성적표-사전등록.md`
  팔·평가일·판정축·통계·하드가드·표본확장 규칙이 전부 그 문서에 박제돼 있다.
  **결과를 보고 바꾸지 않는다.**

■ 팔 (2×2 + 학습 변형) — 판매는 전부 고정 +15분 한 칸
  R    SF-SPT 규칙   · 판매 없음      ← 현행 배차
  R_S  SF-SPT 규칙   · 규칙 판매
  A    채택(동결)     · 판매 없음
  A_S  채택(동결)     · 규칙 판매
  A_P  채택(동결)     · 학습 판매(YR-171-C fixed15 s8400000)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr180_scorecard")
SPEC = ".claude/docs/dashboard-task-specs/YR-180-sfspt-locked-scorecard.md"
PREREG = (".claude/docs/strategy-history/"
          "2026-08-17-YR-180-SF대비-잠금성적표-사전등록.md")
LEARNED_CKPT = Path("outputs/reports/yr171c_slots/fixed15_s8400000/net.pt")

EVAL_SEED0 = 9_300_000          # 새 대역 — 이 판정에만 쓰고 재사용하지 않는다
N_EVAL_DAYS = 16
ARMS = ("R", "R_S", "A", "A_S", "A_P")
CI95_T = 2.13                   # df=15 양측 95%
CI_HALFWIDTH_LIMIT = 60.0       # 초과 시 사전 지정 표본 확장 (폭만 보고 판단)


def eval_days() -> list[int]:
    return [EVAL_SEED0 + i * 1000 for i in range(N_EVAL_DAYS)]


def _worker(args) -> dict:
    import torch
    import torch.multiprocessing as _mp
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..integrated.terminal_stream import OBS_24H
    from ..integrated.yard_layout import terminal_layout
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import KeepAllTrail, run_episode_diurnal
    from .yr179_greedy_baseline import GreedyOfferPolicy
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    arm, day = args
    kf, layout = load_kf(), terminal_layout()

    if arm in ("R", "A"):
        pol = KeepAllTrail()                       # 판매 없음
    elif arm in ("R_S", "A_S"):
        pol = GreedyOfferPolicy(kf, layout)        # 규칙 판매
    else:                                          # A_P — 학습 판매
        from ..integrated.transfer_head import (PpoSellPolicy, TransferActor,
                                                TransferCritic)
        a, c = TransferActor(), TransferCritic()
        st = torch.load(LEARNED_CKPT, map_location="cpu", weights_only=True)
        a.load_state_dict(st["actor"])
        c.load_state_dict(st["critic"])
        pol = PpoSellPolicy(a, c, mode="live", sample=False, layout=layout)

    head = "sf" if arm.startswith("R") else "adopted"
    ep = run_episode_diurnal(
        day, pol, kf, exec_head=head,
        exec_config=(ADOPTED_C0_GUARD if head == "adopted" else None),
        day_plan_public=False, time_slots=False, buy_net=None,
        _return_mbt=True)
    mbt = ep.pop("_mbt")

    # ---- A→O (실제 gate-in → gate-out). 엔진이 종료 시 장부를 닫아 검열 포함.
    ao: list[float] = []
    for sim in mbt.blocks.values():
        tl = getattr(sim, "time_ledger", None)
        if tl is not None:
            ao.extend(tl.terminal_turntime_samples_s())
    ao.sort()
    n = len(ao)
    return {"arm": arm, "day": day,
            "phi_final": round(ep["phi_final"], 4),
            "ao_mean_s": round(sum(ao) / n, 2) if n else None,
            "ao_p95_s": round(ao[min(n - 1, int(0.95 * n))], 1) if n else None,
            "ao_n": n,
            "n_space": ep["n_space"], "n_time": ep["n_time"],
            "policy_exceptions": ep["policy_exceptions"],
            "admitted": ep["admitted"],
            "observe_s": OBS_24H.observe_s}


def _paired(a: dict, b: dict, days: list[int], key: str):
    """같은 날 짝지어 a − b. 독립단위 = 날."""
    from statistics import fmean, pstdev
    d = [a[x][key] - b[x][key] for x in days
         if a.get(x, {}).get(key) is not None and b.get(x, {}).get(key) is not None]
    if len(d) < 2:
        return None
    m = fmean(d)
    se = pstdev(d) / (len(d) - 1) ** 0.5
    hw = CI95_T * se
    return {"n_days": len(d), "mean": round(m, 2), "se": round(se, 2),
            "t": round(m / se, 2) if se else None,
            "ci95": [round(m - hw, 2), round(m + hw, 2)],
            "ci_halfwidth": round(hw, 2),
            "n_negative": sum(1 for v in d if v < 0),
            # 사전등록 ⑤ — CI 가 0 을 배제할 때만 방향을 선언한다.
            "verdict": ("BETTER" if m + hw < 0 else
                        "WORSE" if m - hw > 0 else "INCONCLUSIVE")}


def run() -> Path:
    from concurrent.futures import ProcessPoolExecutor
    from statistics import fmean

    import torch.multiprocessing as _mp
    from ..integrated.repro import code_dirty, repro_stamp
    from ..integrated.terminal_stream import OBS_24H
    _mp.set_sharing_strategy("file_system")

    if not LEARNED_CKPT.exists():
        raise FileNotFoundError(f"학습 판매 가중치 없음: {LEARNED_CKPT}")
    days = eval_days()
    jobs = [(a, d) for a in ARMS for d in days]
    with ProcessPoolExecutor(max_workers=24) as pool:
        rows = list(pool.map(_worker, jobs))

    by = {a: {} for a in ARMS}
    for r in rows:
        by[r["arm"]][r["day"]] = r

    # ---- 하드 가드 (사전등록 ⑥) — 위반 시 실격, 완화하지 않는다
    guards = {
        "admitted_all_3600": all(r["admitted"] == 3600 for r in rows),
        "no_policy_exception": all(r["policy_exceptions"] == 0 for r in rows),
        "code_clean": not code_dirty(),
        "all_cells_present": all(len(by[a]) == len(days) for a in ARMS)}
    guards["all_pass"] = all(guards.values())

    summary = [{
        "arm": a,
        "phi_mean": round(fmean(by[a][d]["phi_final"] for d in days), 2),
        "ao_mean_min": round(fmean(by[a][d]["ao_mean_s"] for d in days) / 60.0, 2),
        "ao_p95_min": round(fmean(by[a][d]["ao_p95_s"] for d in days) / 60.0, 2),
        "n_space_mean": round(fmean(by[a][d]["n_space"] for d in days), 1),
        "n_time_mean": round(fmean(by[a][d]["n_time"] for d in days), 1),
    } for a in ARMS]

    pairs = [("A", "R"), ("R_S", "R"), ("A_S", "A"), ("A_P", "A_S"), ("A_S", "R")]
    contrasts = {}
    for x, y in pairs:
        contrasts[f"{x}_minus_{y}"] = {
            "phi": _paired(by[x], by[y], days, "phi_final"),
            "ao_mean_s": _paired(by[x], by[y], days, "ao_mean_s")}

    widest = max((c["phi"]["ci_halfwidth"] for c in contrasts.values()
                  if c["phi"]), default=0.0)
    res = {"experiment": "YR-180 현 무대 SF-SPT 대비 잠금 성적표",
           "prereg": PREREG,
           "eval_days": days, "n_days": len(days), "arms": list(ARMS),
           "note": "음수 = 앞 팔이 더 싸다/짧다. 같은 날 짝지어, 독립단위=날.",
           "guards": guards,
           "summary": summary, "contrasts": contrasts,
           "widest_ci_halfwidth": round(widest, 2),
           "extend_sample_required": bool(widest > CI_HALFWIDTH_LIMIT),
           "extend_rule": f"반폭 > {CI_HALFWIDTH_LIMIT} 이면 9_400_000 대역 16일 추가 "
                          f"(폭만 보고 판단 — 부호·유의성 미열람)",
           "rows": rows, "code_dirty": bool(code_dirty()),
           "stamp": repro_stamp(experiment="YR-180 SF-SPT 대비 잠금 성적표",
                                seeds={"eval_days": days},
                                params={"arms": list(ARMS), "time_slots": False,
                                        "observation": OBS_24H.as_dict()},
                                prereg=PREREG)}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "scorecard.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8")
    return p


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    p = run()
    d = json.loads(p.read_text(encoding="utf-8"))
    print("하드 가드:", json.dumps(d["guards"], ensure_ascii=False))
    print()
    print(f"{'팔':>5} {'Φ':>10} {'A→O 평균(분)':>13} {'P95(분)':>9} "
          f"{'공간':>7} {'시간':>7}")
    for s in d["summary"]:
        print(f"{s['arm']:>5} {s['phi_mean']:>10.2f} {s['ao_mean_min']:>13.2f} "
              f"{s['ao_p95_min']:>9.2f} {s['n_space_mean']:>7.1f} "
              f"{s['n_time_mean']:>7.1f}")
    print()
    for k, c in d["contrasts"].items():
        for axis in ("phi", "ao_mean_s"):
            v = c[axis]
            if v:
                u = "" if axis == "phi" else "s"
                print(f"  {k:>14} [{axis:>10}] {v['mean']:>+9.2f}{u} "
                      f"± {v['se']:>7.2f}  CI95 {v['ci95']}  "
                      f"t={v['t']:>+6.2f}  {v['verdict']}")
    print()
    print(f"최대 CI 반폭 {d['widest_ci_halfwidth']} "
          f"(확장 필요: {d['extend_sample_required']})")
    print("DONE", p)
