"""YR-171-C 평가 — 48칸이 실제로 이득인가 (같은 날 짝지어, 전건 KEEP 기준).

■ 팔 (전부 같은 날·같은 집행설정·결정론 실행)
  K            아무것도 안 판다 — 기준선
  fixed15      +15분 한 칸 (YR-174 재현)
  slots48      48칸 + 결정론 계산식 견적
  slots48_buy  48칸 + BUY 견적망 견적

■ 무엇을 가르나
  slots48 − fixed15      = **행동 공간을 넓힌 효과** (48칸 자체)
  slots48_buy − slots48  = **견적을 바꾼 효과** (BUY 견적망)
두 축을 한 번에 바꾼 결과로 원인을 주장하지 않는다(YR-171 사다리 규약).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr171c_slots")
SPEC = ".claude/docs/dashboard-task-specs/YR-171-time-sale-slot-contract.md"
N_EVAL_DAYS = 6
EVAL_SEED0 = 9_200_000          # 학습·정답지 수집과 겹치지 않는 대역


def eval_days() -> list[int]:
    return [EVAL_SEED0 + i * 1000 for i in range(N_EVAL_DAYS)]


def _worker(args) -> dict:
    import torch
    import torch.multiprocessing as _mp
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..v1.ppo_policy import PpoSellPolicy, TransferActor, TransferCritic
    from ..integrated.yard_layout import terminal_layout
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import KeepAllTrail, run_episode_diurnal
    from .yr171c_train import _arm_kwargs
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    name, arm, ckpt, day = args
    if arm == "K":
        pol, kw = KeepAllTrail(), {"time_slots": False, "buy_net": None}
    else:
        a, c = TransferActor(), TransferCritic()
        st = torch.load(ckpt, map_location="cpu", weights_only=True)
        a.load_state_dict(st["actor"])
        c.load_state_dict(st["critic"])
        pol = PpoSellPolicy(a, c, mode="live", sample=False,
                            layout=terminal_layout())
        kw = _arm_kwargs(arm)
    ep = run_episode_diurnal(day, pol, load_kf(), exec_config=ADOPTED_C0_GUARD,
                             day_plan_public=True, **kw)
    slots = [r.get("slot") for r in ep["sell_ledger"]
             if r.get("axis") == "TIME" and r.get("decision") == "DEFER"]
    defers = [r.get("defer_s") for r in ep["sell_ledger"]
              if r.get("axis") == "TIME" and r.get("decision") == "DEFER"
              and r.get("defer_s") is not None]
    return {"name": name, "arm": arm, "day": day,
            "phi_final": round(ep["phi_final"], 4),
            "n_space": ep["n_space"], "n_time": ep["n_time"],
            "n_slots_used": len({s for s in slots if s is not None}),
            "defer_mean_s": round(sum(defers) / len(defers), 1) if defers else None,
            "defer_max_s": max(defers) if defers else None,
            "admitted": ep["admitted"]}


def run() -> Path:
    from concurrent.futures import ProcessPoolExecutor
    from statistics import fmean, pstdev

    import torch.multiprocessing as _mp
    from ..integrated.repro import code_dirty, repro_stamp
    from ..integrated.terminal_stream import OBS_24H
    from .yr171c_train import ARMS
    _mp.set_sharing_strategy("file_system")

    days = eval_days()
    jobs = [("K", "K", None, d) for d in days]
    found = []
    for arm in ARMS:
        for p in sorted(OUT.glob(f"{arm}_s*/net.pt")):
            nm = f"{arm}:{p.parent.name.split('_s')[-1]}"
            found.append((nm, arm))
            jobs += [(nm, arm, str(p), d) for d in days]
    if not found:
        raise RuntimeError("학습 가중치가 없다 — yr171c_train 을 먼저 실행할 것")

    with ProcessPoolExecutor(max_workers=24) as pool:
        rows = list(pool.map(_worker, jobs))

    by = {}
    for r in rows:
        by.setdefault(r["name"], []).append(r)
    kphi = {r["day"]: r["phi_final"] for r in by["K"]}

    summary = []
    for nm in ["K"] + [n for n, _ in found]:
        g = sorted(by[nm], key=lambda r: r["day"])
        d = [r["phi_final"] - kphi[r["day"]] for r in g]
        se = pstdev(d) / (len(d) - 1) ** 0.5 if len(d) > 1 else float("nan")
        summary.append({
            "name": nm, "arm": g[0]["arm"], "n_days": len(g),
            "phi_mean": round(fmean(r["phi_final"] for r in g), 2),
            "vs_keep_mean": round(fmean(d), 2), "vs_keep_se": round(se, 2),
            "vs_keep_t": round(fmean(d) / se, 2) if se else None,
            "n_better_than_keep": sum(1 for v in d if v < 0),
            "n_space_mean": round(fmean(r["n_space"] for r in g), 1),
            "n_time_mean": round(fmean(r["n_time"] for r in g), 1),
            "n_slots_used_mean": round(fmean(r["n_slots_used"] for r in g), 1),
            "defer_mean_s": ([r["defer_mean_s"] for r in g
                              if r["defer_mean_s"] is not None] or [None])[0]})

    def _pool(arm):
        v = [s["vs_keep_mean"] for s in summary if s["arm"] == arm]
        return {"arm": arm, "n_seeds": len(v),
                "vs_keep_mean": round(fmean(v), 2) if v else None,
                "n_better": sum(1 for x in v if x < 0)}
    pooled = [_pool(a) for a in ARMS]
    p48 = next((x for x in pooled if x["arm"] == "slots48"), None)
    p15 = next((x for x in pooled if x["arm"] == "fixed15"), None)
    pbuy = next((x for x in pooled if x["arm"] == "slots48_buy"), None)
    contrasts = {
        "slots48_minus_fixed15": (
            None if not (p48 and p15) or p48["vs_keep_mean"] is None
            or p15["vs_keep_mean"] is None
            else round(p48["vs_keep_mean"] - p15["vs_keep_mean"], 2)),
        "buy_minus_calc": (
            None if not (pbuy and p48) or pbuy["vs_keep_mean"] is None
            or p48["vs_keep_mean"] is None
            else round(pbuy["vs_keep_mean"] - p48["vs_keep_mean"], 2)),
        "note": "음수 = 그 축을 바꿔서 좋아졌다. 두 축을 한 번에 바꾼 결과로 "
                "원인을 주장하지 않는다(사다리 규약)."}

    res = {"experiment": "YR-171-C 평가 — 48칸 시간 좌표", "eval_days": days,
           "note": "vs_keep 양수 = 기준선(안 팔기)보다 나쁨. 같은 날 짝지어 비교.",
           "summary": summary, "pooled": pooled, "contrasts": contrasts,
           "rows": rows, "code_dirty": bool(code_dirty()),
           "stamp": repro_stamp(experiment="YR-171-C 평가 — 48칸 시간 좌표",
                                seeds={"eval_days": days},
                                params={"day_plan_public": True, "sample": False,
                                        "observation": OBS_24H.as_dict()},
                                prereg=SPEC)}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "eval_slots.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8")
    return p


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    p = run()
    d = json.loads(p.read_text(encoding="utf-8"))
    for s in d["summary"]:
        print(f"{s['name']:>18} Φ {s['phi_mean']:>9.2f}  vs_KEEP {s['vs_keep_mean']:>+8.2f}"
              f" ± {s['vs_keep_se']:>6.2f} (t={s['vs_keep_t']})  "
              f"공간 {s['n_space_mean']:>6.1f} 시간 {s['n_time_mean']:>6.1f}  "
              f"쓴 칸 {s['n_slots_used_mean']:>4.1f}")
    print("POOLED", json.dumps(d["pooled"], ensure_ascii=False))
    print("대비:", json.dumps(d["contrasts"], ensure_ascii=False))
    print("DONE", p)
