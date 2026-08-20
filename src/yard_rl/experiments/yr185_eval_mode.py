"""YR-185 ① — 최적화한 얼굴과 채점한 얼굴이 달랐나 (재학습 없음).

■ 가설
  학습은 `sample=True`(추첨), 평가는 `sample=False`(최선)다. 같은 가중치인데
  행동이 몇 배씩 다르고(slots48 6.3x·fixed15 4.1x·buy 1.4x), **괴리가 작은
  팔일수록 성적이 좋다**(−40.4 < −73.4 < −175.9). 최적화 대상과 채점 대상이
  다른 물건이었다는 증거로 보인다.

■ 방법 — 재학습하지 않는다
  같은 가중치·같은 6일(YR-171-C 와 동일)을 **추첨 모드로** 다시 평가한다.
  추첨은 난수이므로 **고정 씨앗 3개**로 돌려 평균과 흩어짐을 함께 본다.

■ 사전 동결 (이 파일 커밋 시점에 확정)
  · 두 모드 결과를 **둘 다** 보고한다. 유리한 쪽만 고르지 않는다.
  · 기준선(K)·규칙 판매(greedy)는 기존 산출물을 그대로 인용한다(재실행 없음).
  · 추첨 씨앗은 아래 SAMPLE_SEEDS 로 고정하고 결과를 보고 바꾸지 않는다.
  · 이것은 **진단**이지 성능 판정이 아니다. 새 배포 후보를 만들지 않는다.
"""
from __future__ import annotations

from ..integrated.terminal_stream import DIURNAL_DAY_TOTAL

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr185_eval_mode")
SPEC = ".claude/docs/dashboard-task-specs/YR-185-training-setup-audit.md"
REF_C = Path("outputs/reports/yr171c_slots/eval_slots.json")    # 최선 모드 + K
REF_G = Path("outputs/reports/yr179_greedy/greedy_baseline.json")  # 규칙 판매
CKPT_DIR = Path("outputs/reports/yr171c_slots")
ARMS = ("fixed15", "slots48", "slots48_buy")
TRAIN_SEEDS = (8_400_000, 8_500_000)
SAMPLE_SEEDS = (11, 22, 33)     # 추첨 고정 씨앗 — 결과 보고 바꾸지 않는다


def _worker(args) -> dict:
    import torch
    import torch.multiprocessing as _mp
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..integrated.transfer_head import (PpoSellPolicy, TransferActor,
                                            TransferCritic)
    from ..integrated.yard_layout import terminal_layout
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import run_episode_diurnal
    from .yr171c_train import _arm_kwargs
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    arm, ts, day, ss = args
    a, c = TransferActor(), TransferCritic()
    st = torch.load(CKPT_DIR / f"{arm}_s{ts}" / "net.pt",
                    map_location="cpu", weights_only=True)
    a.load_state_dict(st["actor"])
    c.load_state_dict(st["critic"])
    pol = PpoSellPolicy(a, c, mode="live", sample=True, seed=ss,
                        layout=terminal_layout())
    ep = run_episode_diurnal(day, pol, load_kf(), exec_config=ADOPTED_C0_GUARD,
                             day_plan_public=True, **_arm_kwargs(arm))
    return {"arm": arm, "train_seed": ts, "day": day, "sample_seed": ss,
            "phi_final": round(ep["phi_final"], 4),
            "n_space": ep["n_space"], "n_time": ep["n_time"],
            "admitted": ep["admitted"]}


def run() -> Path:
    from concurrent.futures import ProcessPoolExecutor
    from statistics import fmean, pstdev

    import torch.multiprocessing as _mp
    from ..integrated.repro import code_dirty, repro_stamp
    from ..integrated.terminal_stream import OBS_24H
    from .yr171c_eval import eval_days
    _mp.set_sharing_strategy("file_system")

    days = eval_days()                       # YR-171-C 와 같은 6일
    ref_c = json.loads(REF_C.read_text(encoding="utf-8"))
    ref_g = json.loads(REF_G.read_text(encoding="utf-8"))
    K = {r["day"]: r["phi_final"] for r in ref_c["rows"] if r["arm"] == "K"}
    # 규칙 판매 — 같은 날 기준선 대비 (재실행 없이 인용)
    Kg = {r["day"]: r["phi_final"] for r in ref_g["rows"] if r["arm"] == "K"}
    assert all(abs(K[d] - Kg[d]) < 1e-9 for d in days), "K 기준선 불일치"
    greedy = {r["day"]: r["phi_final"] - K[r["day"]]
              for r in ref_g["rows"] if r["arm"] == "greedy"}

    jobs = [(a, ts, d, ss) for a in ARMS for ts in TRAIN_SEEDS
            for d in days for ss in SAMPLE_SEEDS]
    with ProcessPoolExecutor(max_workers=24) as pool:
        rows = list(pool.map(_worker, jobs))

    # ---- 추첨 모드: 씨앗 평균 → 시드 평균 → 날별 값
    samp: dict[str, dict[int, float]] = {a: {} for a in ARMS}
    spread: dict[str, list[float]] = {a: [] for a in ARMS}
    for a in ARMS:
        for d in days:
            per_ts = []
            for ts in TRAIN_SEEDS:
                v = [r["phi_final"] for r in rows
                     if r["arm"] == a and r["train_seed"] == ts and r["day"] == d]
                per_ts.append(fmean(v))
                if len(v) > 1:
                    spread[a].append(pstdev(v))
            samp[a][d] = fmean(per_ts) - K[d]

    # ---- 최선 모드: 기존 산출물 인용
    argm: dict[str, dict[int, float]] = {a: {} for a in ARMS}
    for a in ARMS:
        for d in days:
            v = [r["phi_final"] - K[d] for r in ref_c["rows"]
                 if r["arm"] == a and r["day"] == d]
            argm[a][d] = fmean(v)

    def stat(vals):
        m = fmean(vals)
        se = pstdev(vals) / (len(vals) - 1) ** 0.5
        return {"mean": round(m, 2), "se": round(se, 2),
                "t": round(m / se, 2) if se else None}

    summary = []
    for a in ARMS:
        sv = [samp[a][d] for d in days]
        av = [argm[a][d] for d in days]
        gv = [greedy[d] for d in days]
        summary.append({
            "arm": a,
            "argmax_vs_keep": stat(av),
            "sample_vs_keep": stat(sv),
            "sample_minus_argmax": stat([sv[i] - av[i] for i in range(len(days))]),
            "sample_minus_greedy": stat([sv[i] - gv[i] for i in range(len(days))]),
            "argmax_minus_greedy": stat([av[i] - gv[i] for i in range(len(days))]),
            "sample_seed_spread_mean": round(fmean(spread[a]), 2) if spread[a] else None,
            "n_space_mean": round(fmean(r["n_space"] for r in rows
                                        if r["arm"] == a), 1),
            "n_time_mean": round(fmean(r["n_time"] for r in rows
                                       if r["arm"] == a), 1)})

    res = {"experiment": "YR-185 ① 평가 모드 진단 (재학습 없음)",
           "kind": "diagnostic", "note": "성능 판정이 아니다. 배포 후보를 만들지 않는다.",
           "eval_days": days, "sample_seeds": list(SAMPLE_SEEDS),
           "greedy_vs_keep_mean": round(
               sum(greedy[d] for d in days) / len(days), 2),
           "frozen": {"report_both_modes": True,
                      "sample_seeds_fixed_before_run": True,
                      "baselines_cited_not_rerun": ["K", "greedy"]},
           "summary": summary, "rows": rows,
           "admitted_all_3600": all(r["admitted"] == DIURNAL_DAY_TOTAL for r in rows),
           "code_dirty": bool(code_dirty()),
           "stamp": repro_stamp(experiment="YR-185 ① 평가 모드 진단",
                                seeds={"eval_days": days,
                                       "sample": list(SAMPLE_SEEDS)},
                                params={"arms": list(ARMS), "sample": True,
                                        "observation": OBS_24H.as_dict()},
                                prereg=SPEC)}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "eval_mode.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8")
    return p


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    p = run()
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"투입 {DIURNAL_DAY_TOTAL} 전건: {d['admitted_all_3600']}  "
          f"규칙 판매 기준 {d['greedy_vs_keep_mean']:+.2f}")
    print()
    print(f"{'팔':13s} {'최선(기존)':>14s} {'추첨(신규)':>14s} "
          f"{'추첨−최선':>16s} {'추첨−규칙':>16s}")
    for s in d["summary"]:
        a, g = s["argmax_vs_keep"], s["sample_vs_keep"]
        sa, sg = s["sample_minus_argmax"], s["sample_minus_greedy"]
        print(f"{s['arm']:13s} {a['mean']:>+9.2f}      {g['mean']:>+9.2f}      "
              f"{sa['mean']:>+8.2f}(t={sa['t']:>+5.2f}) "
              f"{sg['mean']:>+8.2f}(t={sg['t']:>+5.2f})")
    print()
    for s in d["summary"]:
        print(f"  {s['arm']:13s} 추첨 판매량 공간 {s['n_space_mean']:>7.1f} "
              f"시간 {s['n_time_mean']:>7.1f}  "
              f"씨앗간 흩어짐 {s['sample_seed_spread_mean']}")
    print("DONE", p)
