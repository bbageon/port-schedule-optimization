"""YR-185 ②③ — 사전등록한 **추첨 모드** 기준으로 교정본을 잰다.

사전등록 판정은 "학습 곡선(=추첨 모드)이 안 팔기 기준선을 넘는가"였다.
YR-185 ① 실측에서 구 정책의 추첨 모드는 **+14.62**(안 팔기보다 나쁨)였다.
argmax 평가(−129.15)는 대조군도 이미 넘었던 축이라 판별력이 없으므로,
사전등록한 축을 그대로 잰다 — 추첨 씨앗 11·22·33 (YR-185 ① 과 동일 고정).
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("outputs/reports/yr185_retrain")
CKPT = OUT / "net.pt"
REF_C = Path("outputs/reports/yr171c_slots/eval_slots.json")
SAMPLE_SEEDS = (11, 22, 33)


def _worker(args) -> dict:
    import torch
    import torch.multiprocessing as _mp
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..integrated.transfer_head import (PpoSellPolicy, TransferActor,
                                            TransferCritic)
    from ..integrated.yard_layout import terminal_layout
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import run_episode_diurnal
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    day, ss = args
    a, c = TransferActor(), TransferCritic()
    st = torch.load(CKPT, map_location="cpu", weights_only=True)
    a.load_state_dict(st["actor"]); c.load_state_dict(st["critic"])
    pol = PpoSellPolicy(a, c, mode="live", sample=True, seed=ss,
                        layout=terminal_layout())
    ep = run_episode_diurnal(day, pol, load_kf(), exec_config=ADOPTED_C0_GUARD,
                             day_plan_public=True, time_slots=False, buy_net=None)
    return {"day": day, "sample_seed": ss,
            "phi_final": round(ep["phi_final"], 4),
            "n_space": ep["n_space"], "n_time": ep["n_time"]}


def run() -> Path:
    from concurrent.futures import ProcessPoolExecutor
    from statistics import fmean, pstdev

    import torch.multiprocessing as _mp
    from ..integrated.repro import code_dirty, repro_stamp, write_result
    from ..integrated.terminal_stream import OBS_24H
    from .yr171c_eval import eval_days
    _mp.set_sharing_strategy("file_system")

    days = eval_days()
    with ProcessPoolExecutor(max_workers=24) as pool:
        rows = list(pool.map(_worker, [(d, s) for d in days
                                       for s in SAMPLE_SEEDS]))
    ref = json.loads(REF_C.read_text(encoding="utf-8"))
    K = {r["day"]: r["phi_final"] for r in ref["rows"] if r["arm"] == "K"}
    per = [fmean(r["phi_final"] for r in rows if r["day"] == d) - K[d]
           for d in days]
    m = fmean(per); se = pstdev(per) / (len(per) - 1) ** 0.5
    res = {"experiment": "YR-185 ②③ 교정본 — 사전등록 추첨 모드 판정",
           "kind": "diagnostic", "eval_days": days,
           "sample_seeds": list(SAMPLE_SEEDS),
           "sample_vs_keep": {"mean": round(m, 2), "se": round(se, 2),
                              "t": round(m / se, 2),
                              "n_neg": sum(1 for x in per if x < 0),
                              "n": len(per)},
           "control_sample_vs_keep": 14.62,   # YR-185 ① fixed15 (구 학습)
           "note": "음수 = 안 팔기보다 좋다. 사전등록 판정축.",
           "rows": rows, "code_dirty": bool(code_dirty()),
           "stamp": repro_stamp(experiment="YR-185 ②③ 추첨 모드 판정",
                                seeds={"eval_days": days,
                                       "sample": list(SAMPLE_SEEDS)},
                                params={"ckpt": str(CKPT), "sample": True,
                                        "observation": OBS_24H.as_dict()},
                                prereg=".claude/docs/dashboard-task-specs/"
                                       "YR-185-training-setup-audit.md")}
    p = OUT / "eval_fix_sample.json"
    write_result(p, res)
    return p


if __name__ == "__main__":
    p = run()
    d = json.loads(p.read_text(encoding="utf-8"))
    s = d["sample_vs_keep"]
    print(f"구 학습 추첨 모드 : {d['control_sample_vs_keep']:+8.2f}  (안 팔기보다 나쁨)")
    print(f"교정본  추첨 모드 : {s['mean']:+8.2f} ± {s['se']:.2f} "
          f"(t={s['t']:+.2f})  안팔기보다 나은 날 {s['n_neg']}/{s['n']}")
    print("★사전등록 판정:", "넘었다" if s["mean"] + 2.13 * s["se"] < 0 else "못 넘었다")
    print("DONE", p)
