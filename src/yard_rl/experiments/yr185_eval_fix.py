"""YR-185 ②③ 평가 — 교정본을 대조군과 **같은 6일·같은 결정론**으로 잰다.

사전등록의 판정 기준("학습 곡선이 안 팔기 기준선을 넘는가")은 실행 불가능했다:
학습 Φ 는 **학습일**(8400036~39), 안 팔기 기준선은 **평가일**(9200000~) 이라
다른 날짜다. 직접 비교하면 안 된다. 그래서 YR-171-C 와 **같은 평가 규약**으로
다시 잰다 — 같은 6일·argmax·짝 비교. 기준선 K 와 대조군 fixed15 는 기존
산출물을 인용한다(재실행 없음).
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("outputs/reports/yr185_retrain")
REF = Path("outputs/reports/yr171c_slots/eval_slots.json")
CKPT = OUT / "net.pt"


def run() -> Path:
    from concurrent.futures import ProcessPoolExecutor
    from statistics import fmean, pstdev

    import torch.multiprocessing as _mp
    from ..integrated.repro import code_dirty, repro_stamp, write_result
    from ..integrated.terminal_stream import OBS_24H
    from .yr171c_eval import _worker, eval_days
    _mp.set_sharing_strategy("file_system")

    days = eval_days()
    with ProcessPoolExecutor(max_workers=24) as pool:
        rows = list(pool.map(_worker, [("fix", "fixed15", str(CKPT), d)
                                       for d in days]))
    ref = json.loads(REF.read_text(encoding="utf-8"))
    K = {r["day"]: r["phi_final"] for r in ref["rows"] if r["arm"] == "K"}
    ctrl = {}
    for r in ref["rows"]:
        if r["arm"] == "fixed15":
            ctrl.setdefault(r["day"], []).append(r["phi_final"] - K[r["day"]])

    fix = {r["day"]: r["phi_final"] - K[r["day"]] for r in rows}

    def stat(v):
        m = fmean(v); se = pstdev(v) / (len(v) - 1) ** 0.5
        return {"mean": round(m, 2), "se": round(se, 2),
                "t": round(m / se, 2) if se else None,
                "n_neg": sum(1 for x in v if x < 0), "n": len(v)}

    fv = [fix[d] for d in days]
    cv = [fmean(ctrl[d]) for d in days]
    res = {"experiment": "YR-185 ②③ 교정본 평가 (YR-171-C 와 같은 규약)",
           "kind": "diagnostic", "eval_days": days,
           "note": "음수 = 안 팔기보다 좋다. argmax·결정론·같은 날 짝지어.",
           "fix_vs_keep": stat(fv), "control_vs_keep": stat(cv),
           "fix_minus_control": stat([fv[i] - cv[i] for i in range(len(days))]),
           "greedy_vs_keep": -170.05,
           "rows": rows, "code_dirty": bool(code_dirty()),
           "stamp": repro_stamp(experiment="YR-185 ②③ 교정본 평가",
                                seeds={"eval_days": days},
                                params={"ckpt": str(CKPT), "sample": False,
                                        "observation": OBS_24H.as_dict()},
                                prereg=".claude/docs/dashboard-task-specs/"
                                       "YR-185-training-setup-audit.md")}
    p = OUT / "eval_fix.json"
    write_result(p, res)
    return p


if __name__ == "__main__":
    p = run()
    d = json.loads(p.read_text(encoding="utf-8"))
    for k in ("control_vs_keep", "fix_vs_keep"):
        s = d[k]
        print(f"{k:18s} {s['mean']:>+9.2f} ± {s['se']:>6.2f} (t={s['t']:>+6.2f}) "
              f"안팔기보다 나은 날 {s['n_neg']}/{s['n']}")
    s = d["fix_minus_control"]
    print(f"{'교정 − 대조':18s} {s['mean']:>+9.2f} ± {s['se']:>6.2f} "
          f"(t={s['t']:>+6.2f}) 개선된 날 {s['n_neg']}/{s['n']}")
    print(f"{'규칙 판매':18s} {d['greedy_vs_keep']:>+9.2f}")
    print("DONE", p)
