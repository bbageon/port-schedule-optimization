"""YR-179 — 규칙 기반 판매 기준선: 학습이 기여한 몫을 분리한다.

■ 왜
  지금까지 판매 성적은 전부 **학습 정책 대 안 팔기**만 쟀다. 학습 없이 규칙으로
  팔면 얼마인지는 한 번도 재지 않았다 — `UnifiedNetGain` 이 코드에 있는데
  어떤 실험도 import 하지 않았다. 그래서 −84.5 중 "학습"의 몫이 분리되지 않았다.

■ 무엇을 맞추나 (공정 비교 조건)
  규칙은 PPO 와 **같은 자리**만 맡는다 — `무엇을 내놓을지(OFFER)`. `어디로·언제`는
  같은 resolver 가 정한다. resolver 를 바꾸면 판매 메커니즘 자체가 달라져
  "학습의 기여"가 아니라 "다른 실험"이 된다.

  · 무대·집행·평가일·짝 비교 = YR-171-C 와 동일 (`ADOPTED_C0_GUARD`, 9200000~)
  · 행동 공간 = `fixed15` 와 동일 (고정 +15분 한 칸) — 48칸은 YR-177 이후 별도 축
  · 문턱 `gain_margin` = 코드 기본값(κ_T 1σ 유도) **동결** — 기준선을 깎지 않는다

■ 사전 동결
  규칙이 학습을 이겨도 그대로 박제한다. 그 경우 후속은 "왜 학습이 규칙보다
  못한가"이지 규칙을 약화시키는 것이 아니다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr179_greedy")
SPEC = ".claude/docs/dashboard-task-specs/YR-179-greedy-sell-baseline.md"
REF = Path("outputs/reports/yr171c_slots/eval_slots.json")   # 학습 팔 대조군


class GreedyOfferPolicy:
    """규칙 기반 OFFER 정책 — 학습 없음. 기존 `UnifiedNetGain` 을 **그대로** 쓴다.

    두 축 각각의 순이득을 규칙으로 재고 큰 쪽 1건을 내놓는다. 규칙 본체는
    건드리지 않는다(사전 동결) — 여기서는 `decide()` 계약으로 감싸기만 한다.
    """

    mode = "live"

    def __init__(self, kf, layout):
        from ..integrated.sell_gain import UnifiedNetGain
        self.rule = UnifiedNetGain(kf, layout)      # 문턱 기본값 = κ_T 1σ
        # 학습 루프와 같은 인터페이스(`KeepAllTrail` 과 동일 계약). 규칙은 학습하지
        # 않으므로 신경망 입력(rows·critic_in·logp)은 없고, 복기용 필드만 남긴다.
        self.trail: list[dict] = []

    def decide(self, mbt, src: str, cands: list, t: float):
        n0 = len(self.rule.ledger)
        space = [(j, e) for j, e, f in cands if f == "GATE_IN"]
        time_ = [(j, e) for j, e, f in cands if f == "GATE_OUT"]
        picks = []
        if space:
            p = self.rule.decide_space(mbt, src, space, t)
            if p is not None:
                picks.append((self.rule.ledger[-1]["net_gain"], p))
        if time_:
            p = self.rule.decide_time(mbt, src, time_, t)
            if p is not None:
                picks.append((self.rule.ledger[-1]["net_gain"], p))
        # 규칙의 자체 원장은 버린다 — 실제 집행 기록은 resolver 가 남긴다(중복 방지).
        del self.rule.ledger[n0:]
        pick = max(picks, key=lambda x: x[0])[1] if picks else None
        # action 은 PPO 와 같은 뜻 — 0 = KEEP, 1+i = i 번째 후보 OFFER.
        a = 0 if pick is None else 1 + next(
            i for i, (j, _, _) in enumerate(cands) if j == pick)
        self.trail.append({"t": t, "src": src, "action": a, "picked": pick,
                           "n_cands": len(cands), "value": 0.0})
        return pick


def _worker(args) -> dict:
    import torch
    import torch.multiprocessing as _mp
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..integrated.yard_layout import terminal_layout
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import KeepAllTrail, run_episode_diurnal
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    arm, day = args
    kf = load_kf()
    if arm == "K":
        pol = KeepAllTrail()
    else:
        pol = GreedyOfferPolicy(kf, terminal_layout())
    # 행동 공간을 fixed15 와 맞춘다 — 시간 축은 고정 +15분 한 칸.
    ep = run_episode_diurnal(day, pol, kf, exec_config=ADOPTED_C0_GUARD,
                             day_plan_public=True, time_slots=False,
                             buy_net=None)
    return {"name": arm, "arm": arm, "day": day,
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

    days = eval_days()                     # YR-171-C 와 **같은 6일**
    jobs = [(a, d) for a in ("K", "greedy") for d in days]
    with ProcessPoolExecutor(max_workers=24) as pool:
        rows = list(pool.map(_worker, jobs))

    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    kphi = {r["day"]: r["phi_final"] for r in by["K"]}

    summary = []
    for arm in ("K", "greedy"):
        g = sorted(by[arm], key=lambda r: r["day"])
        d = [r["phi_final"] - kphi[r["day"]] for r in g]
        se = pstdev(d) / (len(d) - 1) ** 0.5 if len(d) > 1 else float("nan")
        summary.append({
            "arm": arm, "n_days": len(g),
            "phi_mean": round(fmean(r["phi_final"] for r in g), 2),
            "vs_keep_mean": round(fmean(d), 2),
            "vs_keep_se": round(se, 2) if se else 0.0,
            "vs_keep_t": round(fmean(d) / se, 2) if se else None,
            "n_better_than_keep": sum(1 for v in d if v < 0),
            "n_space_mean": round(fmean(r["n_space"] for r in g), 1),
            "n_time_mean": round(fmean(r["n_time"] for r in g), 1)})

    # ---- 학습 팔과의 대조 (같은 날 짝지어). 학습 팔은 시드 2개 → 날별 평균.
    ref = json.loads(REF.read_text(encoding="utf-8")) if REF.exists() else None
    contrast = None
    if ref:
        rk = {r["day"]: r["phi_final"] for r in ref["rows"] if r["arm"] == "K"}
        gd = {r["day"]: r["phi_final"] - kphi[r["day"]] for r in by["greedy"]}
        learned = {}
        for r in ref["rows"]:
            if r["arm"] == "fixed15":
                learned.setdefault(r["day"], []).append(r["phi_final"] - rk[r["day"]])
        days_c = sorted(set(gd) & set(learned))
        diff = [fmean(learned[d]) - gd[d] for d in days_c]     # 학습 − 규칙
        se = pstdev(diff) / (len(diff) - 1) ** 0.5 if len(diff) > 1 else float("nan")
        contrast = {
            "note": "학습(fixed15, 2시드 날별평균) − 규칙(greedy). "
                    "음수 = 학습이 규칙보다 좋다.",
            "n_days": len(days_c),
            "learned_vs_keep": round(fmean(fmean(learned[d]) for d in days_c), 2),
            "greedy_vs_keep": round(fmean(gd[d] for d in days_c), 2),
            "learned_minus_greedy": round(fmean(diff), 2),
            "se": round(se, 2) if se else None,
            "t": round(fmean(diff) / se, 2) if se else None,
            "n_days_learned_better": sum(1 for v in diff if v < 0)}
        # K 재현 확인 — 같은 날·같은 설정이므로 완전 일치해야 한다.
        contrast["keep_reproduced"] = all(
            abs(kphi[d] - rk[d]) < 1e-9 for d in days_c)

    res = {"experiment": "YR-179 규칙 기반 판매 기준선",
           "eval_days": days,
           "note": "vs_keep 음수 = 안 팔기보다 좋다. 같은 날 짝지어 비교.",
           "frozen": {"gain_margin": "코드 기본값(κ_T 1σ 유도) — 조정 금지",
                      "action_space": "fixed15 와 동일(고정 +15분 한 칸)",
                      "resolver": "학습 팔과 동일 — OFFER 만 규칙으로 교체"},
           "summary": summary, "contrast_vs_learned": contrast, "rows": rows,
           "code_dirty": bool(code_dirty()),
           "stamp": repro_stamp(experiment="YR-179 규칙 기반 판매 기준선",
                                seeds={"eval_days": days},
                                params={"day_plan_public": True,
                                        "time_slots": False,
                                        "observation": OBS_24H.as_dict()},
                                prereg=SPEC)}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "greedy_baseline.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8")
    return p


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    p = run()
    d = json.loads(p.read_text(encoding="utf-8"))
    for s in d["summary"]:
        print(f"{s['arm']:>8} Φ {s['phi_mean']:>9.2f}  vs_KEEP {s['vs_keep_mean']:>+8.2f}"
              f" ± {s['vs_keep_se']:>6.2f} (t={s['vs_keep_t']})  "
              f"공간 {s['n_space_mean']:>6.1f} 시간 {s['n_time_mean']:>6.1f}")
    c = d["contrast_vs_learned"]
    if c:
        print()
        print(f"K 재현: {c['keep_reproduced']}")
        print(f"학습 {c['learned_vs_keep']:+.2f} / 규칙 {c['greedy_vs_keep']:+.2f}")
        print(f"학습 − 규칙 = {c['learned_minus_greedy']:+.2f} ± {c['se']} "
              f"(t={c['t']}, 학습 우위 {c['n_days_learned_better']}/{c['n_days']}일)")
    print("DONE", p)
