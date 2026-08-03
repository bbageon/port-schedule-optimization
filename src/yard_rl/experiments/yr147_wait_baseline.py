"""YR-147 1단계 — 기준 측정 (동결 프로토콜, 판정 아님).

회피 가능 전원대기율 = (진행 조합이 허용 목록에 있는데 전원 WAIT 선택) / (2크레인 결정).
진행 조합 = SERVE 또는 PRE_REHANDLE 을 ≥1 크레인이 드는 조합 (용어 계약).
실현 짝지은 후속비용: 발생 상태에서 sim 복제 2회 — ①정책 예상비용 최저 진행 조합 강제
②실제 선택 전원 WAIT 강제 — 이후 같은 정책으로 종료까지 계속.
D_wait_실현 = Φ(전원 WAIT) − Φ(진행), Φ = v2 실현 hard 총비용. 상한 30상태/초기화.

대상 = YR-145 B2 체크포인트 (결속+one-shot 계약 그대로) · 개발셋 = 훈련 대역 4셀×16
(판정 비사용). 프로토콜 동결 commit 451d997.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import fmean

import torch

from ..integrated import candidates as cand_mod
from ..integrated.baselines import JointRolloutGreedy, _apply
from ..integrated.candidates import CandidateGenerator
from ..integrated.encoding import StateNorm
from ..integrated.joint_distill import JointPairNet
from . import yr088_joint_rl as y88
from .yr090_dense_vessel import BASE, CELLS
from .yr100_candidate_eval import RC_EVAL
from .yr136_softplus_contract import _sim_contract
from .yr138_episode_pilot import _v2_hard_total
from .yr145_prepo_status_gate import OUT as OUT145

OUT = Path("outputs/reports/yr147_wait_baseline")
DEV_EPS = [(c, BASE[c] + i) for c in CELLS for i in range(16)]
MAX_PAIR = 30
PROG = ("SERVE", "PRE_REHANDLE")


def _load(ts: int):
    ck = torch.load(OUT145 / "b2" / f"ppo_s{ts}" / "net.pt", map_location="cpu")
    actor = JointPairNet(250); actor.load_state_dict(ck["actor"]); actor.eval()
    ck0 = torch.load(Path("outputs/reports/yr125_diff_credit") / f"diff1_s{ts}"
                     / "rl_net.pt", map_location="cpu")
    norm = StateNorm(refs=ck0["norm_refs"])
    y88.FORBID_WAIT = True
    return actor, norm, y88.RLPolicy(actor, norm, name=f"b2:{ts}")


def _record_prepo(sim, dp, assign):
    """평가 recorder 와 동일한 one-shot 이력 기록 (커스텀 루프 재현 — B2 계약 유지)."""
    if not cand_mod.PREPO_ONE_SHOT:
        return
    for c in dp.crane_ids:
        ref = getattr(assign[c], "job_ref", None)
        jid = cand_mod.prepo_bound_jid(getattr(ref, "job_id", "") or "")
        if jid is not None:
            if not hasattr(sim, "_prepo_history"):
                sim._prepo_history = set()
            sim._prepo_history.add(jid)


def _continue_to_end(sim, policy, gen):
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            break
        gen_by = {c: gen.generate(sim, c, y88.LEVEL) for c in dp.crane_ids}
        assign = policy.decide(sim, dp, gen_by)
        _record_prepo(sim, dp, assign)
        _apply(sim, assign)
    jobs = list(sim.jobs.values())
    compl = (sum(1 for j in jobs if j.status.name == "DONE") / len(jobs)) if jobs else 1.0
    return {"phi": _v2_hard_total(sim), "compl": round(compl, 4),
            "backlog": sim.unfinished_backlog()}


def run_one(cell, seed, actor, norm, policy, jr, pairs_left: int):
    sim = _sim_contract(cell, seed)
    gen = CandidateGenerator()
    st = {"dec": 0, "multi": 0, "ww_avoidable": 0, "pairs": []}
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            break
        gen_by = {c: gen.generate(sim, c, y88.LEVEL) for c in dp.crane_ids}
        assign = policy.decide(sim, dp, gen_by)
        st["dec"] += 1
        if len(dp.crane_ids) >= 2:
            st["multi"] += 1
            if all(g.kind.name == "WAIT" for g in assign.values()):
                rows, assigns = y88.build_rows(sim, dp, gen_by, norm, jr, 0)
                prog = [i for i, a in enumerate(assigns)
                        if any(a[c].kind.name in PROG for c in a)]
                if prog:
                    st["ww_avoidable"] += 1
                    if pairs_left - len(st["pairs"]) > 0:
                        with torch.no_grad():
                            cost, _ = actor(torch.tensor(rows, dtype=torch.float32))
                        best = min(prog, key=lambda i: float(cost[i]))
                        ww = next((i for i, a in enumerate(assigns)
                                   if all(a[c].kind.name == "WAIT" for c in a)), None)
                        simP = copy.deepcopy(sim)
                        _record_prepo(simP, dp, assigns[best])
                        _apply(simP, assigns[best])
                        rP = _continue_to_end(simP, policy, CandidateGenerator())
                        simW = copy.deepcopy(sim)
                        _apply(simW, assign)
                        rW = _continue_to_end(simW, policy, CandidateGenerator())
                        st["pairs"].append({
                            "cell": cell, "seed": seed, "t": float(sim.now),
                            "pred_ww": (float(cost[ww]) if ww is not None else None),
                            "pred_prog_best": float(cost[best]),
                            "phi_ww": rW["phi"], "phi_prog": rP["phi"],
                            "d_wait_realized": rW["phi"] - rP["phi"],
                            "compl_ww": rW["compl"], "compl_prog": rP["compl"],
                            "backlog_ww": rW["backlog"], "backlog_prog": rP["backlog"]})
        _record_prepo(sim, dp, assign)
        _apply(sim, assign)
    return st


def run(ts: int) -> dict:
    prev = cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT
    cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = True, True   # B2 계약
    try:
        actor, norm, policy = _load(ts)
        jr = JointRolloutGreedy(RC_EVAL, horizon_s=1800.0, generator=CandidateGenerator(),
                                forbid_strategic_wait=True)
        tot = {"dec": 0, "multi": 0, "ww_avoidable": 0}
        pairs = []
        for cell, seed in DEV_EPS:
            st = run_one(cell, seed, actor, norm, policy, jr, MAX_PAIR - len(pairs))
            for k in tot:
                tot[k] += st[k]
            pairs.extend(st["pairs"])
    finally:
        cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = prev
    d = [p["d_wait_realized"] for p in pairs]
    summary = {"ts": ts, **tot,
               "ww_avoidable_rate": (tot["ww_avoidable"] / tot["multi"]
                                     if tot["multi"] else 0.0),
               "n_pairs": len(pairs),
               "d_wait_mean": (fmean(d) if d else None),
               "d_wait_pos_share": (sum(1 for x in d if x > 0) / len(d) if d else None),
               "rank_error_share": (sum(1 for p in pairs
                                        if p["pred_ww"] is not None
                                        and p["pred_ww"] < p["pred_prog_best"]
                                        and p["d_wait_realized"] > 0) / len(pairs)
                                    if pairs else None),
               "compl_break_ww": sum(1 for p in pairs if p["compl_ww"] < 1.0),
               "compl_break_prog": sum(1 for p in pairs if p["compl_prog"] < 1.0)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"baseline_s{ts}.json").write_text(
        json.dumps({"summary": summary, "pairs": pairs, "protocol_commit": "451d997"},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=int, required=True)
    run(ap.parse_args().ts)
    print("DONE")
