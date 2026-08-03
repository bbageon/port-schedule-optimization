"""YR-147 2단계 — A/B/C 학습·계측 (사전등록 동결: spec YR-147 §2단계 구현 계약).

A = 현재 무기한 WAIT (YR-145 B2 체크포인트 재사용 — 학습 없음)
B = DEFER_ALL (후보 삭제 없이 전 대기 유한화, 만료 now+600s)
C = DEFER_TRIGGER (관측 trigger 있을 때만 전략적 DEFER — 부재 시 구조 fallback 전용)
유일 변경 = 대기 행동 의미. 결속+one-shot(B2 계약)·보상·상태·PPO 불변.
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
from .yr139_blockq_v4_ppo import train_one
from .yr145_prepo_status_gate import OUT as OUT145
from .yr147_wait_baseline import PROG, _record_prepo

OUT = Path("outputs/reports/yr147_defer")
ARM_WAIT_MODE = {"a": "WAIT", "b": "DEFER_ALL", "c": "DEFER_TRIGGER"}
ARM_ROOT = {"a": OUT145 / "b2", "b": OUT / "b", "c": OUT / "c"}
DEV_EPS = [(cell, BASE[cell] + i) for cell in CELLS for i in range(16)]
PAIR_CELL_QUOTA, PAIR_EP_CAP, TOPK = 8, 2, 2


def train(ts: int, arm: str):
    prev = cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT, cand_mod.WAIT_MODE
    cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = True, True     # B2 계약 유지
    cand_mod.WAIT_MODE = ARM_WAIT_MODE[arm]
    try:
        return train_one(ts, out_root=OUT / arm)
    finally:
        (cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT, cand_mod.WAIT_MODE) = prev


# ------------------------------------------------------------------ 3단계 라벨 (22차 계약)
def select_progress_combos(assigns, prog_idx, rng, k_max=4):
    """정책독립 후보선정 (22차 — PPO 점수 top-K 금지): 전부 ≤k_max 면 전부.
    초과 시 ①총 계획시간 최소(SF-SPT 근사·결정론) ②행동유형 서명 층화 ③무작위 잔여."""
    if len(prog_idx) <= k_max:
        return list(prog_idx)
    def plan_dur(i):
        a = assigns[i]
        return sum((c.plan.duration_s if getattr(c, "plan", None) is not None
                    else 0.0) for c in a.values())
    chosen = [min(prog_idx, key=plan_dur)]
    sigs = {tuple(sorted(assigns[chosen[0]][c].kind.name for c in assigns[chosen[0]]))}
    for i in sorted(prog_idx):
        if len(chosen) >= k_max:
            break
        sig = tuple(sorted(assigns[i][c].kind.name for c in assigns[i]))
        if sig not in sigs and i not in chosen:
            chosen.append(i); sigs.add(sig)
    rest = [i for i in prog_idx if i not in chosen]
    while len(chosen) < k_max and rest:
        chosen.append(rest.pop(rng.randrange(len(rest))))
    return chosen


def lex_label(sim, dp, assigns, idx_list, policy):
    """공동행동별 (미완 비율, backlog, v2 비용) — 사전식 비교용 (완주→backlog→비용 우선,
    22차 ④: 비용 최솟값 선행 금지). 반사실 미래는 라벨 전용 — 정책 관측 진입 금지."""
    out = {}
    for i in idx_list:
        s2 = copy.deepcopy(sim)
        _record_prepo(s2, dp, assigns[i])
        _apply(s2, assigns[i])
        r = _continue_to_end(s2, policy, CandidateGenerator())
        out[i] = (round(1.0 - r["compl"], 6), r["backlog"], r["phi"])
    return out


# ------------------------------------------------------------------ 2단계 계측 (파일럿)
def _load(arm: str, ts: int):
    ck = torch.load(ARM_ROOT[arm] / f"ppo_s{ts}" / "net.pt", map_location="cpu")
    actor = JointPairNet(250); actor.load_state_dict(ck["actor"]); actor.eval()
    ck0 = torch.load(Path("outputs/reports/yr125_diff_credit") / f"diff1_s{ts}"
                     / "rl_net.pt", map_location="cpu")
    norm = StateNorm(refs=ck0["norm_refs"])
    y88.FORBID_WAIT = True
    return actor, norm, y88.RLPolicy(actor, norm, name=f"{arm}:{ts}")


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
    return {"phi": _v2_hard_total(sim), "compl": compl, "backlog": sim.unfinished_backlog()}


def _measure_ep(cell, seed, actor, norm, policy, jr, do_pairs, pair_budget):
    sim = _sim_contract(cell, seed)
    gen = CandidateGenerator()
    st = {"dec": 0, "multi": 0, "ww_avoidable": 0, "kind_counts": {}, "pairs": [],
          "defer_exec": 0, "defer_triggered": 0, "defer_untriggered": 0}
    ep_pairs = 0
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            break
        gen_by = {c: gen.generate(sim, c, y88.LEVEL) for c in dp.crane_ids}
        assign = policy.decide(sim, dp, gen_by)
        st["dec"] += 1
        for c in dp.crane_ids:
            k = assign[c].kind.name
            st["kind_counts"][k] = st["kind_counts"].get(k, 0) + 1
            if k == "WAIT" and getattr(assign[c], "defer_until", None) is not None:
                st["defer_exec"] += 1
                if getattr(assign[c], "defer_trigger", None) is not None:
                    st["defer_triggered"] += 1
                else:
                    st["defer_untriggered"] += 1
        if len(dp.crane_ids) >= 2:
            st["multi"] += 1
        if len(dp.crane_ids) >= 2 and all(g.kind.name == "WAIT" for g in assign.values()):
            rows, assigns = y88.build_rows(sim, dp, gen_by, norm, jr, 0)
            prog = [i for i, a in enumerate(assigns)
                    if any(a[c].kind.name in PROG for c in a)]
            if prog:
                st["ww_avoidable"] += 1
                if do_pairs and ep_pairs < PAIR_EP_CAP and len(st["pairs"]) < pair_budget:
                    with torch.no_grad():
                        cost, _ = actor(torch.tensor(rows, dtype=torch.float32))
                    top = sorted(prog, key=lambda i: float(cost[i]))[:TOPK]
                    phis = []
                    for i in top:
                        s2 = copy.deepcopy(sim)
                        _record_prepo(s2, dp, assigns[i])
                        _apply(s2, assigns[i])
                        phis.append(_continue_to_end(s2, policy, CandidateGenerator()))
                    sW = copy.deepcopy(sim)
                    _apply(sW, assign)
                    rW = _continue_to_end(sW, policy, CandidateGenerator())
                    st["pairs"].append({
                        "cell": cell, "seed": seed, "t": float(sim.now),
                        "pred_ww": min(float(cost[i]) for i, a in enumerate(assigns)
                                       if all(a[c].kind.name == "WAIT" for c in a)),
                        "pred_prog_topk": [float(cost[i]) for i in top],
                        "phi_ww": rW["phi"], "phi_prog_topk": [p["phi"] for p in phis],
                        "d_wait_realized": rW["phi"] - min(p["phi"] for p in phis),
                        "compl_ww": rW["compl"],
                        "compl_prog_min": min(p["compl"] for p in phis)})
                    ep_pairs += 1
        _record_prepo(sim, dp, assign)
        _apply(sim, assign)
    jobs = list(sim.jobs.values())
    compl = (sum(1 for j in jobs if j.status.name == "DONE") / len(jobs)) if jobs else 1.0
    st.update({"cell": cell, "seed": seed, "v2_total": _v2_hard_total(sim),
               "compl": compl, "backlog": sim.unfinished_backlog(),
               "defer_wakes": sum(1 for e in sim.event_log if e[1] == "DEFER_WAKE")})
    return st


def measure(arm: str, ts: int) -> dict:
    prev = cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT, cand_mod.WAIT_MODE
    cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT = True, True
    cand_mod.WAIT_MODE = ARM_WAIT_MODE[arm]
    try:
        actor, norm, policy = _load(arm, ts)
        jr = JointRolloutGreedy(RC_EVAL, horizon_s=1800.0, generator=CandidateGenerator(),
                                forbid_strategic_wait=True)
        do_pairs = arm in ("b", "c")                 # A 짝은 1단계(비층화) 참조
        eps, pairs = [], []
        for cell in CELLS:                           # 층화: 셀별 짝 할당 (21차 ①)
            budget = PAIR_CELL_QUOTA
            for i in range(16):
                st = _measure_ep(cell, BASE[cell] + i, actor, norm, policy, jr,
                                 do_pairs, budget)
                budget -= len(st["pairs"])
                pairs.extend(st.pop("pairs"))
                eps.append(st)
    finally:
        (cand_mod.BOUND_REPO, cand_mod.PREPO_ONE_SHOT, cand_mod.WAIT_MODE) = prev
    multi = sum(e["multi"] for e in eps)
    ww = sum(e["ww_avoidable"] for e in eps)
    kc = {}
    for e in eps:
        for k, v in e["kind_counts"].items():
            kc[k] = kc.get(k, 0) + v
    tot_k = sum(kc.values()) or 1
    d = [p["d_wait_realized"] for p in pairs]
    summary = {
        "arm": arm, "ts": ts, "n_eps": len(eps),
        "compl_min": min(e["compl"] for e in eps),
        "n_incomplete_eps": sum(1 for e in eps if e["compl"] < 1.0),
        "backlog_total": sum(e["backlog"] for e in eps),
        "ww_avoidable": ww, "multi": multi,
        "ww_rate": (ww / multi if multi else 0.0),
        "v2_mean": fmean(e["v2_total"] for e in eps),
        "share": {k: round(v / tot_k, 4) for k, v in sorted(kc.items())},
        "defer_exec": sum(e["defer_exec"] for e in eps),
        "defer_triggered": sum(e["defer_triggered"] for e in eps),
        "defer_untriggered": sum(e["defer_untriggered"] for e in eps),
        "defer_wakes": sum(e["defer_wakes"] for e in eps),
        "n_pairs": len(pairs),
        "pair_cells": {c: sum(1 for p in pairs if p["cell"] == c) for c in CELLS},
        "d_wait_neg": (sum(1 for x in d if x < 0) / len(d) if d else None),
        "d_wait_zero": (sum(1 for x in d if x == 0) / len(d) if d else None),
        "d_wait_pos": (sum(1 for x in d if x > 0) / len(d) if d else None)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pilot_{arm}_s{ts}.json").write_text(
        json.dumps({"summary": summary, "eps": eps, "pairs": pairs,
                    "prereg_commit": "2369af8"}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--measure", type=int, default=0)
    ap.add_argument("--arm", choices=("a", "b", "c"), required=True)
    a = ap.parse_args()
    if a.train:
        assert a.arm in ("b", "c"), "A 는 YR-145 체크포인트 재사용 — 학습 없음"
        train(a.train, a.arm)
    if a.measure:
        measure(a.arm, a.measure)
    print("DONE")
