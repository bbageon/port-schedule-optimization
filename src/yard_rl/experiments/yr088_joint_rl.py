"""YR-088 — 중앙 공동정책 값기반 RL (교사 흉내 아닌 보상 학습).

사용자 결정(2026-07-23): 지도학습(증류)은 새 환경 일반화 불가 → RL 로 전환.
방법: joint-combo Q-학습. JointPairNet 이 이미 Q(s, 조합)=score·argmin 구조라
손실만 "교사 CE"에서 "실현 기준재 보상 + 부트스트랩 TD"로 교체.

핵심 이점(과거 RL 실패 회피): 조합 하나=값 하나 → **per-크레인 신용 배분 없음**
→ 신용 희석·퇴화(yr061~068 병목) 원천 소거. QMIX mixer 우회 대신 직접 joint-Q.

Q = 기대 누적 기준재 **비용**(최소화). 보상 = 실현 구간비용(rollout 아닌 on-policy).
TD: Q(s,a) ← r + γ^(Δt/ref)·min_{a'} Q_target(s',a'). ε-greedy 탐험, target net.
재사용: 조합열거·JointPairNet·run_joint_episode·guard·mid/high 일반화 (조사 A).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import deque
from pathlib import Path
from statistics import fmean

import torch
from torch import nn

from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.adapter import capture
from ..integrated.baselines import (ActionMixError, JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, _apply, _feasible_joint,
                                    _wait_of, assert_healthy_action_mix, run_joint_episode)
from ..contract import CandidateKind
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.encoding import StateNorm, encode_observation
from ..integrated.joint_distill import JointPairNet
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr059_state_norm import fit_state_norm

LEVEL = InformationLevel.PRE_ADVICE
# 기준재 보상 (v5 채택 — v6 sts_wait 지배 재조정은 berth 악화·건전깨짐으로 실패, 되돌림).
# - crane_travel=0.1: 작은 이동비용(REPOSITION 남용 방지 보조).
# - sts_wait=5.0: 본선 선행신호(약). vessel_delay 33 유지.
# 진단: 본선 통제여지는 lookahead(미래계획) 의존 — 반응형 RL 은 보상 shaping 만으론 미포착
# (v6 실패가 확증). 본선 심화는 credit(n-step) 또는 hybrid rollout 축, 보상축 아님.
RC = RewardCalculator.numeraire({"crane_travel": 0.1, "empty_travel": 0.1, "sts_wait": 5.0})
GAMMA, REF_S = 0.99, 600.0
UNSERVED = 30.0          # 미완료 job 1건당 종결 페널티 (퇴화방지 ② — 완주 학습신호)
FORBID_WAIT = True       # 퇴화방지 ① — 일할 게 있으면 전략적 WAIT 조합 제외 (YR-052)
REPO_PENALTY = 0.5       # 퇴화방지 ③ — REPOSITION **행동당** 고정 벌점 (per-meter travel 은
#                          scale 7200m 탓에 개별 이동이 무시할 값 → 행동당 shaping 이 옳은 레버).
#                          reward-shaping(훈련만) — 평가 num 은 순수 기준재 유지.
CELLS = {"mid-loose": ("mid", 2.0), "high-loose": ("high", 2.0),
         "mid-tight": ("mid", 0.5), "high-tight": ("high", 0.5)}
BASE = {"mid-loose": 830000, "high-loose": 830100, "mid-tight": 830200, "high-tight": 830300}
SLOTS = tuple(sorted(c.crane_id for c in build_calibrated_profile().cranes))
OUT = Path("outputs/reports/yr088_joint_rl")


def _sim(cell, seed):
    prof = build_calibrated_profile()
    lvl, dm = CELLS[cell]
    s = TerminalSimulator(prof, generate_terminal_scenario(
        prof, seed, calibrated_load_params(lvl, vessel_deadline_mult=dm)), check_invariants=True)
    s.info_level = LEVEL
    return s


def build_rows(sim, dp, gen_by, norm, jr, k):
    """결정 시점의 feasible 공동조합 → (행렬 list, assigns). decide 행구성 재사용(use_vessel)."""
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "rl", k, generator=jr.generator)
    encs = {ob.crane_id: encode_observation(state, ob, norm=norm) for ob in obs}
    ca, cb = SLOTS
    ea, eb = encs.get(ca), encs.get(cb)
    ref = ea or eb
    z_yc, z_q, z_c = [0.0] * len(ref.yc), [0.0] * len(ref.queue), [0.0] * len(ref.cand[0])
    ctx_a = list(ref.g) + list(ref.vessel) + (list(ea.yc) + list(ea.queue) if ea else z_yc + z_q)
    ctx_b = (list(eb.yc) + list(eb.queue)) if eb else z_yc + z_q
    rows, assigns = [], []
    for combo in jr._admissible_combos(sim, dp, gen_by):
        assign = dict(zip(dp.crane_ids, combo))
        if not _feasible_joint(sim, assign):
            continue
        blk_a = (list(ea.cand[ea.candidate_ids.index(assign[ca].candidate_id)])
                 if ea and ca in assign else z_c)
        blk_b = (list(eb.cand[eb.candidate_ids.index(assign[cb].candidate_id)])
                 if eb and cb in assign else z_c)
        rows.append(ctx_a + blk_a + ctx_b + blk_b)
        assigns.append(assign)
    return rows, assigns


def collect_episode(cell, seed, net, norm, epsilon, rng):
    """현재 net(argmin)+ε 로 1 에피소드 실행, transition 수집.
    transition = (rows_k, pos_k, r_k, gamma_dt_k, rows_next_k|None).

    비용회계: 매 결정 경계에서 cut → pend(직전 실행 조합)의 보상에 누적. 강제 WAIT
    구간비용도 직전 실행 조합에 흡수(SMDP — WAIT 는 결정 아님). 다음 상태 = combos 있는
    다음 결정의 rows (WAIT 결정은 건너뜀). 종결은 다음상태 None."""
    sim = _sim(cell, seed)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RC, horizon_s=1800.0, generator=gen,
                            forbid_strategic_wait=FORBID_WAIT)     # 조합 열거 (WAIT 금지)
    trans, k = [], 0
    dp = sim.run_until_decision()
    sim.cost.cut()                    # 첫 결정 이전 구간 폐기
    last_b = sim.now
    pend = None                       # {rows,pos,t_act,r}
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        rows, assigns = build_rows(sim, dp, gen_by, norm, jr, k)
        raw = sim.cost.cut()          # [last_b, now] 구간비용
        if pend is not None:
            pend["r"] += RC.cost_for(interval_start_s=last_b, interval_end_s=sim.now,
                                     raw=raw, risk_max=0.0).total_normalized
        last_b = sim.now
        if pend is not None and assigns:      # 유효 다음상태 도달 → pend transition 마감
            gdt = GAMMA ** ((sim.now - pend["t_act"]) / REF_S)
            trans.append([pend["rows"], pend["pos"], pend["r"], gdt, rows])
            pend = None
        if not assigns:                       # 강제 WAIT (비용은 pend 에 계속 누적)
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        else:
            if net is not None and rng.random() >= epsilon:
                with torch.no_grad():
                    sc, _ = net(torch.tensor(rows, dtype=torch.float32))
                pick = int(torch.argmin(sc))
            else:
                pick = rng.randrange(len(assigns))
            n_repo = sum(1 for c in dp.crane_ids
                         if assigns[pick][c].kind == CandidateKind.REPOSITION)
            pend = {"rows": rows, "pos": pick, "t_act": sim.now, "r": REPO_PENALTY * n_repo}
            _apply(sim, assigns[pick])
        dp = sim.run_until_decision()
        k += 1
    jobs = list(sim.jobs.values())
    n_unserved = sum(1 for j in jobs if j.status.name != "DONE")
    if pend is not None:              # 종결 transition (+ 미완료 페널티)
        raw = sim.cost.cut()
        pend["r"] += RC.cost_for(interval_start_s=last_b, interval_end_s=sim.now,
                                 raw=raw, risk_max=0.0).total_normalized
        pend["r"] += UNSERVED * n_unserved
        trans.append([pend["rows"], pend["pos"], pend["r"], 1.0, None])
    stat = {"cell": cell, "seed": seed, "n": k,
            "completion": sum(1 for j in jobs if j.status.name == "DONE") / len(jobs)}
    return trans, stat


class RLPolicy:
    """학습 net 을 훈련과 **동일 조합집합**(forbid_strategic_wait)으로 argmin 실행.
    run_joint_episode 하네스용 (CentralJointValuePolicy 대신 — 훈련/평가 조합 일치)."""

    def __init__(self, net, norm, name="RL"):
        self.net = net
        self.norm = norm
        self.name = name
        self.jr = JointRolloutGreedy(RC, horizon_s=1800.0, generator=CandidateGenerator(),
                                     forbid_strategic_wait=FORBID_WAIT)

    def decide(self, sim, dp, gen_by):
        rows, assigns = build_rows(sim, dp, gen_by, self.norm, self.jr, 0)
        if not assigns:
            return {c: _wait_of(gen_by[c]) for c in dp.crane_ids}
        with torch.no_grad():
            sc, _ = self.net(torch.tensor(rows, dtype=torch.float32))
        return assigns[int(torch.argmin(sc))]


def train_step(net, target, opt, batch):
    """joint-combo TD: y = r + γ·min_{a'} Q_target(s',a'), Huber(Q(s,chosen), y). 비용최소화."""
    losses = []
    for rows_k, pos_k, r, gdt, rows_next in batch:
        sc, _ = net(torch.tensor(rows_k, dtype=torch.float32))
        q = sc[pos_k]
        if rows_next is None:
            y = torch.tensor(float(r))
        else:
            with torch.no_grad():
                scn, _ = target(torch.tensor(rows_next, dtype=torch.float32))
                boot = scn.min()
            y = torch.tensor(float(r)) + gdt * boot
        losses.append(nn.functional.smooth_l1_loss(q, y.detach()))
    loss = torch.stack(losses).mean()
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()
    return float(loss)


def _eval(net, norm, cells, seeds_map):
    """학습 net 을 run_joint_episode 로 평가 (SF/교사와 동일 하네스). guard 포함."""
    rows = []
    for cell in cells:
        for s in seeds_map[cell]:
            pol = RLPolicy(net, norm)
            row = run_joint_episode(_sim(cell, s), pol, RC, generator=CandidateGenerator())
            healthy = True
            try:
                assert_healthy_action_mix(row["_mix"], label=f"{cell}/RL/s{s}")
            except ActionMixError:
                healthy = False
            rows.append({"cell": cell, "seed": s, "berth": row["berth_overrun_min"],
                         "mean_wait": row["mean_wait_min"], "p95": row["p95_wait_min"],
                         "num": row["total_cost"], "completion": row["completion_rate"],
                         "healthy": healthy,
                         "repo": row["action_mix"]["shares"].get("REPOSITION", 0.0)})
    return rows


def run(out=OUT, episodes=200, seeds_per_cell=4, batch=32, lr=1e-3,
        target_sync=10, eps0=1.0, eps_min=0.1) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    prof = build_calibrated_profile()
    norm, _ = fit_state_norm(prof, calibrated_load_params("high", vessel_deadline_mult=0.5),
                             [BASE["high-tight"] + i for i in range(5)], progress=lambda *_: None)
    rng = random.Random(88_000)
    torch.manual_seed(88_000)
    train_cells = list(CELLS)
    train_seeds = {c: [BASE[c] + i for i in range(seeds_per_cell)] for c in train_cells}
    val_seeds = {c: [BASE[c] + 50 + i for i in range(2)] for c in train_cells}
    replay = deque(maxlen=20_000)
    net = target = opt = None
    hist, best = [], {"val": float("inf"), "state": None, "ep": 0}
    t0 = time.perf_counter()
    for ep in range(1, episodes + 1):
        eps = max(eps_min, eps0 * (1.0 - ep / episodes))
        cell = train_cells[ep % len(train_cells)]
        seed = train_seeds[cell][rng.randrange(seeds_per_cell)]
        trans, st = collect_episode(cell, seed, net, norm, eps if net else 1.0, rng)
        if net is None and trans:                       # 첫 조합폭에서 net 생성
            in_dim = len(trans[0][0][0])
            net = JointPairNet(in_dim); target = copy.deepcopy(net)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
        replay.extend(trans)
        if net is not None and len(replay) >= batch:
            for _ in range(max(1, len(trans) // batch)):
                loss = train_step(net, target, opt, rng.sample(replay, batch))
            if ep % target_sync == 0:
                target.load_state_dict(net.state_dict())
        row = {"ep": ep, "eps": round(eps, 3), "cell": cell,
               "compl": round(st["completion"], 3), "replay": len(replay)}
        if net is not None and ep % 20 == 0:
            net.eval()
            ev = _eval(net, norm, train_cells, val_seeds)
            net.train()
            vnum = round(fmean(r["num"] for r in ev), 2)
            vwait = round(fmean(r["mean_wait"] for r in ev), 3)
            vberth = round(fmean(r["berth"] for r in ev), 2)
            healthy_rate = fmean(1.0 if r["healthy"] else 0.0 for r in ev)
            compl_rate = fmean(r["completion"] for r in ev)
            row.update(val_num=vnum, val_wait=vwait, val_berth=vberth,
                       val_healthy=round(healthy_rate, 2), val_compl=round(compl_rate, 3))
            # 체크포인트: **참 목적**(트럭대기+본선 berth, RC-무관 고정척도)로 선택 — shaping RC 로
            # val_num 의미가 바뀌어도 견고. berth 0.3 가중(트럭-min 과 균형). 완주·건전 비례 페널티.
            score = vwait + 0.3 * vberth + 300.0 * (1.0 - compl_rate) + 100.0 * (1.0 - healthy_rate)
            if score < best["val"]:
                best = {"val": score, "num": vnum, "state": copy.deepcopy(net.state_dict()),
                        "ep": ep, "healthy": round(healthy_rate, 2), "compl": round(compl_rate, 3)}
            print(f"[ep{ep}] eps={eps:.2f} val_num={vnum} wait={vwait} berth={vberth} "
                  f"healthy={healthy_rate:.2f} compl={compl_rate:.2f} replay={len(replay)}", flush=True)
        hist.append(row)
    if best["state"] is not None:
        net.load_state_dict(best["state"])
    net.eval()
    torch.save({"fmt": "yard-rl-joint-rl-v1", "state": net.state_dict(), "in_dim": net.in_dim,
                "norm_refs": norm.refs, "best_ep": best["ep"]}, out / "rl_net.pt")
    # 최종 test 평가 (train 밖 seed) + SF 대조
    test_seeds = {c: [BASE[c] + 100 + i for i in range(6)] for c in train_cells}
    rl_rows = _eval(net, norm, train_cells, test_seeds)
    sf_rows = []
    for cell in train_cells:
        for s in test_seeds[cell]:
            row = run_joint_episode(_sim(cell, s), ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                                    RC, generator=CandidateGenerator())
            sf_rows.append({"cell": cell, "seed": s, "berth": row["berth_overrun_min"],
                            "mean_wait": row["mean_wait_min"], "p95": row["p95_wait_min"],
                            "num": row["total_cost"], "completion": row["completion_rate"]})
    res = {"best_ep": best["ep"], "best_val_num": best.get("num", best["val"]),
           "best_healthy": best.get("healthy"), "best_compl": best.get("compl"),
           "wall_s": round(time.perf_counter() - t0), "cells": {}}
    print("\n=== 최종 test (RL vs SF, num=기준재총비용 낮을수록↑) ===", flush=True)
    for cell in train_cells:
        rl = [r for r in rl_rows if r["cell"] == cell]
        sf = [r for r in sf_rows if r["cell"] == cell]
        g = {"rl_num": round(fmean(r["num"] for r in rl), 2), "sf_num": round(fmean(r["num"] for r in sf), 2),
             "rl_berth": round(fmean(r["berth"] for r in rl), 1), "sf_berth": round(fmean(r["berth"] for r in sf), 1),
             "rl_wait": round(fmean(r["mean_wait"] for r in rl), 2), "sf_wait": round(fmean(r["mean_wait"] for r in sf), 2),
             "rl_healthy": all(r["healthy"] for r in rl), "rl_compl": all(r["completion"] == 1.0 for r in rl)}
        res["cells"][cell] = g
        print(f"  {cell:11s} num RL {g['rl_num']:7.2f} vs SF {g['sf_num']:7.2f} | "
              f"berth {g['rl_berth']:6.1f}/{g['sf_berth']:6.1f} wait {g['rl_wait']:5.2f}/{g['sf_wait']:5.2f} "
              f"| healthy={g['rl_healthy']} compl={g['rl_compl']}", flush=True)
    (out / "results.json").write_text(json.dumps({"res": res, "hist": hist}, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"\nbest_ep={best['ep']} wall={res['wall_s']}s\nDONE", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seeds-per-cell", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    run(Path(a.out), episodes=a.episodes, seeds_per_cell=a.seeds_per_cell)
