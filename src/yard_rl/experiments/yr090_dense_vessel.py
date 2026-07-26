"""YR-090 — 본선 점증 학습신호 (dense potential-based shaping) 판가름 실험.

■ 사전등록 (2026-07-26, 결과 미열람 동결)
- 가설: 본선 학습이 시드 운인 주범 = **본선 보상의 희소·지연**(vessel_delay 가 선박 완료 시
  lump — 트럭 시간적분 dense 와 비대칭, 코드 확인). 신호를 촘촘하게 하면 재현된다.
- 처방: 잠재함수 Φ(s) = (ρ_vessel/3600)·Σ_미완선박 expected_delay_s(now) —
  **정책불변 potential shaping** F = γ^Δt·Φ(s′) − Φ(s) 를 **훈련 보상에만** 가산.
  Φ 는 lump 를 시간축으로 **재분배**(완료 순간 Φ 가 실현지연만큼 떨어져 lump 와 상쇄 —
  telescoping)라 이중계상·트럭 재희생 아님. 평가는 순수 numeraire_v1 (shaping 無).
- arm 2 (유일 차이 = Φ 항): DENSE(레시피+Φ) vs CONTROL(기존 최고 레시피 그대로).
  레시피 = 보상 r/20 스케일·Double-DQN·Polyak τ=.005·EMA τ=.01·점증 트럭비용 RAMP=1.0·
  500ep·시나리오 **time_contract_v2=True**(YR-089 계약)·물리 정정 후(YR-091/092).
- 학습시드 3 (88000·99000·123000) × arm 2 = 6 훈련. **주판정 = 재현성**:
  DENSE 가 3/3 시드에서 berth Δ(vs SF, 평가 +500 대역 high-loose/high-tight×10, paired CI)
  상한 < 0 이고 CONTROL 은 3/3 실패(기왕 확립: 시드 운)면 가설 확정. 2/3 = 부분 지지.
  부지표(보고): 트럭 평균·P95·완주(전 arm 1.0 요구)·healthy.
- 기대치 명시: 목표는 **본선 학습의 재현성**이지 계획법(rollout) 추월이 아님.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import random
import time
from collections import deque
from pathlib import Path
from statistics import fmean, stdev

import torch
from torch import nn

from ..domain.enums import InformationLevel, JobStatus
from ..integrated import TerminalSimulator
from ..integrated.baselines import (ActionMixError, JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, _apply, _wait_of,
                                    assert_healthy_action_mix, run_joint_episode)
from ..contract import CandidateKind
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.encoding import StateNorm
from ..integrated.joint_distill import JointPairNet
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr059_state_norm import fit_state_norm
from .yr088_joint_rl import (FORBID_WAIT, GAMMA, LEVEL, RC as RC_TRAIN, REF_S, REPO_PENALTY,
                             RLPolicy, UNSERVED, build_rows)

RC_EVAL = RewardCalculator.numeraire_v1()
RHO_VESSEL, SCALE = 33.0, 20.0
WARN_FRAC, RAMP = 0.5, 1.0            # 점증 트럭비용 (채택 레시피, YR-088 정책최적화 §5)
CELLS = {"mid-loose": ("mid", 2.0), "high-loose": ("high", 2.0),
         "mid-tight": ("mid", 0.5), "high-tight": ("high", 0.5)}
BASE = {"mid-loose": 830000, "high-loose": 830100, "mid-tight": 830200, "high-tight": 830300}
EVAL_CELLS = ("high-loose", "high-tight")
TRAIN_SEEDS = (88_000, 99_000, 123_000)
OUT = Path("outputs/reports/yr090_dense_vessel")
_TC = {19: 2.093}


def _sim(cell: str, seed: int) -> TerminalSimulator:
    prof = build_calibrated_profile()
    lvl, dm = CELLS[cell]
    p = dataclasses.replace(calibrated_load_params(lvl, vessel_deadline_mult=dm),
                            time_contract_v2=True)
    s = TerminalSimulator(prof, generate_terminal_scenario(prof, seed, p), check_invariants=True)
    s.info_level = LEVEL
    return s


# ---------------------------------------------------------------- shaping
def vessel_potential(sim) -> float:
    """Φ(s) — 미완 선박의 예상 지연(초) 합 × ρ/3600 (numeraire 단위). 관측가능 계획값만
    사용(planned_completion·remaining_moves·cadence — VesselProcess.expected_delay_s)."""
    tot = 0.0
    for v in sim.vessels.values():
        if v.done:
            continue
        d = v.expected_delay_s(sim.now)
        if d:
            tot += d
    return RHO_VESSEL * tot / 3600.0


def graduated_wait_shaping(sim, dt_s: float) -> float:
    """트럭 점증비용 (채택 — 대기율 f>0.5 convex 램프, SLA 서 시간당 +RAMP)."""
    sla = sim.profile.long_wait_sla_s
    tot = 0.0
    for jid, j in sim.jobs.items():
        if j.is_external_truck and j.status == JobStatus.WAITING:
            f = sim.cum_wait(jid) / sla
            if f > WARN_FRAC:
                tot += ((f - WARN_FRAC) / (1.0 - WARN_FRAC)) ** 2
    return RAMP * tot * (dt_s / 3600.0)


def collect_episode(cell, seed, net, norm, epsilon, rng, *, dense_vessel: bool):
    """yr088 collect + 점증 트럭비용 + (DENSE arm) 본선 잠재함수 shaping."""
    sim = _sim(cell, seed)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RC_TRAIN, horizon_s=1800.0, generator=gen,
                            forbid_strategic_wait=FORBID_WAIT)
    trans, k = [], 0
    dp = sim.run_until_decision()
    sim.cost.cut()
    last_b = sim.now
    pend = None
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        rows, assigns = build_rows(sim, dp, gen_by, norm, jr, k)
        raw = sim.cost.cut()
        if pend is not None:
            pend["r"] += RC_TRAIN.cost_for(interval_start_s=last_b, interval_end_s=sim.now,
                                           raw=raw, risk_max=0.0).total_normalized
            pend["r"] += graduated_wait_shaping(sim, sim.now - last_b)
        last_b = sim.now
        if pend is not None and assigns:
            gdt = GAMMA ** ((sim.now - pend["t_act"]) / REF_S)
            if dense_vessel:                       # F = γ^Δt·Φ(s′) − Φ(s) — 정책불변
                pend["r"] += gdt * vessel_potential(sim) - pend["phi"]
            trans.append([pend["rows"], pend["pos"], pend["r"], gdt, rows])
            pend = None
        if not assigns:
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
            pend = {"rows": rows, "pos": pick, "t_act": sim.now,
                    "r": REPO_PENALTY * n_repo, "phi": vessel_potential(sim)}
            _apply(sim, assigns[pick])
        dp = sim.run_until_decision()
        k += 1
    jobs = list(sim.jobs.values())
    n_unserved = sum(1 for j in jobs if j.status.name != "DONE")
    if pend is not None:
        raw = sim.cost.cut()
        pend["r"] += RC_TRAIN.cost_for(interval_start_s=last_b, interval_end_s=sim.now,
                                       raw=raw, risk_max=0.0).total_normalized
        pend["r"] += graduated_wait_shaping(sim, sim.now - last_b)
        if dense_vessel:
            pend["r"] += vessel_potential(sim) - pend["phi"]
        pend["r"] += UNSERVED * n_unserved
        trans.append([pend["rows"], pend["pos"], pend["r"], 1.0, None])
    return trans, {"completion": sum(1 for j in jobs if j.status.name == "DONE") / len(jobs)}


# ---------------------------------------------------------------- train (grad4 레시피)
def _soft(dst, src, tau):
    with torch.no_grad():
        for dp_, sp_ in zip(dst.parameters(), src.parameters()):
            dp_.mul_(1 - tau).add_(tau * sp_.data)


def _train_step(net, target, opt, batch):
    losses = []
    for rows_k, pos_k, r, gdt, rows_next in batch:
        sc, _ = net(torch.tensor(rows_k, dtype=torch.float32))
        q = sc[pos_k]
        if rows_next is None:
            y = torch.tensor(r / SCALE)
        else:
            with torch.no_grad():
                on, _ = net(torch.tensor(rows_next, dtype=torch.float32))
                tg, _ = target(torch.tensor(rows_next, dtype=torch.float32))
                y = torch.tensor(r / SCALE) + gdt * tg[int(torch.argmin(on))]
        losses.append(nn.functional.smooth_l1_loss(q, y.detach()))
    loss = torch.stack(losses).mean()
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()


def _val(net, norm, cells, seeds_map):
    rows = []
    for cell in cells:
        for s in seeds_map[cell]:
            r = run_joint_episode(_sim(cell, s), RLPolicy(net, norm), RC_TRAIN,
                                  generator=CandidateGenerator())
            healthy = True
            try:
                assert_healthy_action_mix(r["_mix"], label="val")
            except ActionMixError:
                healthy = False
            rows.append((r["mean_wait_min"], r["berth_overrun_min"], r["completion_rate"], healthy))
    return rows


def train_one(arm: str, base_seed: int, episodes=500, spc=16, batch=64, lr=5e-4) -> Path:
    dense = arm == "DENSE"
    out = OUT / f"{arm.lower()}_s{base_seed}"
    out.mkdir(parents=True, exist_ok=True)
    prof = build_calibrated_profile()
    norm, _ = fit_state_norm(
        prof, dataclasses.replace(calibrated_load_params("high", vessel_deadline_mult=0.5),
                                  time_contract_v2=True),
        [BASE["high-tight"] + i for i in range(5)], progress=lambda *_: None)
    rng = random.Random(base_seed); torch.manual_seed(base_seed)
    cells = list(CELLS)
    tr = {c: [BASE[c] + i for i in range(spc)] for c in cells}
    va = {c: [BASE[c] + 50 + i for i in range(4)] for c in cells}
    replay = deque(maxlen=40_000)
    net = target = ema = opt = None
    best = {"val": float("inf"), "state": None, "ep": 0}
    for ep in range(1, episodes + 1):
        eps = max(0.05, 1.0 - ep / episodes)
        cell = cells[ep % len(cells)]
        trans, _st = collect_episode(cell, tr[cell][rng.randrange(spc)], net, norm,
                                     eps if net else 1.0, rng, dense_vessel=dense)
        if net is None and trans:
            net = JointPairNet(len(trans[0][0][0]))
            target, ema = copy.deepcopy(net), copy.deepcopy(net)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
        replay.extend(trans)
        if net is not None and len(replay) >= batch:
            for _ in range(max(1, len(trans) // batch)):
                _train_step(net, target, opt, rng.sample(replay, batch))
                _soft(target, net, 0.005); _soft(ema, net, 0.01)
        if net is not None and ep % 25 == 0:
            ema.eval(); rows = _val(ema, norm, cells, va); ema.train()
            w = fmean(r[0] for r in rows); b = fmean(r[1] for r in rows)
            compl = fmean(r[2] for r in rows); hl = fmean(1.0 if r[3] else 0.0 for r in rows)
            score = w + 0.3 * b + 300.0 * (1 - compl) + 100.0 * (1 - hl)
            if score < best["val"]:
                best = {"val": score, "state": copy.deepcopy(ema.state_dict()), "ep": ep}
            print(f"[{arm} s{base_seed} ep{ep}] wait={w:.2f} berth={b:.1f} "
                  f"healthy={hl:.2f} compl={compl:.3f}", flush=True)
    ema.load_state_dict(best["state"]); ema.eval()
    torch.save({"state": ema.state_dict(), "in_dim": ema.in_dim, "norm_refs": norm.refs,
                "best_ep": best["ep"], "arm": arm, "train_seed": base_seed}, out / "rl_net.pt")
    return out / "rl_net.pt"


# ---------------------------------------------------------------- eval (순수 numeraire·paired)
def _ci(d):
    m, sd, n = fmean(d), stdev(d), len(d); se = sd / n ** 0.5
    return round(m, 2), round(m - _TC.get(n - 1, 2.1) * se, 2), round(m + _TC.get(n - 1, 2.1) * se, 2)


def evaluate(ck_path: Path, sf_cache: dict) -> dict:
    ck = torch.load(ck_path, map_location="cpu")
    net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
    norm = StateNorm(refs=ck["norm_refs"])
    db, dw, dp95, compl = [], [], [], []
    for cell in EVAL_CELLS:
        for i in range(10):
            seed = BASE[cell] + 500 + i
            key = (cell, seed)
            if key not in sf_cache:
                sf_cache[key] = run_joint_episode(
                    _sim(cell, seed), ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                    RC_EVAL, generator=CandidateGenerator())
            s = sf_cache[key]
            r = run_joint_episode(_sim(cell, seed), RLPolicy(net, norm), RC_EVAL,
                                  generator=CandidateGenerator())
            db.append(r["berth_overrun_min"] - s["berth_overrun_min"])
            dw.append(r["mean_wait_min"] - s["mean_wait_min"])
            dp95.append(r["p95_wait_min"] - s["p95_wait_min"])
            compl.append(r["completion_rate"])
    return {"berth_ci": _ci(db), "wait_ci": _ci(dw), "p95_ci": _ci(dp95),
            "compl_min": min(compl), "best_ep": ck["best_ep"]}


def run(episodes=500):
    OUT.mkdir(parents=True, exist_ok=True)
    sf_cache: dict = {}
    res: dict = {}
    for arm in ("DENSE", "CONTROL"):
        for ts in TRAIN_SEEDS:
            t0 = time.perf_counter()
            ck = train_one(arm, ts, episodes=episodes)
            ev = evaluate(ck, sf_cache)
            ev["train_wall_s"] = round(time.perf_counter() - t0)
            res[f"{arm}_s{ts}"] = ev
            print(f"== {arm} s{ts}: berth {ev['berth_ci']} wait {ev['wait_ci']} "
                  f"p95 {ev['p95_ci']} compl_min {ev['compl_min']} best_ep {ev['best_ep']}",
                  flush=True)
    # 사전등록 판정: DENSE 3/3 시드 berth CI 상한<0 & CONTROL 은 그렇지 않음
    dense_ok = sum(1 for ts in TRAIN_SEEDS if res[f"DENSE_s{ts}"]["berth_ci"][2] < 0)
    ctrl_ok = sum(1 for ts in TRAIN_SEEDS if res[f"CONTROL_s{ts}"]["berth_ci"][2] < 0)
    res["_verdict"] = {"dense_significant_seeds": dense_ok, "control_significant_seeds": ctrl_ok,
                       "prereg": "DENSE 3/3 & CONTROL<3 → 가설 확정 / DENSE 2/3 → 부분 지지 / "
                                 "그 외 → 기각(희소보상 단독 원인 아님)"}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\nVERDICT: DENSE {dense_ok}/3 · CONTROL {ctrl_ok}/3 유의 | saved {OUT/'results.json'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    a = ap.parse_args()
    run(a.episodes)
    print("YR090 DONE")
