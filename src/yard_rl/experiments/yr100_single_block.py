"""YR-100 [2] — 단일 블록 검증: 계산비용(잔여 분해) 주입 CALC vs CONTROL.

■ 설계 (사전등록 성격 — 결과 미열람 작성)
- CALC arm: ExecutionQ 를 잔여 분해 — 행동선택 argmin_a [c(a)/SCALE + Q_res(s,a)],
  TD 표적 y = (r − c(a))/SCALE + γ^Δt · [c(a')/SCALE + Q_res_target(s',a')]|a*=온라인선택.
  c(a) = candidate_vessel_delta (YR-100 공식, LOAD 전용 계산 — 학습 아님).
  본선 긴급도 feature 를 학습입력에 재유입하지 않는다(스펙 가드) — c 는 숫자로만 주입.
- CONTROL: yr090 CONTROL 레시피 그대로 (기학습 checkpoint 재사용 — 동일 코드상태:
  물리정정 YR-091/092 + time_contract_v2 이후 학습).
- 검증 3종 (spec 검증·중단 조건):
  (a) 단조성 — 결정 표본에서 LOAD slack 버킷별 본선서빙 선택률 단조 증가 (probe).
  (b) 트럭 무회귀 — CALC vs CONTROL paired 트럭 평균·P95 (하드게이트).
  (c) 반사실 일치 — 본선/비본선 갈림 결정에서 SF 연속 rollout 실현비용과 선택 일치율.
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

from ..integrated.baselines import (ActionMixError, JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, _apply, _feasible_joint,
                                    _wait_of, assert_healthy_action_mix, run_joint_episode)
from ..contract import CandidateKind
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.encoding import StateNorm
from ..integrated.joint_distill import JointPairNet
from ..integrated.vessel import VesselWorkType
from ..integrated.vessel_cost import candidate_vessel_delta
from ..integrated.vessel_signals import schedule_slack_s
from .yr059_state_norm import fit_state_norm
from .yr088_joint_rl import FORBID_WAIT, GAMMA, LEVEL, RC as RC_TRAIN, REF_S, REPO_PENALTY, UNSERVED
from .yr090_dense_vessel import (BASE, CELLS, SCALE, TRAIN_SEEDS, _ci, _sim, _soft,
                                 graduated_wait_shaping)

RC_EVAL = RewardCalculator.numeraire_v1()
OUT = Path("outputs/reports/yr100_single_block")
YR090 = Path("outputs/reports/yr090_dense_vessel")


# ---------------------------------------------------------------- residual 전용 마스크
# 스펙 가드(YR-100 §받는다 vs 학습 안 한다): 잔여망 입력에서 본선 긴급도 feature 를 제거 —
# 본선 그룹 전체 + candidate(is_vessel·deadline_slack_s·vessel_risk_delta) +
# queue(vessel_urgency_max) + global(sts_wait_accum_s). 인덱스는 스키마에서 유도(하드코딩 금지).
from ..contract import SCHEMA  # noqa: E402

_BAN_FIELDS = {"candidate": ("is_vessel", "deadline_slack_s", "vessel_risk_delta"),
               "queue": ("vessel_urgency_max",), "global": ("sts_wait_accum_s",)}


def _grp_ban(group: str) -> list[int]:
    specs = SCHEMA.group_specs(group)
    f = len(specs)
    out = []
    for i, sp in enumerate(specs):
        if sp.name in _BAN_FIELDS.get(group, ()):
            out.extend((i, f + i))                    # value + known 지시자 둘 다
    return out


def _ban_indices(g2: int, v2: int, yc2: int, q2: int, c2: int) -> tuple[int, ...]:
    ban: list[int] = []
    ban.extend(_grp_ban("global"))                                   # g @0
    ban.extend(range(g2, g2 + v2))                                   # vessel 블록 전체
    qa = g2 + v2 + yc2
    ban.extend(qa + i for i in _grp_ban("queue"))                    # queue_a
    ca = qa + q2
    ban.extend(ca + i for i in _grp_ban("candidate"))                # cand_a
    qb = ca + c2 + yc2
    ban.extend(qb + i for i in _grp_ban("queue"))                    # queue_b
    cb = qb + q2
    ban.extend(cb + i for i in _grp_ban("candidate"))                # cand_b
    return tuple(ban)


_BAN_CACHE: dict[tuple, tuple[int, ...]] = {}


def mask_rows(rows, ref, ea, eb):
    """잔여망 입력 마스크 — 금지 feature 위치를 0 으로. 부재 assert (스펙 가드)."""
    any_e = ea or eb
    dims = (len(ref.g), len(ref.vessel), len(any_e.yc), len(any_e.queue), len(ref.cand[0]))
    ban = _BAN_CACHE.get(dims)
    if ban is None:
        ban = _BAN_CACHE[dims] = _ban_indices(*dims)
    out = []
    for r in rows:
        r2 = list(r)
        for i in ban:
            r2[i] = 0.0
        out.append(r2)
    if out:
        assert all(out[0][i] == 0.0 for i in ban)     # 금지 feature 부재 assert
    return out


def build_rows_cvec(sim, dp, gen_by, norm, jr, k):
    """yr088.build_rows 와 동일 행구성 + 잔여망 마스크 + 조합별 c(a) — capture 1회.

    (yr088:75-96 로컬 복제 — 마스크를 행 조립 시점에 적용하기 위함. 계약: 행 레이아웃
    ctx_a+blk_a+ctx_b+blk_b 동일, 차이는 금지 feature 0 마스크와 cvec 뿐.)"""
    from ..integrated.adapter import capture
    from ..integrated.encoding import encode_observation
    from .yr088_joint_rl import SLOTS
    from ..integrated.baselines import _feasible_joint
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
    if rows:
        rows = mask_rows(rows, ref, ea, eb)
    cvec = [candidate_vessel_delta(sim, a) for a in assigns]
    return rows, assigns, cvec


def _pick_calc(net, rows, cvec):
    with torch.no_grad():
        sc, _ = net(torch.tensor(rows, dtype=torch.float32))
    tot = sc + torch.tensor(cvec, dtype=torch.float32) / SCALE
    return int(torch.argmin(tot))


def collect_episode(cell, seed, net, norm, epsilon, rng):
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
        rows, assigns, cvec = build_rows_cvec(sim, dp, gen_by, norm, jr, k)
        raw = sim.cost.cut()
        if pend is not None:
            pend["r"] += RC_TRAIN.cost_for(interval_start_s=last_b, interval_end_s=sim.now,
                                           raw=raw, risk_max=0.0).total_normalized
            pend["r"] += graduated_wait_shaping(sim, sim.now - last_b)
        last_b = sim.now
        if pend is not None and assigns:
            gdt = GAMMA ** ((sim.now - pend["t_act"]) / REF_S)
            trans.append([pend["rows"], pend["pos"], pend["r"] - pend["c"], gdt, rows, cvec])
            pend = None
        if not assigns:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        else:
            if net is not None and rng.random() >= epsilon:
                pick = _pick_calc(net, rows, cvec)
            else:
                pick = rng.randrange(len(assigns))
            n_repo = sum(1 for c in dp.crane_ids
                         if assigns[pick][c].kind == CandidateKind.REPOSITION)
            pend = {"rows": rows, "pos": pick, "t_act": sim.now,
                    "r": REPO_PENALTY * n_repo, "c": cvec[pick]}
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
        pend["r"] += UNSERVED * n_unserved
        trans.append([pend["rows"], pend["pos"], pend["r"] - pend["c"], 1.0, None, None])
    return trans, {"completion": sum(1 for j in jobs if j.status.name == "DONE") / len(jobs)}


def _train_step(net, target, opt, batch):
    losses = []
    for rows_k, pos_k, r_res, gdt, rows_next, cvec_next in batch:
        sc, _ = net(torch.tensor(rows_k, dtype=torch.float32))
        q = sc[pos_k]
        if rows_next is None:
            y = torch.tensor(r_res / SCALE)
        else:
            with torch.no_grad():
                cn = torch.tensor(cvec_next, dtype=torch.float32) / SCALE
                on, _ = net(torch.tensor(rows_next, dtype=torch.float32))
                tg, _ = target(torch.tensor(rows_next, dtype=torch.float32))
                a_star = int(torch.argmin(on + cn))
                y = torch.tensor(r_res / SCALE) + gdt * (tg[a_star] + cn[a_star])
        losses.append(nn.functional.smooth_l1_loss(q, y.detach()))
    loss = torch.stack(losses).mean()
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()


class CalcPolicy:
    """argmin[c/SCALE + Q_res] 실행 정책. probe=list 면 (slack_min, has_vessel, chose) 기록."""

    def __init__(self, net, norm, probe=None):
        self.net, self.norm, self.probe = net, norm, probe
        self.name = "CALC"
        self.jr = JointRolloutGreedy(RC_TRAIN, horizon_s=1800.0, generator=CandidateGenerator(),
                                     forbid_strategic_wait=FORBID_WAIT)

    def decide(self, sim, dp, gen_by):
        rows, assigns, cvec = build_rows_cvec(sim, dp, gen_by, self.norm, self.jr, 0)
        if not assigns:
            return {c: _wait_of(gen_by[c]) for c in dp.crane_ids}
        pick = _pick_calc(self.net, rows, cvec)
        if self.probe is not None:
            slacks = [schedule_slack_s(v, sim.now) for v in sim.vessels.values()
                      if v.work_type == VesselWorkType.LOAD and not v.done]
            slacks = [s for s in slacks if s is not None]
            has_v = any(c < -1e-9 for c in cvec)
            if slacks and has_v:
                self.probe.append((min(slacks), cvec[pick] < -1e-9))
        return assigns[pick]


def train_one(base_seed: int, episodes=500, spc=16, batch=64, lr=5e-4) -> Path:
    out = OUT / f"calc_s{base_seed}"
    out.mkdir(parents=True, exist_ok=True)
    import dataclasses
    from ..integrated.profiles import build_calibrated_profile
    from ..integrated.scenario_gen import calibrated_load_params
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
                                     eps if net else 1.0, rng)
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
            ema.eval()
            rows = []
            for cell_ in cells:
                for s in va[cell_]:
                    r = run_joint_episode(_sim(cell_, s), CalcPolicy(ema, norm), RC_TRAIN,
                                          generator=CandidateGenerator())
                    healthy = True
                    try:
                        assert_healthy_action_mix(r["_mix"], label="val")
                    except ActionMixError:
                        healthy = False
                    rows.append((r["mean_wait_min"], r["berth_overrun_min"],
                                 r["completion_rate"], healthy))
            ema.train()
            w = fmean(r[0] for r in rows); b = fmean(r[1] for r in rows)
            compl = fmean(r[2] for r in rows); hl = fmean(1.0 if r[3] else 0.0 for r in rows)
            score = w + 0.3 * b + 300.0 * (1 - compl) + 100.0 * (1 - hl)
            if score < best["val"]:
                best = {"val": score, "state": copy.deepcopy(ema.state_dict()), "ep": ep}
            print(f"[CALC s{base_seed} ep{ep}] wait={w:.2f} berth={b:.1f} "
                  f"healthy={hl:.2f} compl={compl:.3f}", flush=True)
    if best["state"] is not None:
        ema.load_state_dict(best["state"])
    ema.eval()
    torch.save({"state": ema.state_dict(), "in_dim": ema.in_dim, "norm_refs": norm.refs,
                "best_ep": best["ep"], "arm": "CALC", "train_seed": base_seed},
               out / "rl_net.pt")
    return out / "rl_net.pt"


# ---------------------------------------------------------------- 검증 (a)(b)(c)
def _load(ck_path: Path):
    ck = torch.load(ck_path, map_location="cpu")
    net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
    return net, StateNorm(refs=ck["norm_refs"])


def behave(train_seed: int, n_eval=10) -> dict:
    """(a) 단조성 probe + (b) CALC vs CONTROL paired 무회귀 + 완주·건전."""
    from .yr088_joint_rl import RLPolicy
    calc_net, calc_norm = _load(OUT / f"calc_s{train_seed}" / "rl_net.pt")
    ctrl_net, ctrl_norm = _load(YR090 / f"control_s{train_seed}" / "rl_net.pt")
    probe: list = []
    dw, dp95, db, compl = [], [], [], []
    for cell in ("high-loose", "high-tight"):
        for i in range(n_eval):
            seed = BASE[cell] + 500 + i
            rc = run_joint_episode(_sim(cell, seed), CalcPolicy(calc_net, calc_norm, probe),
                                   RC_EVAL, generator=CandidateGenerator())
            rk = run_joint_episode(_sim(cell, seed), RLPolicy(ctrl_net, ctrl_norm),
                                   RC_EVAL, generator=CandidateGenerator())
            dw.append(rc["mean_wait_min"] - rk["mean_wait_min"])
            dp95.append(rc["p95_wait_min"] - rk["p95_wait_min"])
            db.append(rc["berth_overrun_min"] - rk["berth_overrun_min"])
            compl.append(rc["completion_rate"])
    buckets = {"<0": [], "0-15m": [], "15-45m": [], ">45m": []}
    for slack, chose in probe:
        k = ("<0" if slack < 0 else "0-15m" if slack < 900 else
             "15-45m" if slack < 2700 else ">45m")
        buckets[k].append(1.0 if chose else 0.0)
    mono = {k: (round(fmean(v), 3), len(v)) for k, v in buckets.items() if v}
    return {"monotonic_pick_rate": mono, "d_wait_ci": _ci(dw), "d_p95_ci": _ci(dp95),
            "d_berth_ci": _ci(db), "compl_min": min(compl)}


def counterfactual(train_seed: int, cell="high-tight", n_samples=12, seed_off=520) -> dict:
    """(c) 갈림 결정: 본선최선 vs 비본선최선 각각 SF 로 에피소드 끝까지 → 실현비용 비교."""
    calc_net, calc_norm = _load(OUT / f"calc_s{train_seed}" / "rl_net.pt")
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RC_TRAIN, horizon_s=1800.0, generator=gen,
                            forbid_strategic_wait=FORBID_WAIT)
    agree, checked = 0, 0
    sim = _sim(cell, BASE[cell] + seed_off)
    dp = sim.run_until_decision()
    while dp is not None and checked < n_samples:
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        rows, assigns, cvec = build_rows_cvec(sim, dp, gen_by, calc_norm, jr, 0)
        if assigns:
            vi = [i for i, c in enumerate(cvec) if c < -0.02]
            ni = [i for i, c in enumerate(cvec) if c >= -1e-9]
            if vi and ni:
                best_v = min(vi, key=lambda i: cvec[i])
                cost_v = _rollforward(sim, assigns[best_v])
                cost_n = _rollforward(sim, assigns[ni[0]])
                pick = _pick_calc(calc_net, rows, cvec)
                policy_vessel = cvec[pick] < -1e-9
                truth_vessel = cost_v < cost_n
                agree += int(policy_vessel == truth_vessel)
                checked += 1
            pick = _pick_calc(calc_net, rows, cvec)
            _apply(sim, assigns[pick])
        else:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        dp = sim.run_until_decision()
    return {"agree": agree, "checked": checked,
            "rate": round(agree / checked, 3) if checked else None}


def _rollforward(sim0, assign) -> float:
    """분기: assign 적용 후 SF 로 끝까지 — 실현 numeraire 총비용 (분기점 이후)."""
    sim = copy.deepcopy(sim0)
    sim.cost.cut()
    _apply(sim, assign)
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gen = CandidateGenerator()
    dp = sim.run_until_decision()
    total, last_b = 0.0, sim.now
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        raw = sim.cost.cut()
        total += RC_EVAL.cost_for(interval_start_s=last_b, interval_end_s=sim.now,
                                  raw=raw, risk_max=0.0).total_normalized
        last_b = sim.now
        _apply(sim, pol.decide(sim, dp, gen_by))
        dp = sim.run_until_decision()
    raw = sim.cost.cut()
    total += RC_EVAL.cost_for(interval_start_s=last_b, interval_end_s=sim.now,
                              raw=raw, risk_max=0.0).total_normalized
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, help="CALC 학습 (train seed)")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--behave", type=int, help="검증 (a)(b) — train seed")
    ap.add_argument("--cf", type=int, help="검증 (c) — train seed")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.train is not None:
        t0 = time.perf_counter()
        p = train_one(a.train, episodes=a.episodes)
        print(f"TRAINED {p} wall={time.perf_counter()-t0:.0f}s")
    if a.behave is not None:
        r = behave(a.behave)
        (OUT / f"behave_s{a.behave}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(json.dumps(r, ensure_ascii=False))
    if a.cf is not None:
        r = counterfactual(a.cf)
        (OUT / f"cf_s{a.cf}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(json.dumps(r, ensure_ascii=False))
    print("DONE")
