"""YR-088 일반화 — 학습된 RL 정책을 **재학습 없이(zero-shot)** 새 환경에 적용.

RL 로 바꾼 근본 이유 = 지도학습(증류)은 새 환경 일반화 불가. 이 테스트가 그 전제를 검증:
학습 RL(rl_net.pt)을 학습 밖 조건·다른 터미널에 그대로 굴려 SF 와 대조.
- Tier1(안 배운 조건): current 부하(40대)·마감 mult 1.0·0.3 (학습은 mid/high·2.0/0.5만).
- Tier2(다른 터미널): 충실(faithful) 등록 터미널 프로파일 (학습은 calibrated 표준만).
크레인 슬롯은 **프로파일에서 동적**으로 잡아 프로파일 호환. 판정 = berth·트럭대기·완주·건전.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, stdev

import torch

import yard_rl.experiments.yr088_joint_rl as base
from ..integrated import TerminalSimulator
from ..integrated.adapter import capture
from ..integrated.baselines import (ActionMixError, JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, _feasible_joint, _wait_of,
                                    assert_healthy_action_mix, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.encoding import StateNorm, encode_observation
from ..integrated.joint_distill import JointPairNet
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario

LEVEL = base.LEVEL
RC = base.RC
OUT = Path("outputs/reports/yr088_generalize")


def build_rows_slots(sim, dp, gen_by, norm, jr, slots):
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "gen", 0, generator=jr.generator)
    encs = {ob.crane_id: encode_observation(state, ob, norm=norm) for ob in obs}
    ca, cb = slots
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


class GenRLPolicy:
    def __init__(self, net, norm, slots):
        self.net, self.norm, self.slots, self.name = net, norm, slots, "RL"
        self.jr = JointRolloutGreedy(RC, horizon_s=1800.0, generator=CandidateGenerator(),
                                     forbid_strategic_wait=base.FORBID_WAIT)

    def decide(self, sim, dp, gen_by):
        rows, assigns = build_rows_slots(sim, dp, gen_by, self.norm, self.jr, self.slots)
        if not assigns:
            return {c: _wait_of(gen_by[c]) for c in dp.crane_ids}
        with torch.no_grad():
            sc, _ = self.net(torch.tensor(rows, dtype=torch.float32))
        return assigns[int(torch.argmin(sc))]


def _mk_sim(profile, level, mult, seed):
    s = TerminalSimulator(profile, generate_terminal_scenario(
        profile, seed, calibrated_load_params(level, vessel_deadline_mult=mult)),
        check_invariants=True)
    s.info_level = LEVEL
    return s


def eval_env(net, norm, profile, level, mult, seeds):
    slots = tuple(sorted(c.crane_id for c in profile.cranes))
    out = {"RL": [], "SF": []}
    for s in seeds:
        for arm, pol in [("RL", GenRLPolicy(net, norm, slots)),
                         ("SF", ResolverPolicy(ServiceFirstSPTPreference(), "SF"))]:
            try:
                row = run_joint_episode(_mk_sim(profile, level, mult, s), pol, RC,
                                        generator=CandidateGenerator())
            except Exception as e:                      # 프로파일 비호환 등 — 정직히 기록
                out[arm].append({"seed": s, "error": str(e)[:80]})
                continue
            healthy = True
            try:
                assert_healthy_action_mix(row["_mix"], label=arm)
            except ActionMixError:
                healthy = False
            out[arm].append({"seed": s, "berth": row["berth_overrun_min"],
                             "wait": row["mean_wait_min"], "num": row["total_cost"],
                             "completion": row["completion_rate"], "healthy": healthy})
    return out


def _agg(rows, k):
    v = [r[k] for r in rows if k in r]
    return round(fmean(v), 2) if v else None


def run(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    d = torch.load("outputs/reports/yr088_joint_rl/rl_net.pt", weights_only=False)
    net = JointPairNet(d["in_dim"]); net.load_state_dict(d["state"]); net.eval()
    norm = StateNorm(refs=d["norm_refs"], basis="fitted_baseline_p90")
    cal = build_calibrated_profile()
    seeds = [850000 + i for i in range(5)]

    envs = [   # (라벨, 프로파일, level, mult, 안배운축)
        ("current-loose", cal, "current", 2.0, "부하40(안배움)"),
        ("current-tight", cal, "current", 0.5, "부하40(안배움)"),
        ("mid-mid", cal, "mid", 1.0, "마감1.0(안배움)"),
        ("high-vtight", cal, "high", 0.3, "마감0.3(안배움)"),
    ]
    # Tier2: 다른 터미널 프로파일 (충실 등록군, 학습 profile 과 다름)
    try:
        from ..integrated.terminal_registry import build_stress_profile, faithful_terminals
        for tid in faithful_terminals()[:2]:
            se = build_stress_profile(tid)
            if len(se.profile.cranes) == 2:
                envs.append((f"terminal:{tid}", se.profile, "mid", 2.0, "다른터미널"))
    except Exception as e:
        print(f"[warn] 터미널 프로파일 로드 실패: {str(e)[:80]}", flush=True)

    res = {"envs": {}}
    print("=== zero-shot 일반화 (학습 RL vs SF, 재학습 없음) ===", flush=True)
    for label, prof, level, mult, axis in envs:
        ev = eval_env(net, norm, prof, level, mult, seeds)
        rl, sf = ev["RL"], ev["SF"]
        g = {"axis": axis,
             "rl_berth": _agg(rl, "berth"), "sf_berth": _agg(sf, "berth"),
             "rl_wait": _agg(rl, "wait"), "sf_wait": _agg(sf, "wait"),
             "rl_num": _agg(rl, "num"), "sf_num": _agg(sf, "num"),
             "rl_compl": _agg(rl, "completion"), "rl_healthy_rate": _agg(rl, "healthy"),
             "errors": sum(1 for r in rl if "error" in r)}
        res["envs"][label] = g
        vs = ("num RL {rl_num}/SF {sf_num} · berth {rl_berth}/{sf_berth} · "
              "wait {rl_wait}/{sf_wait} · compl {rl_compl} healthy {rl_healthy_rate}").format(**g)
        print(f"  [{label:16s}] {axis:14s} {vs} err={g['errors']}", flush=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print("\nDONE", flush=True)
    return res


if __name__ == "__main__":
    run()
