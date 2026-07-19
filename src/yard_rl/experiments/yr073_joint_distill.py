"""YR-073 실행 드라이버 — 수집(교사 로깅)→학습 v0→DAgger→학습 v1→locked 평가.

prereg: .claude/docs/strategy-history/2026-07-19-YR-073-순위증류-prereg.md (동결).
실행 환경: WSL (~/.venvs/yard-rl, torch CPU). 출력 outputs/reports/yr073_distill/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import fmean

from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.adapter import capture
from ..integrated.baselines import (ActionMixError, JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, _apply, _feasible_joint,
                                    _rollout_cost, _wait_of, assert_healthy_action_mix,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.encoding import encode_observation
from ..integrated.joint_distill import (CentralJointValuePolicy, JointDecisionSample,
                                        load_student, save_student, top1_agreement,
                                        train_joint_net)
from ..integrated.objectives import hierarchy_key
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr059_state_norm import fit_state_norm
from .yr071_realign_g0 import _paired_ci

LEVEL = InformationLevel.PRE_ADVICE
RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr073_distill")
HORIZON = 1_800.0                               # YR-078 권고 — 교사 창
TRAIN = {"mid": range(740000, 740030), "high": range(740100, 740130)}
DAGGER = {"mid": range(741000, 741010), "high": range(741100, 741110)}
VAL = {"mid": range(743000, 743004), "high": range(743100, 743104)}
TEST = {"mid": range(742000, 742020), "high": range(742100, 742120)}
FIT_SEEDS = list(range(740100, 740105))         # high train 선두 5 (yr059 관행)
SLOTS = tuple(sorted(c.crane_id for c in build_calibrated_profile().cranes))


def _sim(level: str, seed: int):
    profile = build_calibrated_profile()
    sim = TerminalSimulator(profile,
                            generate_terminal_scenario(profile, seed,
                                                       calibrated_load_params(level)),
                            check_invariants=True)
    sim.info_level = LEVEL
    return sim


def collect_episode(level: str, seed: int, *, norm, execute: str = "teacher",
                    student=None, tag: str = "r0") -> tuple[list, dict]:
    """교사(JR_NEW 1800s) 로깅 결정 루프 — execute='student' 면 DAgger (학생 실행·교사 라벨)."""
    sim = _sim(level, seed)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RC, horizon_s=HORIZON, generator=gen, objective=hierarchy_key)
    sf = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    samples, k, n_dis = [], 0, 0
    t0 = time.perf_counter()
    dp = sim.run_until_decision()
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "collect", k, generator=gen)
        encs = {ob.crane_id: encode_observation(state, ob, norm=norm) for ob in obs}
        ca, cb = SLOTS                          # 프로파일 고정 슬롯 (1크레인 결정 대응)
        pos = lambda c, assign: (encs[c].candidate_ids.index(assign[c].candidate_id)
                                 if c in assign else -1)
        combos, tiers, assigns, best, best_key = [], [], [], None, None
        for combo in jr._admissible_combos(sim, dp, gen_by):
            assign = dict(zip(dp.crane_ids, combo))
            if not _feasible_joint(sim, assign):
                continue
            sink: dict = {}
            _rollout_cost(sim, assign, RC, horizon_s=HORIZON,
                          base_policy=jr.base_policy, generator=gen, term_sink=sink)
            key = hierarchy_key(sink)
            tie = tuple((c, assign[c].candidate_id) for c in sorted(dp.crane_ids))
            if best_key is None or (key, tie) < best_key:
                best_key, best = (key, tie), len(assigns)
            combos.append((pos(ca, assign), pos(cb, assign)))
            tiers.append(key[0])
            assigns.append(assign)
        if assigns:
            sfa = sf.decide(sim, dp, gen_by)
            sf_pair = (pos(ca, sfa), pos(cb, sfa))
            sf_pos = combos.index(sf_pair) if sf_pair in combos else None
            disagree = sf_pos is None or sf_pos != best
            n_dis += int(disagree)
            ea, eb = encs.get(ca), encs.get(cb)
            samples.append(JointDecisionSample(
                (ea or eb).g, ea.yc if ea else (), ea.queue if ea else (),
                ea.cand if ea else (), eb.yc if eb else (), eb.queue if eb else (),
                eb.cand if eb else (), tuple(combos), tuple(tiers), best, sf_pos,
                disagree, tag))
            chosen = (assigns[best] if execute == "teacher"
                      else student.decide(sim, dp, gen_by))
            _apply(sim, chosen)
        else:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        dp = sim.run_until_decision()
        k += 1
    jobs = list(sim.jobs.values())
    stat = {"seed": seed, "level": level, "n_dec": k, "n_disagree": n_dis,
            "completion": sum(1 for j in jobs if j.status.name == "DONE") / len(jobs),
            "wall_s": round(time.perf_counter() - t0, 1)}
    return samples, stat


def _val_fn_factory(norm):
    def val_fn(net) -> float:
        waits = []
        for level, seeds in VAL.items():
            for s in seeds:
                pol = CentralJointValuePolicy(net, norm, CandidateGenerator(), SLOTS)
                row = run_joint_episode(_sim(level, s), pol, RC,
                                        generator=CandidateGenerator())
                waits.append(row["mean_wait_min"])
        return round(fmean(waits), 4)
    return val_fn


def _eval_row(level: str, seed: int, policy_factory, name: str) -> dict:
    t0 = time.perf_counter()
    row = run_joint_episode(_sim(level, seed), policy_factory(), RC,
                            generator=CandidateGenerator())
    mix = row["action_mix"]
    healthy = True
    try:
        assert_healthy_action_mix(row["_mix"], label=f"{level}/{name}/s{seed}")
    except ActionMixError:
        healthy = False
    return {"seed": seed, "level": level, "arm": name,
            "wall_s": round(time.perf_counter() - t0, 2),
            "mean_wait": round(row["mean_wait_min"], 4),
            "p95_wait": round(row["p95_wait_min"], 4),
            "total_cost_OLD": round(row["total_cost"], 2),
            "completion": row["completion_rate"], "backlog": row["backlog"],
            "swa": mix["serve_when_available"],
            "serve_share": mix["shares"].get("SERVE", 0.0),
            "repo_share": mix["shares"].get("REPOSITION", 0.0),
            "healthy_mix": healthy}


def _by(rows, level, arm):
    return sorted((r for r in rows if r["level"] == level and r["arm"] == arm),
                  key=lambda r: r["seed"])


def _diff(rows, level, a, b, key):
    ra, rb = _by(rows, level, a), _by(rows, level, b)
    return _paired_ci([x[key] - y[key] for x, y in zip(ra, rb)])


def run_yr073(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    profile = build_calibrated_profile()
    print("[phase] state_norm fit", flush=True)
    norm, detail = fit_state_norm(profile, calibrated_load_params("high"), FIT_SEEDS)
    (out / "state_norm.json").write_text(json.dumps(
        {"basis": norm.basis, "refs": norm.refs, "n_fields": len(norm.refs)},
        ensure_ascii=False, indent=2), encoding="utf-8")

    r0, stats = [], []
    for level, seeds in TRAIN.items():
        for s in seeds:
            smp, st = collect_episode(level, s, norm=norm, tag="r0")
            r0 += smp
            stats.append(st)
            print(f"[collect r0 {level}] seed={s} dec={st['n_dec']} "
                  f"dis={st['n_disagree']} wall={st['wall_s']}s", flush=True)
    val_fn = _val_fn_factory(norm)
    print("[phase] train v0", flush=True)
    tr0 = train_joint_net(r0, val_fn=val_fn)
    save_student(out / "student_v0.pt", tr0, norm.refs)
    v0_val = min(h["val"] for h in tr0.history if "val" in h)

    print("[phase] DAgger r1 (학생 실행·교사 라벨)", flush=True)
    r1 = []
    for level, seeds in DAGGER.items():
        for s in seeds:
            pol = CentralJointValuePolicy(tr0.net, norm, CandidateGenerator(), SLOTS)
            smp, st = collect_episode(level, s, norm=norm, execute="student",
                                      student=pol, tag="r1")
            r1 += smp
            stats.append(st)
            print(f"[collect r1 {level}] seed={s} dec={st['n_dec']} "
                  f"dis={st['n_disagree']} wall={st['wall_s']}s", flush=True)
    print("[phase] train v1 (r0+r1)", flush=True)
    tr1 = train_joint_net(r0 + r1, val_fn=val_fn)
    save_student(out / "student_v1.pt", tr1, norm.refs)
    v1_val = min(h["val"] for h in tr1.history if "val" in h)
    final_tag = "STUDENT_R1" if v1_val <= v0_val else "STUDENT_R0"
    agree = {"v0": top1_agreement(tr0.net, r0), "v1": top1_agreement(tr1.net, r0 + r1)}

    print(f"[phase] eval (final={final_tag} v0_val={v0_val} v1_val={v1_val})", flush=True)
    arms = {
        "SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
        "JR_NEW_1800": lambda: _teacher(),
        "STUDENT_R0": lambda: CentralJointValuePolicy(tr0.net, norm, CandidateGenerator(),
                                                      SLOTS, name="STUDENT_R0"),
        "STUDENT_R1": lambda: CentralJointValuePolicy(tr1.net, norm, CandidateGenerator(),
                                                      SLOTS, name="STUDENT_R1"),
    }
    rows = []
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for level, seeds in TEST.items():
            for name, fac in arms.items():
                for s in seeds:
                    r = _eval_row(level, s, fac, name)
                    rows.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                ag = _by(rows, level, name)
                print(f"[eval {level}/{name}] wait={fmean(x['mean_wait'] for x in ag):.3f} "
                      f"repo={fmean(x['repo_share'] for x in ag):.3f} "
                      f"wall={fmean(x['wall_s'] for x in ag):.1f}s", flush=True)
    det_ok = all(
        _eval_row(level, list(TEST[level])[i], arms[final_tag], final_tag)[k]
        == next(r for r in rows if r["level"] == level and r["arm"] == final_tag
                and r["seed"] == list(TEST[level])[i])[k]
        for level in TEST for i in range(2)
        for k in ("mean_wait", "p95_wait", "total_cost_OLD"))

    res: dict = {"prereg": "2026-07-19-YR-073-순위증류-prereg.md", "horizon_s": HORIZON,
                 "final": final_tag, "val": {"v0": v0_val, "v1": v1_val},
                 "train_meta": {"n_r0": len(r0), "n_r1": len(r1),
                                "disagree_share_r0": round(
                                    sum(s.disagree for s in r0) / len(r0), 4),
                                "agreement": agree,
                                "best": {"v0": tr0.best_tag, "v1": tr1.best_tag}},
                 "determinism_ok": det_ok, "levels": {}}
    for level in TEST:
        dw = _diff(rows, level, final_tag, "SF_SPT", "mean_wait")
        dp95 = _diff(rows, level, final_tag, "SF_SPT", "p95_wait")
        dtch = _diff(rows, level, final_tag, "JR_NEW_1800", "mean_wait")
        arm_stats = {a: {k: round(fmean(r[k] for r in _by(rows, level, a)), 4)
                         for k in ("mean_wait", "p95_wait", "total_cost_OLD", "swa",
                                   "serve_share", "repo_share", "wall_s")}
                     | {"healthy_all": all(r["healthy_mix"] for r in _by(rows, level, a)),
                        "completion_all1": all(r["completion"] == 1.0
                                               for r in _by(rows, level, a)),
                        "backlog_all0": all(r["backlog"] == 0
                                            for r in _by(rows, level, a))}
                     for a in arms}
    # ---- 게이트
        st, tch, sfr = arm_stats[final_tag], arm_stats["JR_NEW_1800"], arm_stats["SF_SPT"]
        teacher_gain = sfr["mean_wait"] - tch["mean_wait"]
        res["levels"][level] = {
            "arms": arm_stats, "d_wait_vs_SF": dw, "d_p95_vs_SF": dp95,
            "d_wait_vs_teacher": dtch,
            "g1_pass": dw["hi"] < 0.0,
            "g2_capture": round((sfr["mean_wait"] - st["mean_wait"])
                                / teacher_gain, 3) if teacher_gain > 0 else None,
            "g3_wall_ratio": round(st["wall_s"] / tch["wall_s"], 3),
            "guard_all": (st["completion_all1"] and st["backlog_all0"]
                          and st["healthy_all"] and st["repo_share"] <= 0.15
                          and not (dp95["lo"] > 0.0)),
        }
    res["G1_PASS"] = (det_ok and all(res["levels"][lv]["g1_pass"]
                                     and res["levels"][lv]["guard_all"] for lv in TEST))
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (out / "collect_stats.json").write_text(json.dumps(stats, ensure_ascii=False,
                                                       indent=2), encoding="utf-8")
    _report(res, out / "yr073_report.md")
    print(f"\nG1_PASS={res['G1_PASS']} final={final_tag}", flush=True)
    print("DONE", flush=True)
    return res


def _teacher():
    pol = JointRolloutGreedy(RC, horizon_s=HORIZON, generator=CandidateGenerator(),
                             objective=hierarchy_key)
    pol.name = "JR_NEW_1800"
    return pol


def _report(res: dict, path: Path) -> None:
    tm = res["train_meta"]
    lines = ["# YR-073 — 중앙 공동가치망 (JR_NEW 1800s 순위 증류)", "",
             f"> prereg 동결 실행 · final={res['final']} (val v0 {res['val']['v0']} / "
             f"v1 {res['val']['v1']}) · 결정론 {'OK' if res['determinism_ok'] else 'FAIL'}"
             " · **문헌 보정 시뮬레이션 조건**", "",
             f"- 표본: r0 {tm['n_r0']}·r1 {tm['n_r1']} 결정, 분기(교사≠SF) 비율 "
             f"{tm['disagree_share_r0']} · 순위 일치(top-1) v1 전체 "
             f"{tm['agreement']['v1']['top1_all']}·분기 한정 "
             f"{tm['agreement']['v1']['top1_disagree']}", "",
             "| 수준 | arm | 평균대기 | P95 | swa | REPO | wall(s) |", "|---|---|---|---|---|---|---|"]
    for level, lv in res["levels"].items():
        for a, s in lv["arms"].items():
            lines.append(f"| {level} | {a} | {s['mean_wait']} | {s['p95_wait']} "
                         f"| {s['swa']} | {s['repo_share']} | {s['wall_s']} |")
    lines += ["", "## 판정", ""]
    for level, lv in res["levels"].items():
        w = lv["d_wait_vs_SF"]
        t = lv["d_wait_vs_teacher"]
        lines.append(f"- **{level}**: G1 Δ대기 vs SF **{w['mean']}분 [{w['lo']}, {w['hi']}]**"
                     f" {'✅' if lv['g1_pass'] else '❌'} · vs 교사 {t['mean']} "
                     f"[{t['lo']}, {t['hi']}] (교사 이득 확보율 {lv['g2_capture']}) · "
                     f"wall 비율 {lv['g3_wall_ratio']} · guard "
                     f"{'OK' if lv['guard_all'] else 'FAIL'}")
    lines += ["", f"**G1 {'통과' if res['G1_PASS'] else '기각'}** · 원자료 rows.jsonl·"
              "collect_stats.json·student_v0/v1.pt"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr073()
