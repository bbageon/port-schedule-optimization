"""YR-074 — NEW 반사실 미세조정 (prereg 2026-07-20 동결 실행).

학습 신호 = 환경 반사실: 학생 선택 vs SF-SPT 선택이 갈린 결정에서만 두 공동행동을
1800s rollout, A = tierA(학생)−tierA(SF). 결정당 rollout 2회 (교사 전량의 ~1/10).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import fmean

import torch

from ..integrated.baselines import (ActionMix, ActionMixError, ResolverPolicy,
                                    ServiceFirstSPTPreference, _apply, _rollout_cost,
                                    _wait_of, assert_healthy_action_mix,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.joint_distill import (CentralJointValuePolicy, finetune_pairwise,
                                        load_student)
from ..integrated.objectives import hierarchy_key
from .yr071_realign_g0 import _paired_ci
from .yr073_joint_distill import SLOTS, VAL, _sim
from .yr073b_confirm import MODEL

RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr074_finetune")
HORIZON = 1_800.0
ROUNDS = {1: {"mid": range(745000, 745010), "high": range(745100, 745110)},
          2: {"mid": range(745010, 745020), "high": range(745110, 745120)}}
TEST = {"mid": list(range(746000, 746020)), "high": list(range(746100, 746120))}


def collect_corrective(level: str, seed: int, net, norm) -> tuple[list, dict]:
    sim = _sim(level, seed)
    gen = CandidateGenerator()
    pol = CentralJointValuePolicy(net, norm, gen, SLOTS)
    sf = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    base = ResolverPolicy(ServiceFirstSPTPreference(), "BASE")
    pairs, k, n_dis = [], 0, 0
    t0 = time.perf_counter()
    dp = sim.run_until_decision()
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, sim.info_level) for c in dp.crane_ids}
        chosen = pol.decide(sim, dp, gen_by)
        rows, assigns = getattr(pol, "_last", ([], []))
        if assigns:
            ci = next(i for i, a in enumerate(assigns) if all(
                a[c].candidate_id == chosen[c].candidate_id for c in a))
            sfa = sf.decide(sim, dp, gen_by)
            si = next((i for i, a in enumerate(assigns) if all(
                a[c].candidate_id == sfa[c].candidate_id for c in a)), None)
            if si is not None and si != ci:
                n_dis += 1
                s1, s2 = {}, {}
                _rollout_cost(sim, assigns[ci], RC, horizon_s=HORIZON,
                              base_policy=base, generator=gen, term_sink=s1)
                _rollout_cost(sim, assigns[si], RC, horizon_s=HORIZON,
                              base_policy=base, generator=gen, term_sink=s2)
                a = hierarchy_key(s1)[0] - hierarchy_key(s2)[0]
                if abs(a) > 1e-6:
                    pairs.append((rows[ci], rows[si], a))
        _apply(sim, chosen if assigns else {c: _wait_of(gen_by[c])
                                            for c in dp.crane_ids})
        dp = sim.run_until_decision()
        k += 1
    return pairs, {"seed": seed, "level": level, "n_dec": k, "n_disagree": n_dis,
                   "n_pairs": len(pairs), "wall_s": round(time.perf_counter() - t0, 1)}


def _val_fn(norm):
    def f(net) -> float:
        waits = []
        for level, seeds in VAL.items():
            for s in seeds:
                row = run_joint_episode(
                    _sim(level, s),
                    CentralJointValuePolicy(net, norm, CandidateGenerator(), SLOTS),
                    RC, generator=CandidateGenerator())
                if row["completion_rate"] < 1.0:
                    return float("inf")        # 즉시 복귀 트리거 (완주 실패)
                waits.append(row["mean_wait_min"])
        return round(fmean(waits), 4)
    return f


def _row74(level, seed, policy, name, agg: ActionMix) -> dict:
    t0 = time.perf_counter()
    row = run_joint_episode(_sim(level, seed), policy, RC,
                            generator=CandidateGenerator())
    mix = row["_mix"]
    for kind, n in mix.counts.items():
        agg.counts[kind] = agg.counts.get(kind, 0) + n
    agg.serve_available += mix.serve_available
    agg.serve_taken += mix.serve_taken
    return {"seed": seed, "level": level, "arm": name,
            "wall_s": round(time.perf_counter() - t0, 2),
            "mean_wait": round(row["mean_wait_min"], 4),
            "p95_wait": round(row["p95_wait_min"], 4),
            "completion": row["completion_rate"], "backlog": row["backlog"],
            "repo_share": row["action_mix"]["shares"].get("REPOSITION", 0.0),
            "interference": round(row["term_contrib"].get("interference", 0.0), 3),
            "truck_wait_term": round(row["term_contrib"].get("truck_wait", 0.0), 3),
            "total_OLD": round(row["total_cost"], 2)}


def run_yr074(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    net0, norm = load_student(MODEL)
    val_fn = _val_fn(norm)
    net, pairs, stats, tags = net0, [], [], []
    for r, bands in ROUNDS.items():
        for level, seeds in bands.items():
            for s in seeds:
                p, st = collect_corrective(level, s, net, norm)
                pairs += p
                stats.append(st)
                print(f"[collect r{r} {level}] seed={s} dis={st['n_disagree']} "
                      f"pairs={st['n_pairs']} wall={st['wall_s']}s", flush=True)
        print(f"[phase] finetune round {r} (pairs={len(pairs)})", flush=True)
        net, hist, tag = finetune_pairwise(net0, pairs, val_fn=val_fn)
        tags.append({"round": r, "best": tag,
                     "val_curve": [h.get("val") for h in hist if "val" in h]})
    torch.save({"fmt": "yard-rl-joint-ft-v1", "state": net.state_dict(),
                "in_dim": net.in_dim, "norm_refs": norm.refs,
                "tags": tags}, out / "student_ft.pt")

    print("[phase] test", flush=True)
    facs = {"SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
            "STUDENT_R1": lambda: CentralJointValuePolicy(
                net0, norm, CandidateGenerator(), SLOTS, name="STUDENT_R1"),
            "STUDENT_FT": lambda: CentralJointValuePolicy(
                net, norm, CandidateGenerator(), SLOTS, name="STUDENT_FT")}
    rows, aggs = [], {}
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for level, seeds in TEST.items():
            for name, fac in facs.items():
                agg = aggs.setdefault((level, name), ActionMix())
                for s in seeds:
                    r = _row74(level, s, fac(), name, agg)
                    rows.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                sel = [x for x in rows if x["level"] == level and x["arm"] == name]
                print(f"[test {level}/{name}] wait={fmean(x['mean_wait'] for x in sel):.3f}"
                      f" intf={fmean(x['interference'] for x in sel):.1f}", flush=True)
    det_ok = all(
        _row74(level, TEST[level][i], facs["STUDENT_FT"](), "STUDENT_FT", ActionMix())[k]
        == next(r for r in rows if r["level"] == level and r["arm"] == "STUDENT_FT"
                and r["seed"] == TEST[level][i])[k]
        for level in TEST for i in range(2) for k in ("mean_wait", "p95_wait"))
    res: dict = {"prereg": "2026-07-20-YR-074-반사실미세조정-prereg.md",
                 "ft_tags": tags, "n_pairs": len(pairs), "determinism_ok": det_ok,
                 "levels": {}}
    for level in TEST:
        by = lambda a: sorted((r for r in rows if r["level"] == level
                               and r["arm"] == a), key=lambda r: r["seed"])
        ft, r1, sf = by("STUDENT_FT"), by("STUDENT_R1"), by("SF_SPT")
        d1 = _paired_ci([a["mean_wait"] - b["mean_wait"] for a, b in zip(ft, r1)])
        dp95 = _paired_ci([a["p95_wait"] - b["p95_wait"] for a, b in zip(ft, r1)])
        dsf = _paired_ci([a["mean_wait"] - b["mean_wait"] for a, b in zip(ft, sf)])
        dintf = _paired_ci([a["interference"] - b["interference"]
                            for a, b in zip(ft, r1)])
        agg = aggs[(level, "STUDENT_FT")]
        try:
            assert_healthy_action_mix(agg, label=f"{level}/FT")
            healthy = True
        except ActionMixError:
            healthy = False
        stats_of = lambda xs: {k: round(fmean(r[k] for r in xs), 3)
                               for k in ("mean_wait", "p95_wait", "repo_share",
                                         "interference", "total_OLD", "wall_s")}
        lv = {"FT": stats_of(ft), "R1": stats_of(r1), "SF": stats_of(sf),
              "d_wait_FT_vs_R1": d1, "d_p95_FT_vs_R1": dp95,
              "d_wait_FT_vs_SF": dsf, "d_intf_FT_vs_R1": dintf,
              "a1_pass": d1["hi"] < 0.0,
              "guards": {"completion_all1": all(r["completion"] == 1.0 for r in ft),
                         "backlog_all0": all(r["backlog"] == 0 for r in ft),
                         "p95_ok": not (dp95["lo"] > 0.0),
                         "policy_mix_healthy": healthy,
                         "repo_relative_ok": fmean(r["repo_share"] for r in ft)
                         <= fmean(r["repo_share"] for r in sf) + 0.10}}
        lv["guard_all"] = all(lv["guards"].values())
        res["levels"][level] = lv
    res["A1_PASS"] = det_ok and all(res["levels"][lv]["a1_pass"]
                                    and res["levels"][lv]["guard_all"] for lv in TEST)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (out / "collect_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# YR-074 — 반사실 미세조정 (A1 {'통과' if res['A1_PASS'] else '기각'})",
             "", f"> pairs {len(pairs)} · ft {tags} · 결정론 {det_ok} · 문헌 보정 조건", ""]
    for level, lv in res["levels"].items():
        lines += [f"- **{level}**: FT {lv['FT']['mean_wait']} vs R1 {lv['R1']['mean_wait']}"
                  f" vs SF {lv['SF']['mean_wait']}분 · Δ(FT−R1) {lv['d_wait_FT_vs_R1']['mean']}"
                  f" [{lv['d_wait_FT_vs_R1']['lo']}, {lv['d_wait_FT_vs_R1']['hi']}] "
                  f"{'✅' if lv['a1_pass'] else '❌'} · Δ간섭(FT−R1) "
                  f"{lv['d_intf_FT_vs_R1']['mean']} [{lv['d_intf_FT_vs_R1']['lo']}, "
                  f"{lv['d_intf_FT_vs_R1']['hi']}] · guards {lv['guards']}"]
    (out / "yr074_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nA1_PASS={res['A1_PASS']} tags={tags}", flush=True)
    print("DONE", flush=True)
    return res


if __name__ == "__main__":
    run_yr074()
