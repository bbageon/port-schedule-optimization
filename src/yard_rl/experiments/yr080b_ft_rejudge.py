"""YR-080 후보 B (1부) — 채택본 FT 를 균형 목적(LIT)으로 **재채점** (재학습 없음).

병렬 세션 프로토타입(be1c056)은 단순규칙 vs JR 만 LIT 로 봤다. 여기선 **채택 정책
FT**(구 목적으로 학습됨)를 LIT 로 재채점해 "FT 의 이득이 본선 넣으면 얼마나 남나,
본선을 얼마나 희생했나"를 정량화한다. LIT 가중치·정규화 방식은 프로토타입 그대로
(divergence 방지), v2 환경에 scale 재적합. 진단(사전등록 아님)·재학습은 YR-080 정식 뒤.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from ..contract.schema import COST_TERMS
from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.baselines import (JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost import ASSUMED_SCALE
from ..integrated.cost_config import (Provenance, ProvBasis, RewardCalculator,
                                      default_assumed_config)
from ..integrated.joint_distill import (CentralJointValuePolicy,
                                        adopted_slot_selector, load_student)
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario

LEVEL = InformationLevel.PRE_ADVICE
OUT = Path("outputs/reports/yr080b_ft_rejudge")
MODEL = Path("outputs/reports/yr074_finetune/student_ft.pt")
FIT_SEEDS = list(range(753500, 753503))
SEEDS = list(range(753000, 753012))
CELLS = [("L80/F65", 80, 0.65), ("L112/F65", 112, 0.65)]   # 본선 스트레스 (프로토타입 정합)
HZN = 600.0                                                 # 프로토타입 속도용

# 프로토타입 LIT 가중치 그대로 (2026-07-20-YR-080 프로토타입 §1)
LIT_W = {"truck_wait": 1.0, "long_wait": 1.0, "transfer_wait": 1.0,
         "sts_wait": 1.0, "vessel_delay": 1.0, "depart_delay": 1.0,
         "crane_travel": 0.3, "empty_travel": 0.3, "rehandle": 0.3,
         "imbalance": 0.3, "lane_cong": 0.3,
         "interference": 0.0, "resequence": 0.0}
RC_OLD = RewardCalculator.assumed_default()


def _sim(seed, n_ext, fill):
    p = build_calibrated_profile()
    sc = generate_terminal_scenario(p, seed,
                                    calibrated_load_params("high", n_external=n_ext,
                                                           fill_ratio=fill))
    s = TerminalSimulator(p, sc, check_invariants=True)
    s.info_level = LEVEL
    return s


def _fit_scales():
    rows = []
    for s in FIT_SEEDS:
        r = run_joint_episode(_sim(s, 80, 0.65),
                              ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                              RC_OLD, generator=CandidateGenerator())
        rows.append(r["term_contrib"])
    return {t: max(fmean(r.get(t, 0.0) for r in rows) * ASSUMED_SCALE[t],
                   ASSUMED_SCALE[t] * 0.05) for t in COST_TERMS}


def _build_lit(scale):
    prov = Provenance(ProvBasis.FITTED_BASELINE, "v2 SF_SPT baseline", "YR-080b 재판정")
    cfg = default_assumed_config().with_scale(scale, prov=prov).with_weight(
        {t: LIT_W[t] for t in COST_TERMS})
    return RewardCalculator(cfg)


def _run(sim_fac, pol_fac, rc):
    r = run_joint_episode(sim_fac(), pol_fac(), rc, generator=CandidateGenerator())
    tc = r["term_contrib"]
    tot = max(1e-9, r["total_cost"])
    return {"total": round(r["total_cost"], 2), "wait": round(r["mean_wait_min"], 3),
            "vdelay": round(r["vessel_delay_min"], 2), "compl": round(r["completion_rate"], 4),
            "vessel_share": round(sum(tc.get(t, 0.0) for t in
                                      ("sts_wait", "vessel_delay", "depart_delay")) / tot, 3)}


def run_yr080b(out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    net, norm = load_student(MODEL)
    slots = tuple(sorted(c.crane_id for c in build_calibrated_profile().cranes))
    RC_LIT = _build_lit(_fit_scales())

    def ft():
        return CentralJointValuePolicy(net, norm, CandidateGenerator(), slots)

    def ft_with_h1_sim(seed, n_ext, fill):
        s = _sim(seed, n_ext, fill)
        s.slot_selector = adopted_slot_selector()      # 채택본 = FT + H1
        return s

    res = {"lit_weight": LIT_W, "cells": {}}
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for cid, n_ext, fill in CELLS:
            cell = {}
            for name, decide_rc in (("SF", None), ("FT", None),
                                    ("JR_OLD", RC_OLD), ("JR_LIT", RC_LIT)):
                for score_name, score_rc in (("OLD", RC_OLD), ("LIT", RC_LIT)):
                    # SF·FT 결정은 RC 무관 → 두 점수 다 산출. JR 은 자기 목적만.
                    if name.startswith("JR") and decide_rc is not score_rc:
                        continue
                    rows = []
                    for s in SEEDS:
                        if name == "SF":
                            sf = lambda: _sim(s, n_ext, fill)
                            pf = lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF")
                        elif name == "FT":
                            sf = lambda s=s: ft_with_h1_sim(s, n_ext, fill)
                            pf = ft
                        else:
                            sf = lambda: _sim(s, n_ext, fill)
                            pf = lambda drc=decide_rc: JointRolloutGreedy(
                                drc, horizon_s=HZN, max_combos=64,
                                generator=CandidateGenerator())
                        r = _run(sf, pf, score_rc)
                        rows.append(r)
                        f.write(json.dumps({"cell": cid, "arm": f"{name}/{score_name}",
                                            "seed": s, **r}, ensure_ascii=False) + "\n")
                    agg = {k: round(fmean(r[k] for r in rows), 3) for k in
                           ("total", "wait", "vdelay", "compl", "vessel_share")}
                    cell[f"{name}/{score_name}"] = agg
                    print(f"[{cid}] {name}/{score_name:3s} LITtot={agg['total']:>7.1f} "
                          f"wait={agg['wait']:>6.2f} vdelay={agg['vdelay']:>6.1f} "
                          f"vshare={agg['vessel_share']:.2f} compl={agg['compl']:.3f}",
                          flush=True)
            res["cells"][cid] = cell
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr080b_report.md")
    print("DONE", flush=True)
    return res


def _report(res, path):
    lines = ["# YR-080 후보 B(1부) — 채택본 FT 균형 목적 재채점 (재학습 없음)", "",
             "> LIT 가중치=프로토타입 동일·v2 scale 재적합·12 seed·창 600s · **진단(비확정)**",
             "> 본선 스트레스 셀. 핵심 질문: FT 이득이 본선 넣으면 얼마나 남나·본선 희생량.", "",
             "| 셀 | arm | 목적 | 총비용 | 트럭대기 | 본선지연(분) | 본선기여 | 완주 |",
             "|---|---|---|---|---|---|---|---|"]
    for cid, cell in res["cells"].items():
        for arm, a in cell.items():
            name, obj = arm.split("/")
            lines.append(f"| {cid} | {name} | {obj} | {a['total']} | {a['wait']} "
                         f"| {a['vdelay']} | {a['vessel_share']} | {a['compl']} |")
    lines += ["", "## 읽는 법",
              "- **LIT 목적 기준(총비용)** SF/FT/JR_LIT 순위가 진짜 판정 — FT 가 SF·JR_LIT 를",
              "  LIT 로도 이기면 채택 유효, 지면 균형 목적 재학습 필요(YR-080 정식 후).",
              "- **FT 본선지연** vs JR_LIT 본선지연 = FT 가 구 목적 학습으로 본선 희생한 양.",
              "- 재학습(교사→증류→FT, 수 시간)은 YR-080 정식 가중치 확정 후 (헛일 방지)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr080b()
