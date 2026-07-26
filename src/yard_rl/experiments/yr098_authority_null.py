"""YR-098 — 통제권한 감사 (authority-null): 블록 Q 의 본선 여지가 어디에 얼마나 있나.

■ 사전등록 (2026-07-26, 결과 미열람 동결 — spec: dashboard-task-specs/YR-098)
- **수정 아닌 측정** (학습기·Q·인코딩·보상 0 수정). 고정 resolver preference 4팔:
  SF(기준) · VesselFirst(본선 SERVE 절대우선) · LoadOnlyFirst(적하만 tier0) ·
  ConditionalVesselFirst(그 선박 flow_margin_s<0 일 때만 tier0).
- 셀 4(yr088 CELLS) × seed 10(BASE+500+i, **선택 대역**) · v2 시간계약 · 물리 정정 후 ·
  평가비용 = 순수 numeraire_v1. 수집: total_cost·berth·트럭평균·완주·term_contrib +
  선박별 지연 max(0, actual−planned) 을 **LOAD/DISCHARGE 버킷 분해**.
- **게이트 = 최선 규칙팔 − SF** (무조건 VF−SF 아님 — 거짓 NO-GO 방지). 선택 후 CI 는
  낙관 편향이 있으므로 **확증 대역(+700, 10 seed)에서 선택된 팔만 재검증** — 확증 CI 가
  판정 기준: total_cost 절감 CI 하한>0 ⇒ GO(여지 존재). berth 도 병행 보고.
- **GO 의 의미**: "행동공간에 타이밍 여지 존재"이지 학습가능성이 아님 — YR-097/후속의
  스코핑 신호(LOAD 한정 여부는 버킷 분해가 답함).
- YT 스윕: transfer.n_units∈{3,6,12,24} × {SF, VesselFirst} × high-tight×5 seed —
  적하 병목이 YT-bound 가 아님(여지가 크레인 타이밍 소관임) 재확인.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from statistics import fmean

from ..contract.schema import CandidateKind
from ..domain.enums import InformationLevel, JobFlow
from ..integrated import TerminalSimulator
from ..integrated.baselines import (BaselinePreference, ResolverPolicy,
                                    ServiceFirstSPTPreference, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from ..integrated.vessel_signals import flow_margin_s
from .yr071_realign_g0 import _paired_ci
from .yr080e_vessel_controllability import VesselFirstServe

RC = RewardCalculator.numeraire_v1()
CELLS = {"mid-loose": ("mid", 2.0), "high-loose": ("high", 2.0),
         "mid-tight": ("mid", 0.5), "high-tight": ("high", 0.5)}
BASE = {"mid-loose": 830000, "high-loose": 830100, "mid-tight": 830200, "high-tight": 830300}
N, SELECT_OFF, CONFIRM_OFF = 10, 500, 700
OUT = Path("outputs/reports/yr098_authority_null")


def _sim(cell: str, seed: int, n_units: int | None = None) -> TerminalSimulator:
    prof = build_calibrated_profile()
    if n_units is not None:                     # YT 스윕 — 이송 유닛 수 오버라이드
        prof = dataclasses.replace(prof, transfer=dataclasses.replace(
            prof.transfer, n_units=n_units))
    lvl, dm = CELLS[cell]
    p = dataclasses.replace(calibrated_load_params(lvl, vessel_deadline_mult=dm),
                            time_contract_v2=True)
    s = TerminalSimulator(prof, generate_terminal_scenario(prof, seed, p), check_invariants=True)
    s.info_level = InformationLevel.PRE_ADVICE
    return s


# ---------------------------------------------------------------- 규칙 4팔
class LoadOnlyFirstServe(BaselinePreference):
    """적하(VESSEL_LOAD) SERVE 만 절대우선 — 양하는 일반 SERVE 취급 (통제권한 스코핑 팔)."""

    def rank(self, sim, crane_id, gc) -> tuple:
        is_serve = gc.kind == CandidateKind.SERVE
        dur = gc.plan.duration_s if gc.plan is not None else float("inf")
        is_load = False
        if is_serve and gc.job_ref is not None and gc.job_ref.is_vessel:
            j = sim.jobs.get(gc.job_ref.job_id)
            is_load = j is not None and j.flow == JobFlow.VESSEL_LOAD
        tier = 0 if is_load else (1 if is_serve else 2)
        return (tier, dur) + super().rank(sim, crane_id, gc)


class ConditionalVesselFirstServe(BaselinePreference):
    """그 선박의 flow_margin_s < 0 (버퍼 여유 소진) 일 때만 본선 SERVE tier0 — 표적 개입 팔."""

    def rank(self, sim, crane_id, gc) -> tuple:
        is_serve = gc.kind == CandidateKind.SERVE
        dur = gc.plan.duration_s if gc.plan is not None else float("inf")
        urgent = False
        if is_serve and gc.job_ref is not None and gc.job_ref.is_vessel:
            j = sim.jobs.get(gc.job_ref.job_id)
            v = sim.vessels.get(j.vessel_id) if (j is not None and j.vessel_id) else None
            if v is not None and not v.done:
                fm = flow_margin_s(sim, v, sim.now)
                urgent = fm is not None and fm < 0.0
        tier = 0 if urgent else (1 if is_serve else 2)
        return (tier, dur) + super().rank(sim, crane_id, gc)


ARMS = {"SF": ServiceFirstSPTPreference, "VF": VesselFirstServe,
        "LOAD1ST": LoadOnlyFirstServe, "CONDVF": ConditionalVesselFirstServe}


def _vessel_delay_split(sim) -> dict:
    """선박별 max(0, actual−planned) 분(LOAD/DISCHARGE 버킷) — 미완은 end 기준."""
    out = {"LOAD": 0.0, "DISCHARGE": 0.0}
    for v in sim.vessels.values():
        pc = v.plan.planned_completion_s
        if pc is None:
            continue
        actual = v.truth.actual_completion_s if v.done else sim.end
        out[v.work_type.value] += max(0.0, actual - pc) / 60.0
    return out


def _episode(cell, seed, arm, n_units=None) -> dict:
    sim = _sim(cell, seed, n_units)
    r = run_joint_episode(sim, ResolverPolicy(ARMS[arm](), arm), RC,
                          generator=CandidateGenerator())
    split = _vessel_delay_split(sim)
    return {"cell": cell, "seed": seed, "arm": arm, "total_cost": r["total_cost"],
            "berth_overrun_min": r["berth_overrun_min"], "mean_wait_min": r["mean_wait_min"],
            "p95_wait_min": r["p95_wait_min"], "completion_rate": r["completion_rate"],
            "backlog": r["backlog"], "berth_load_min": round(split["LOAD"], 2),
            "berth_disch_min": round(split["DISCHARGE"], 2),
            "term_vessel": round(r["term_contrib"].get("vessel_delay", 0.0), 3),
            "term_truck": round(r["term_contrib"].get("truck_wait", 0.0)
                                + r["term_contrib"].get("long_wait", 0.0), 3)}


def _band(offset: int, arms) -> list[dict]:
    rows = []
    for cell in CELLS:
        for i in range(N):
            for arm in arms:
                rows.append(_episode(cell, BASE[cell] + offset + i, arm))
        print(f"  [{'select' if offset == SELECT_OFF else 'confirm'}] {cell} done", flush=True)
    return rows


def _diffs(rows, a, b, key):
    ra = {(r["cell"], r["seed"]): r[key] for r in rows if r["arm"] == a}
    rb = {(r["cell"], r["seed"]): r[key] for r in rows if r["arm"] == b}
    return [ra[k] - rb[k] for k in sorted(ra) if k in rb]


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== 선택 대역 (+500) — 4팔 ===", flush=True)
    sel = _band(SELECT_OFF, list(ARMS))
    res: dict = {"select": {}, "confirm": {}, "yt_sweep": {}, "split": {}}
    for arm in ("VF", "LOAD1ST", "CONDVF"):
        res["select"][arm] = {
            "total_cost": _paired_ci(_diffs(sel, arm, "SF", "total_cost")),
            "berth": _paired_ci(_diffs(sel, arm, "SF", "berth_overrun_min")),
            "wait": _paired_ci(_diffs(sel, arm, "SF", "mean_wait_min"))}
        print(f"  {arm}−SF: cost {res['select'][arm]['total_cost']} | "
              f"berth {res['select'][arm]['berth']} | wait {res['select'][arm]['wait']}", flush=True)
    best = min(("VF", "LOAD1ST", "CONDVF"),
               key=lambda a: res["select"][a]["total_cost"]["mean"])
    res["best_arm"] = best
    # LOAD/DISCH 버킷 (선택 대역 · SF 기준 스냅샷)
    for arm in ARMS:
        rows_a = [r for r in sel if r["arm"] == arm]
        res["split"][arm] = {"load": round(fmean(r["berth_load_min"] for r in rows_a), 2),
                             "disch": round(fmean(r["berth_disch_min"] for r in rows_a), 2)}
    print(f"\n=== 확증 대역 (+700) — 선택된 팔 {best} vs SF ===", flush=True)
    con = _band(CONFIRM_OFF, ["SF", best])
    res["confirm"][best] = {
        "total_cost": _paired_ci(_diffs(con, best, "SF", "total_cost")),
        "berth": _paired_ci(_diffs(con, best, "SF", "berth_overrun_min")),
        "wait": _paired_ci(_diffs(con, best, "SF", "mean_wait_min")),
        "p95": _paired_ci(_diffs(con, best, "SF", "p95_wait_min"))}
    c = res["confirm"][best]["total_cost"]
    res["gate"] = {"arm": best, "go": c["hi"] < 0,
                   "meaning": "GO=행동공간에 타이밍 여지 존재(학습가능성 아님) — 확증 대역 CI 기준"}
    print(f"  확증 {best}−SF: cost {c} | berth {res['confirm'][best]['berth']} | "
          f"GATE {'GO' if c['hi'] < 0 else 'NO-GO'}", flush=True)
    print("\n=== YT 스윕 (high-tight × 5 seed × SF/VF) ===", flush=True)
    for nu in (3, 6, 12, 24):
        rows_nu = [_episode("high-tight", BASE["high-tight"] + SELECT_OFF + i, arm, n_units=nu)
                   for i in range(5) for arm in ("SF", "VF")]
        res["yt_sweep"][nu] = {
            arm: {"berth": round(fmean(r["berth_overrun_min"] for r in rows_nu
                                       if r["arm"] == arm), 1),
                  "load": round(fmean(r["berth_load_min"] for r in rows_nu
                                      if r["arm"] == arm), 1)}
            for arm in ("SF", "VF")}
        print(f"  n_units={nu}: {res['yt_sweep'][nu]}", flush=True)
    (OUT / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sel + con), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\nGATE: {res['gate']} | saved {OUT/'results.json'}")
    return res


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
    print("YR098 DONE")
