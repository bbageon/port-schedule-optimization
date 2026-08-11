"""YR-167 — 5차 계약(도착률·24시간 이중 피크) **자격 시험**.

사전등록 §F 의 검사 목록을 그대로 이행한다:
  W1′ 명단 준수(계획 도착 시각 == 실현 gate-in·누락 0)   ← W1(재공량 유지) 대체
  W2 장부 보존 / W3 결정론 / W4 본선 24시간 교대 분산 / W5 블록별 스냅샷
  W6 프로파일 H-21 YT / W7 내부타당성 6종
  W8′ 재고 소진 0                                        ← W8(pool 소진) 대체
  W9 flow fallback 0 / W11 피크 도달(계획 시간대별 최대 461.2 ±10%)
성능 주장 없음 — 환경 자격이다. 부수 관측으로 **벌금 구간 도달률**(체류 > 2,580초)과
시간대별 체류 곡선을 보고한다(판정 아님).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    _apply, _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.policy_config import LEGACY_DEFAULT
from ..integrated.profiles import build_h21_profile
from ..integrated.repro import code_dirty
from ..integrated.sell_review import ANNOUNCE_LEAD_S
from ..integrated.terminal_stream import (OBS_24H, ObservationContract,
                                          ensure_time_ledger,
                                          ScheduledAnnouncer, admission_epochs,
                                          build_diurnal, diurnal_rate)
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from
from .yr150_h21_pilot import _git, _sha256

OUT = Path("outputs/reports/yr167_diurnal_qual")
PREREG = Path(".claude/docs/strategy-history/"
              "2026-08-11-24시간-이중피크-상수-유도-사전등록.md")
SEED = 8_100_000                 # 5차 자격 전용 시드 대역
REPS = 3                         # 독립 시드 3
SLA_PENALTY_S = 2_580.0          # 벌금 임계(= SLA_ANCHOR 780 + long_wait 1800)
W11_TOL = 0.10


def planned_hourly_peak(obs: ObservationContract, total: int) -> float:
    """계획 시간대별 최대 도착 — 동결 λ(t) 에서 유도(사전등록 A7)."""
    hourly = [sum(diurnal_rate((h + i / 60) * 3600, day_s=obs.observe_s,
                               total=total) * 60 for i in range(60))
              for h in range(int(obs.observe_s // 3600))]
    return max(hourly)


def run_cell(rep: int) -> dict:
    obs = OBS_24H
    prof = build_h21_profile()
    layout = terminal_layout()
    built = build_diurnal(prof, SEED + rep, obs=obs, layout=layout,
                          background_seed=SEED)
    # 5차 계약: 초기 트럭 0 → v2 장부를 런너에서 활성화(엔진 골든 경로 불변)
    mbt = MultiBlockTerminal(
        {b: ensure_time_ledger(_sim_from(s))
         for b, s in built["scenarios"].items()},
        extra_review_epochs=admission_epochs(obs))
    ann = ScheduledAnnouncer(built["schedule"], lead_s=ANNOUNCE_LEAD_S,
                             end_s=obs.observe_s)
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator(config=LEGACY_DEFAULT))
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            exc["n"] += 1
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    mbt.run(policy, review_fn=ann.review)
    try:
        mbt.check_invariants()
        invariants = True
    except Exception:
        invariants = False

    # 장부 수집
    rows = []
    for bid, sim in mbt.blocks.items():
        tl = getattr(sim, "time_ledger", None)
        if tl is None:
            continue
        for jid, r in tl.records.items():
            rows.append({"job": jid, "block": bid, "a": r.gate_in,
                         "b": r.block_arrival, "o": r.gate_out})
    plan = {e["job_id"]: e["arrival_s"] for e in built["schedule"]}
    admitted = {e["job_id"] for e in ann.ledger if e["event"] == "ADMIT"}
    tail = {e["job_id"] for e in ann.ledger if e["event"] == "SKIP_TAIL"}
    skips = [e for e in ann.ledger if e["event"] == "SKIP"]

    # W1′ 명단 준수 — 투입된 전건의 실현 gate-in 이 계획 도착과 일치
    w1 = (not skips
          and all(abs(r["a"] - plan[r["job"]]) < 1e-6 for r in rows if r["job"] in plan)
          and {r["job"] for r in rows} == admitted)
    # W8′ 재고 소진 0 (반출 대상 부족으로 인한 전환이 명단 생성 단계에서 0)
    w8 = built["flow_fallbacks_total"] == 0

    # W11 피크 도달 — 측정창의 시간대별 실현 도착 최대
    hourly = [0] * int(obs.observe_s // 3600)
    for r in rows:
        h = int(r["a"] // 3600)
        if 0 <= h < len(hourly):
            hourly[h] += 1
    peak_obs = max(hourly)
    peak_plan = planned_hourly_peak(obs, built["day_total"])
    w11 = abs(peak_obs - peak_plan) <= W11_TOL * peak_plan

    # 부수 관측(판정 아님) — 측정창 완료분의 체류·벌금 구간 도달
    fin = [r for r in rows if r["o"] is not None and r["a"] is not None
           and obs.warmup_s <= r["o"] < obs.observe_s]
    stay = [r["o"] - r["a"] for r in fin]
    penal = [s for s in stay if s > SLA_PENALTY_S]
    by_hour: dict[int, list[float]] = {}
    for r in fin:
        by_hour.setdefault(int(r["a"] // 3600), []).append(r["o"] - r["a"])
    stay_curve = {h: round(fmean(v), 1) for h, v in sorted(by_hour.items())}

    # W3 결정론
    again = build_diurnal(prof, SEED + rep, obs=obs, layout=layout,
                          background_seed=SEED)
    deterministic = again["schedule"] == built["schedule"]
    # W4 본선 24시간 교대
    vs = built["vessel_schedule"]
    w4 = (len(vs) == 30 and len({v["block"] for v in vs}) >= 15
          and max(v["start_s"] for v in vs) >= 0.75 * obs.observe_s * 0.8)
    # W7 정보경계 — 예측 도착 = 통지 + 기대 주행
    leak_free = True
    for bid, sim in mbt.blocks.items():
        base = layout.gate_to_block_s(bid)
        for j in sim.jobs.values():
            if j.is_external_truck and getattr(j, "notified_gate_in_s", None):
                if abs(j.estimated_block_arrival - (j.notified_gate_in_s + base)) > 1e-6:
                    leak_free = False
    order = all((r["b"] is None or r["a"] <= r["b"] + 1e-6)
                and (r["o"] is None or r["b"] is None or r["b"] <= r["o"] + 1e-6)
                for r in rows)
    achievable = all(
        v.plan.phys_min_completion_s is not None
        and v.plan.planned_completion_s is not None
        and v.plan.planned_completion_s >= v.plan.phys_min_completion_s - 1e-6
        for s in built["scenarios"].values() for v in s.vessels)

    return {
        "rep": rep, "seed": SEED + rep,
        "profile": {"terminal_id": prof.terminal_id,
                    "transfer_kind": prof.transfer.kind},
        "n_plan": len(built["schedule"]), "n_admitted": ann.n_admitted,
        "n_tail_skipped": len(tail), "n_skips": len(skips),
        "n_registered": len(rows),
        "W1p_schedule_honored": bool(w1),
        "W2_ledger_conserved": len(rows) == ann.n_admitted,
        "W3_deterministic_build": bool(deterministic),
        "W4_vessels_spread_24h": bool(w4),
        "W5_block_snapshots": len(mbt.blocks) == 21,
        "W6_profile_h21_yt": prof.transfer.kind == "YT",
        "W7_internal": {"event_time_order": order,
                        "information_boundary": leak_free,
                        "physical_constraints": invariants,
                        "achievable_vessel_deadline": achievable},
        "W8p_no_stock_exhaustion": bool(w8),
        "W9_flow_fallback_zero": built["flow_fallbacks_total"] == 0,
        "W11_peak_reached": bool(w11),
        "peak_observed": peak_obs, "peak_planned": round(peak_plan, 1),
        "policy_exceptions": exc["n"],
        # 부수 관측 (판정 아님)
        "obs_mean_stay_s": round(fmean(stay), 1) if stay else None,
        "obs_max_stay_s": round(max(stay), 1) if stay else None,
        "obs_penalty_share": round(len(penal) / len(stay), 4) if stay else None,
        "obs_hourly_arrivals": hourly,
        "obs_hourly_mean_stay_s": stay_curve,
    }


def run(cells: list[dict] | None = None) -> dict:
    """cells 가 주어지면 병렬 셀 산출물을 합산(재실행 없음), 아니면 순차 실행."""
    from_files = cells is not None
    cells = cells if from_files else [run_cell(r) for r in range(REPS)]
    checks = {
        "W1p_schedule_honored": all(c["W1p_schedule_honored"] for c in cells),
        "W2_ledger_conserved": all(c["W2_ledger_conserved"] for c in cells),
        "W3_deterministic_build": all(c["W3_deterministic_build"] for c in cells),
        "W4_vessels_spread_24h": all(c["W4_vessels_spread_24h"] for c in cells),
        "W5_block_snapshots": all(c["W5_block_snapshots"] for c in cells),
        "W6_profile_h21_yt": all(c["W6_profile_h21_yt"] for c in cells),
        "W7_internal_checks_pass": all(all(c["W7_internal"].values()) for c in cells),
        "W8p_no_stock_exhaustion": all(c["W8p_no_stock_exhaustion"] for c in cells),
        "W9_flow_fallback_zero": all(c["W9_flow_fallback_zero"] for c in cells),
        "W11_peak_reached": all(c["W11_peak_reached"] for c in cells),
        "no_policy_exceptions": all(c["policy_exceptions"] == 0 for c in cells),
    }
    verdict = {
        "qualification_all_pass": all(checks.values()),
        "checks": checks,
        "observed_penalty_share": [c["obs_penalty_share"] for c in cells],
        "observed_mean_stay_s": [c["obs_mean_stay_s"] for c in cells],
        "note": "5차 계약 자격 — 성능 주장 없음. 벌금 구간 도달률·체류 곡선은 "
                "부수 관측이며 판정 임계가 아니다.",
    }
    res = {"task": "YR-167", "contract": "diurnal_24h",
           "runtime": {"commit": _git("rev-parse", "HEAD"),
                       "git_dirty": bool(code_dirty()),
                       "prereg_file": str(PREREG),
                       "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
                       "observation": OBS_24H.as_dict(),
                       "seeds": [SEED + r for r in range(REPS)],
                       "cells_from_files": from_files,
                       "cell_sha256": ({f"cell_rep{r}.json":
                                        _sha256(OUT / f"cell_rep{r}.json")
                                        for r in range(REPS)} if from_files else None)},
           "verdict": verdict, "cells": cells}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "diurnal_qual.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "diurnal_qual.json.sha256").write_text(_sha256(p) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=None)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarize", action="store_true",
                    help="병렬로 산출된 cell_rep*.json 을 합산(재실행 없음)")
    a = ap.parse_args()
    if a.summarize:
        run([json.loads((OUT / f"cell_rep{r}.json").read_text(encoding="utf-8"))
             for r in range(REPS)])
        print("DONE")
        raise SystemExit(0)
    if a.rep is not None:
        c = run_cell(a.rep)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"cell_rep{a.rep}.json").write_text(
            json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in c.items()
                          if not k.startswith("obs_hourly")}, ensure_ascii=False))
    elif a.run:
        run()
    print("DONE")
