"""YR-150 4차 재정의 — H-21 **고정 재공량(WIP)** 자격 파일럿 (사용자 결정 2026-08-08).

■ 계약: `L ∈ {50,75,100,125,150}` = **터미널 안에 유지하는 외부트럭 대수**.
  초기 채움 L 대 + 트럭이 나갈 때마다 pool 에서 교체 투입(주기 60초).
■ 무엇을 확인하는가 (자격 — 성능 아님)
  W1 유지: 유지구간(채움 완료 후 ~ 관측종료−7분) 스냅샷 WIP 가 L±5% 안
  W2 장부 보존: 등록 = 초기 채움 + 투입, 트럭 증발 0
  W3 결정론: 같은 시드 재구성 시 채움·pool 이 동일
  W4 본선 분산 / W5 블록별 스냅샷 / W6 프로파일 = H-21 YT
  W7 내부타당성: 사건순서·정보경계(walk-in 예측 누출 0)·물리 불변식·달성가능 본선마감
  W8 pool 미소진 (소진은 자격 실패)
  W9 flow fallback 0 (감사 2026-08-09) — 반출 추첨이 재고 부족으로 반입 전환된 건이
     하나라도 있으면 자격 실패(계획 혼합비 60:40 이 조용히 틀어진 시드는 부적격)
■ **공정성 한계 (계약 박제)**: 고정 WIP 는 빨리 처리하는 정책일수록 트럭을 더 받아
  **정책별 처리 물량이 달라진다**. 성능 판정은 "같은 재공량에서 처리량 + 시간당 비용"
  공동 판정만 허용하며 에피소드 총비용 단독 비교는 금지다.
■ 이전 1단계 하네스(yr150_h21_pilot, 고정 유입량 계약)는 판정 이력 보존을 위해 불변.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference, _apply,
                                    _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.profiles import build_h21_profile
from ..integrated.repro import code_dirty
from ..integrated.scenario_gen import GATE_BLOCK_MAX_S
from ..integrated.terminal_stream import (ObservationContract,
                                          WipAdmissionController, admission_epochs,
                                          build_fixed_wip)
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from
from .yr150_h21_pilot import (_git, _sha256, classify, free_flow_a2o_s, hourly,
                              snapshots)

OUT = Path("outputs/reports/yr150_h21_wip_pilot")
PREREG = Path(".claude/docs/dashboard-task-specs/YR-150-continuous-inflow-steady-state.md")
WIP_LEVELS = (50, 75, 100, 125, 150)     # 터미널 유지 대수 (동결)
SEED = 6_200_000                         # 고정 WIP 파일럿 전용 대역
WIP_TOL_FRAC = 0.05                      # 유지 허용편차 ±5% (정의값 — 자료 적합 아님)


def _run(built: dict, obs: ObservationContract):
    mbt = MultiBlockTerminal({b: _sim_from(s) for b, s in built["scenarios"].items()},
                             extra_review_epochs=admission_epochs(obs))
    ctrl = WipAdmissionController(built["pool"], wip_target=built["wip_target"],
                                  end_s=obs.observe_s)
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0, "decisions": 0}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator())
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        exc["decisions"] += 1
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            exc["n"] += 1
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    mbt.run(policy, review_fn=ctrl.review)
    return mbt, ctrl, exc


def _rows(mbt) -> list[dict]:
    rows = []
    for bid, sim in mbt.blocks.items():
        tl = getattr(sim, "time_ledger", None)
        if tl is None:
            continue
        for jid, r in tl.records.items():
            rows.append({"job": jid, "block": bid, "a": r.gate_in,
                         "b": r.block_arrival, "o": r.gate_out})
    return rows


def run_cell(wip: int, obs: ObservationContract) -> dict:
    prof = build_h21_profile()
    layout = terminal_layout()
    built = build_fixed_wip(prof, SEED + wip, wip_target=wip, obs=obs, layout=layout)
    mbt, ctrl, exc = _run(built, obs)
    try:
        mbt.check_invariants()
        invariants = True
    except Exception:
        invariants = False
    rows = _rows(mbt)
    snaps = snapshots(rows, obs, tuple(built["scenarios"]))
    hrs = hourly(rows, obs)

    # W1 유지 판정 — 채움 완료(WIP_FILL_SPAN 이후 첫 스냅샷) ~ 관측종료−7분
    from ..integrated.terminal_stream import WIP_FILL_SPAN_S
    hold = [s for s in snaps
            if s["t"] >= max(obs.warmup_s, WIP_FILL_SPAN_S)
            and s["t"] <= obs.observe_s - GATE_BLOCK_MAX_S]
    wips = [s["wip"] for s in hold]
    lo, hi = wip * (1 - WIP_TOL_FRAC), wip * (1 + WIP_TOL_FRAC)
    maintained = bool(wips) and all(lo <= w <= hi for w in wips)

    # W3 결정론 — 재구성 대조
    again = build_fixed_wip(prof, SEED + wip, wip_target=wip, obs=obs, layout=layout)
    deterministic = again["pool"] == built["pool"] and again["fill"] == built["fill"]

    # W7 정보경계 — walk-in 예측 = gate-in + 기대 주행 (실현 잔여편차 미참조)
    leak_free = True
    for bid, sim in mbt.blocks.items():
        base = layout.gate_to_block_s(bid)
        for j in sim.jobs.values():
            if not j.is_external_truck:
                continue
            if abs(j.estimated_block_arrival - (j.appointment_gate_time + base)) > 1e-6:
                leak_free = False
    order = all((r["b"] is None or r["a"] <= r["b"] + 1e-6)
                and (r["o"] is None or r["b"] is None or r["b"] <= r["o"] + 1e-6)
                for r in rows)
    achievable = all(
        v.plan.phys_min_completion_s is not None
        and v.plan.planned_completion_s is not None
        and v.plan.planned_completion_s >= v.plan.phys_min_completion_s - 1e-6
        for s in built["scenarios"].values() for v in s.vessels)

    fin = [r for r in rows if r["o"] is not None and r["a"] is not None
           and obs.warmup_s <= r["o"] < obs.observe_s]
    mean_a2o = fmean(r["o"] - r["a"] for r in fin) if fin else None
    free_flow = free_flow_a2o_s(rows, layout, 180.0, 300.0)
    vessel_starts = sorted(round(v.plan.planned_start_s, 1)
                           for s in built["scenarios"].values() for v in s.vessels)
    vessel_blocks = sorted(b for b, s in built["scenarios"].items() if s.vessels)
    skips = [e for e in ctrl.ledger if e["event"] == "SKIP"]
    return {
        "wip_target": wip,
        "profile": {"terminal_id": prof.terminal_id,
                    "transfer_kind": prof.transfer.kind},
        "n_fill": len(built["fill"]), "n_admitted": ctrl.n_admitted,
        "n_registered": len(rows),
        "ledger_conserved": len(rows) == len(built["fill"]) + ctrl.n_admitted,
        "hold_window_s": [max(obs.warmup_s, WIP_FILL_SPAN_S),
                          obs.observe_s - GATE_BLOCK_MAX_S],
        "wip_min": min(wips) if wips else None,
        "wip_max": max(wips) if wips else None,
        "wip_mean": round(fmean(wips), 2) if wips else None,
        "wip_maintained_pm5pct": maintained,
        "pool_exhausted_at": ctrl.exhausted_at,
        "n_skips": len(skips),
        "skip_reasons": sorted({e["reason"].split(":")[-1].strip()[:24]
                                for e in skips})[:5],
        "throughput_per_h": round(len(fin) / (obs.measure_s / 3600.0), 2),
        "vessel_starts_s": vessel_starts, "vessel_blocks": vessel_blocks,
        "vessel_placement": built["vessel_placement"],   # 배치 원장 (감사 2026-08-09)
        "flow_fallbacks_total": built["flow_fallbacks_total"],
        "internal_checks": {"event_time_order": order,
                            "information_boundary": leak_free,
                            "physical_constraints": invariants,
                            "ledger_conservation": len(rows) == len(built["fill"]) + ctrl.n_admitted,
                            "achievable_vessel_deadline": achievable,
                            "deterministic_replay": deterministic},
        "snapshots": snaps, "hourly": hrs,
        "classification": classify(snaps, hrs, obs, mean_a2o_s=mean_a2o,
                                   free_flow_s=free_flow),
        "policy_exceptions": exc["n"], "decisions": exc["decisions"],
    }


def run() -> dict:
    obs = ObservationContract()
    cells = [run_cell(w, obs) for w in WIP_LEVELS]
    checks = {
        "W1_wip_maintained": all(c["wip_maintained_pm5pct"] for c in cells),
        "W2_ledger_conserved": all(c["ledger_conserved"] for c in cells),
        "W3_deterministic_build": all(
            c["internal_checks"]["deterministic_replay"] for c in cells),
        "W4_vessels_spread": all(
            len(set(c["vessel_blocks"])) >= 5
            and max(c["vessel_starts_s"]) >= 0.7 * obs.observe_s for c in cells),
        "W5_block_level_snapshots": all(
            all(len(s["wip_by_block"]) == 21 for s in c["snapshots"]) for c in cells),
        "W6_profile_is_h21_yt": all(c["profile"]["transfer_kind"] == "YT" for c in cells),
        "W7_internal_checks_pass": all(all(c["internal_checks"].values())
                                       for c in cells),
        "W8_pool_not_exhausted": all(c["pool_exhausted_at"] is None for c in cells),
        "W9_flow_fallback_zero": all(c["flow_fallbacks_total"] == 0 for c in cells),
        "no_policy_exceptions": all(c["policy_exceptions"] == 0 for c in cells),
    }
    verdict = {
        "qualification_all_pass": all(checks.values()),
        "checks": checks,
        "states": {c["wip_target"]: c["classification"]["state"] for c in cells},
        "note": "고정 WIP 자격 파일럿 — 정책 비교·성능 주장 없음. 공정성 한계: 고정 WIP "
                "에서는 정책별 처리 물량이 달라지므로 성능은 처리량+시간당 비용 공동 "
                "판정만 허용(에피소드 총비용 단독 금지).",
    }
    dirty = bool(code_dirty())
    res = {"stage": "1-WIP", "task": "YR-150", "structure": "H-21",
           "runtime": {"commit": _git("rev-parse", "HEAD"), "git_dirty": dirty,
                       "remote_ref": "origin/master",
                       "remote_head": _git("rev-parse", "origin/master"),
                       "prereg_file": str(PREREG),
                       "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
                       "params": {"WIP_LEVELS": list(WIP_LEVELS),
                                  "WIP_TOL_FRAC": WIP_TOL_FRAC,
                                  "observation": obs.as_dict(),
                                  "layout": terminal_layout().as_dict()},
                       "seeds": {"cells": [SEED + w for w in WIP_LEVELS]}},
           "verdict": verdict, "cells": cells}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "pilot_h21_wip.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "pilot_h21_wip.json.sha256").write_text(_sha256(p) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "dirty": dirty}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        obs = ObservationContract(warmup_s=1800.0, measure_s=7200.0, snapshot_s=300.0)
        c = run_cell(60, obs)
        print(json.dumps({k: v for k, v in c.items()
                          if k not in ("snapshots", "hourly")},
                         ensure_ascii=False, indent=1))
    elif a.run:
        run()
    print("DONE")
