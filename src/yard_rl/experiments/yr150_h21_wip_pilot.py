"""YR-150 4차 재정의 — H-21 **고정 재공량(WIP)** 자격 파일럿 (사용자 결정 2026-08-08).

■ 계약 (★32차 감사 순서 정정 2026-08-09 — 최종 SELL 환경과 동일 환경으로만 자격):
  `L ∈ {50,75,100,125,150}` = **내부 트럭 + 진입 확약된 예고 트럭 = L** (사용자 채택).
  사전 예고 lead = 30분(`ANNOUNCE_LEAD_S` — 판매 검토 창과 동일). 초기 채움 L 대 +
  부족분 60초 격자 교체 투입. 배경(본선 배치·초기 적재)은 **전 부하 셀 공통 시드**
  (`background_seed=SEED`) — L 효과가 배경 변주와 교락하지 않게(부채 3 정정).
■ 무엇을 확인하는가 (자격 — 성능 아님)
  W1 유지: 유지구간(채움 완료 후 ~ 관측종료−(lead+7분)) **격자 EPOCH 의 내부+확약**이
     L±5% 안 (계약량 = inside+pipeline. 스냅샷 내부 대수는 참고 보조)
  W2 장부 보존: 등록 = 초기 채움 + 투입, 트럭 증발 0
  W3 결정론: 같은 시드 재구성 시 채움·pool 이 동일
  W4 본선 분산 / W5 블록별 스냅샷 / W6 프로파일 = H-21 YT
  W7 내부타당성: 사건순서·정보경계(walk-in 예측 누출 0)·물리 불변식·달성가능 본선마감
  W8 pool 미소진 (소진은 자격 실패)
  W9 flow fallback 0 — **실제 투입분(채움+투입된 pool 접두) 한정**(부채 1 정정: 미사용
     pool 꼬리는 계상하지 않음)·fallback 작업별 원장을 공식 결과에 박제(부채 2 정정)
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
from ..integrated.sell_review import ANNOUNCE_LEAD_S
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
                                  lead_s=ANNOUNCE_LEAD_S,   # ★lead 30분 계약 (32차)
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


def run_cell(wip: int, obs: ObservationContract, *,
             params=None, seed: int | None = None,
             background_seed: int | None = None) -> dict:
    """자격 1셀 — params/seed 를 받으면 YR-157 hotspot 셀 등 변형 계약도 같은 검사로
    자격을 본다(기본값 = YR-150 균등 계약, 기존 호출 불변)."""
    prof = build_h21_profile()
    layout = terminal_layout()
    # ★배경 공통 시드(부채 3): 전 부하 셀이 같은 본선 배치·초기 적재를 공유한다.
    built = build_fixed_wip(prof, SEED + wip if seed is None else seed,
                            wip_target=wip, obs=obs, layout=layout, params=params,
                            background_seed=(SEED if background_seed is None
                                             else background_seed))
    mbt, ctrl, exc = _run(built, obs)
    try:
        mbt.check_invariants()
        invariants = True
    except Exception:
        invariants = False
    rows = _rows(mbt)
    snaps = snapshots(rows, obs, tuple(built["scenarios"]))
    hrs = hourly(rows, obs)

    # W1 유지 판정(★32차 재정의) — 계약량 = 내부+확약(pipeline). 관측 대상은 컨트롤러
    # 격자 EPOCH 원장이고, 유지구간은 [채움완료, 관측종료−(lead+7분)] (그 뒤는 TAIL_STOP
    # 자연 감소 구간이라 판정 제외 — 하네스 계약).
    from ..integrated.terminal_stream import WIP_FILL_SPAN_S
    hold_lo = max(obs.warmup_s, WIP_FILL_SPAN_S)
    hold_hi = obs.observe_s - (ANNOUNCE_LEAD_S + GATE_BLOCK_MAX_S)
    # 계약의 제어점 = 격자 투입 **직후**: EPOCH 원장의 inside/pipeline 은 투입 직전
    # 관측이므로 그 epoch 투입분(admitted — 즉시 pipeline 합류)을 더해 읽는다.
    # 직전 값으로 재면 60초 사이 이탈분만큼 낮게 읽혀(측정 시점 인공물) 컨트롤러가
    # 정상인데도 실패로 판정된다. 투입 실패·pool 소진은 여전히 L 미달로 잡힌다.
    held = [e["inside"] + e["pipeline"] + e["admitted"] for e in ctrl.ledger
            if e["event"] == "EPOCH" and hold_lo <= e["t"] <= hold_hi]
    lo, hi = wip * (1 - WIP_TOL_FRAC), wip * (1 + WIP_TOL_FRAC)
    maintained = bool(held) and all(lo <= h <= hi for h in held)
    inside_hold = [s["wip"] for s in snaps if hold_lo <= s["t"] <= hold_hi]

    # W3 결정론 — 재구성 대조 (변형 셀도 같은 인자로 재구성)
    again = build_fixed_wip(prof, SEED + wip if seed is None else seed,
                            wip_target=wip, obs=obs, layout=layout, params=params,
                            background_seed=(SEED if background_seed is None
                                             else background_seed))
    deterministic = again["pool"] == built["pool"] and again["fill"] == built["fill"]

    # W9(★32차 재정의) — 실제 투입분(채움 + 투입된 pool 접두)만 계상 + 작업별 원장.
    used_entries = list(built["fill"]) + list(built["pool"][:ctrl.cursor])
    fallback_jobs = [{"job_id": e["job_id"], "block": e["block"],
                      "requested": e["requested_flow"], "realized": e["flow"],
                      "reason": e["fallback_reason"]}
                     for e in used_entries if e.get("fallback_reason")]

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
        "lead_s": ANNOUNCE_LEAD_S,
        "hold_window_s": [hold_lo, hold_hi],
        "held_min": min(held) if held else None,      # 계약량 = 내부+확약 (W1 대상)
        "held_max": max(held) if held else None,
        "held_mean": round(fmean(held), 2) if held else None,
        "inside_wip_min": min(inside_hold) if inside_hold else None,   # 참고(스냅샷)
        "inside_wip_max": max(inside_hold) if inside_hold else None,
        "wip_maintained_pm5pct": maintained,
        "pool_exhausted_at": ctrl.exhausted_at,
        "n_skips": len(skips),
        "skip_reasons": sorted({e["reason"].split(":")[-1].strip()[:24]
                                for e in skips})[:5],
        "throughput_per_h": round(len(fin) / (obs.measure_s / 3600.0), 2),
        "vessel_starts_s": vessel_starts, "vessel_blocks": vessel_blocks,
        "vessel_placement": built["vessel_placement"],   # 배치 원장 (감사 2026-08-09)
        "flow_fallbacks_used": len(fallback_jobs),       # W9 대상 (실제 투입분 한정)
        "flow_fallback_jobs": fallback_jobs,             # 작업별 원장 (부채 2)
        "flow_fallbacks_drawn_total": built["flow_fallbacks_total"],   # 참고(추첨 전량)
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
        "W9_flow_fallback_zero": all(c["flow_fallbacks_used"] == 0 for c in cells),
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
                                  "ANNOUNCE_LEAD_S": ANNOUNCE_LEAD_S,
                                  "wip_contract": "내부 + 진입 확약 예고 = L (32차)",
                                  "background_seed": SEED,
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
