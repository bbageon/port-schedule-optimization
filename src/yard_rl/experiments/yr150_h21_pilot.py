"""YR-150 1단계 — H-21 21블록 지속 유입 **자격 파일럿** (정책 비교 없음).

■ 무엇을 확인하는가 (spec 실험 사다리 1단계)
  Q1 터미널 master stream 하나가 `p` 로 21블록에 배정되고 합계가 부풀지 않는가
  Q2 유입이 **관측시간 끝까지** 이어지고 관측시간에서 그대로 종료하는가(미완 보존)
  Q3 5~10분 스냅샷과 1시간 구간 집계가 장부와 정합한가
  Q4 본선이 블록·관측창 전체에 분산됐는가
  Q5 도착−완료 균형·장부 보존(트럭이 증발하지 않는가)
  Q6 부하 5종의 상태를 **결과로 사후 분류**(CLEAR/용량초과)하는가
■ **금지**: 정책 비교·성능 주장. 정책은 규칙(SF-SPT) 하나만 쓴다. 이 단계 수치는
  "환경이 계약대로 도는가"의 근거이지 "비용이 줄었다"의 근거가 아니다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import fmean

from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference, _apply,
                                    _wait_of)
from ..integrated.block_congestion import SVC_REF_S
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.profiles import build_h21_profile
from ..integrated.repro import code_dirty
from ..integrated.terminal_stream import (LOAD_WINDOW_S, ObservationContract,
                                          TerminalStreamParams, build_terminal)
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from

OUT = Path("outputs/reports/yr150_h21_pilot")
PREREG = Path(".claude/docs/dashboard-task-specs/YR-150-continuous-inflow-steady-state.md")
LOADS = (50, 75, 100, 125, 150)      # 터미널 전체 4시간 유입량 (spec 동결)
SEED = 5_200_000                     # 파일럿 전용 — 확증은 겹치지 않는 대역에서


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _slope(xs: list[float], ys: list[float]) -> float:
    """최소제곱 기울기 — backlog 가 발산하는지 보는 지표(시간당 대수)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = fmean(xs), fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return 0.0 if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def _run_terminal(built: dict, obs: ObservationContract) -> dict:
    """규칙 정책으로 21블록을 돌리고 구간별 비용을 모은다 (이송 없음 — 자격 확인용)."""
    mbt = MultiBlockTerminal({b: _sim_from(s) for b, s in built["scenarios"].items()})
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

    res = mbt.run(policy)
    return {"mbt": mbt, "run": res, "policy_exceptions": exc["n"],
            "decisions": exc["decisions"]}


def _ledger_rows(mbt: MultiBlockTerminal) -> list[dict]:
    """전 블록 트럭 장부를 A(게이트 진입)·O(출문) 로 모은다 — 미완은 O=None 으로 남긴다."""
    rows = []
    for bid, sim in mbt.blocks.items():
        tl = getattr(sim, "time_ledger", None)
        if tl is None:
            continue
        for jid, r in tl.records.items():
            rows.append({"job": jid, "block": bid, "a": r.gate_in,
                         "b": r.block_arrival, "o": r.gate_out})
    return rows


def snapshots(rows: list[dict], obs: ObservationContract,
              blocks: tuple[str, ...]) -> list[dict]:
    """5~10분 스냅샷 — 재공량 WIP(t) = #{A ≤ t < O} 를 장부에서 **정확히** 재구성한다.

    표본추출이 아니라 사건시각에서 직접 세므로 근사 오차가 없다(라이브 훅과 같은 값).
    **터미널 합계와 블록별을 모두 남긴다** — 합계만 보면 "어떤 블록이 순간적으로 붐비고
    어떤 블록이 비었는지"를 알 수 없고, 그 불균형이 바로 SELL 연구의 대상이다
    (외부 감사 지적 2026-08-06: 합계만으로 "모든 블록이 늘 한가하다"고 말할 수 없다).
    """
    out = []
    for t in obs.snapshot_times():
        per: dict[str, int] = {b: 0 for b in blocks}
        wip = arrived = done = 0
        for r in rows:
            a, o = r["a"], r["o"]
            if a is not None and a <= t:
                arrived += 1
                if o is None or o > t:
                    wip += 1
                    per[r["block"]] = per.get(r["block"], 0) + 1
            if o is not None and o <= t:
                done += 1
        vals = sorted(per.values())
        out.append({"t": round(t, 3), "wip": wip, "arrived": arrived, "done": done,
                    "wip_by_block": per,
                    "block_wip_max": vals[-1], "block_wip_min": vals[0],
                    "block_wip_spread": vals[-1] - vals[0],
                    "n_blocks_idle": sum(1 for v in vals if v == 0)})
    return out


def hourly(rows: list[dict], obs: ObservationContract) -> list[dict]:
    """1시간 구간 집계 — 구간별 도착·완료·평균 A→O."""
    out = []
    n = int(obs.measure_s // 3600.0)
    for i in range(n):
        t0 = obs.warmup_s + i * 3600.0
        t1 = t0 + 3600.0
        arr = [r for r in rows if r["a"] is not None and t0 <= r["a"] < t1]
        fin = [r for r in rows if r["o"] is not None and t0 <= r["o"] < t1]
        a2o = [r["o"] - r["a"] for r in fin if r["a"] is not None]
        out.append({"hour": i, "t0": t0, "t1": t1, "arrivals": len(arr),
                    "completions": len(fin),
                    "mean_a2o_s": round(fmean(a2o), 1) if a2o else None})
    return out


BUSY_CONGESTION_RATIO = 1.5     # 자유흐름 대비 체류시간 배수 — 정의값(자료 적합 아님)


def free_flow_a2o_s(rows: list[dict], layout, svc_ref_s: float,
                    exit_mu_s: float) -> float:
    """대기가 전혀 없을 때의 A→O — **계약값에서 유도**한다(관측 최소값이 아니다).

    게이트→블록 주행(블록별) + 기준 서비스시간 + 출문 주행. 관측치를 쓰지 않으므로
    "결과를 보고 임계를 정하는" 문제가 생기지 않는다.
    """
    if not rows:
        return float("nan")
    travel = fmean(layout.gate_to_block_s(r["block"]) for r in rows)
    return travel + svc_ref_s + exit_mu_s


def classify(snaps: list[dict], hrs: list[dict], obs: ObservationContract,
             *, mean_a2o_s: float | None, free_flow_s: float) -> dict:
    """상태를 **결과로** 3구간으로 분류한다 — 이름표(양호·혼잡)를 미리 붙이지 않는다.

    · OVERLOADED : 후반 재공량이 발산하거나 처리량이 유입을 못 따라감(불안정)
    · BUSY       : 안정적이지만 체류시간이 자유흐름의 1.5배 이상 — **성능 주판정 구간**
    · CLEAR      : 그 외(여유)

    BUSY 임계 1.5배는 **정의**이지 자료에 맞춘 값이 아니다. 자유흐름 기준선을 계약값
    (주행+기준 서비스+출문)에서 유도하므로 관측 결과가 임계를 움직이지 못한다.
    (외부 감사 지적 2026-08-06 — 2구간으로는 성능시험 구간을 지목할 수 없다.)
    """
    half = len(snaps) // 2
    late = snaps[half:]
    xs = [s["t"] / 3600.0 for s in late]
    slope = _slope(xs, [float(s["wip"]) for s in late])
    arr = sum(h["arrivals"] for h in hrs)
    fin = sum(h["completions"] for h in hrs)
    diverging = slope > 1.0                      # 시간당 재공량 +1대 이상 누적
    starved = fin < 0.9 * arr                    # 처리량이 유입을 못 따라감
    ratio = (None if not mean_a2o_s or not free_flow_s or free_flow_s != free_flow_s
             else mean_a2o_s / free_flow_s)
    if diverging or starved:
        state = "OVERLOADED"
    elif ratio is not None and ratio >= BUSY_CONGESTION_RATIO:
        state = "BUSY"
    else:
        state = "CLEAR"
    return {"late_wip_slope_per_h": round(slope, 4),
            "measure_arrivals": arr, "measure_completions": fin,
            "completion_ratio": round(fin / arr, 4) if arr else None,
            "free_flow_a2o_s": round(free_flow_s, 1),
            "mean_a2o_s": None if mean_a2o_s is None else round(mean_a2o_s, 1),
            "congestion_ratio": None if ratio is None else round(ratio, 3),
            "busy_threshold_ratio": BUSY_CONGESTION_RATIO,
            "state": state,
            "rule": ("불안정(후반 재공량 기울기>1/h 또는 처리량<유입 90%)=OVERLOADED, "
                     "안정이면서 체류시간≥자유흐름×1.5=BUSY, 그 외 CLEAR")}


def internal_checks(mbt, built: dict, rows: list[dict], obs: ObservationContract,
                    load: int) -> dict:
    """YR-153 현실성 게이트가 요구하는 **내부타당성 6종**을 실제로 확인해 제출한다."""
    layout = terminal_layout()
    # ① 사건 시각 순서 A ≤ B ≤ O
    order = all((r["b"] is None or r["a"] <= r["b"] + 1e-6)
                and (r["o"] is None or r["b"] is None or r["b"] <= r["o"] + 1e-6)
                for r in rows if r["a"] is not None)
    # ② 정보경계 — 예측 블록도착이 실현이 아니라 예약+기대 주행으로만 만들어졌는가
    leak_free = True
    for b, scn in built["scenarios"].items():
        for j in scn.jobs:
            if not j.is_external_truck:
                continue
            want = j.appointment_gate_time + layout.gate_to_block_s(b)
            if abs(j.estimated_block_arrival - want) > 1e-6 or abs(j.provided_eta - want) > 1e-6:
                leak_free = False
    # ③ 물리제약 — 엔진 불변식(크레인 안전·적재·용량·소유권)
    try:
        mbt.check_invariants()
        physical = True
    except Exception:
        physical = False
    # ④ 장부 보존 — 배정된 트럭 수와 등록 수가 같다
    assigned = sum(len([j for j in s.jobs if j.is_external_truck])
                   for s in built["scenarios"].values())
    conserved = len(rows) == assigned == built["n_total"]
    # ⑤ 달성 가능한 본선 마감 — 계획완료가 물리 최소완료 이상
    achievable = all(
        v.plan.phys_min_completion_s is not None
        and v.plan.planned_completion_s is not None
        and v.plan.planned_completion_s >= v.plan.phys_min_completion_s - 1e-6
        for s in built["scenarios"].values() for v in s.vessels)
    # ⑥ 결정론 재현 — 같은 시드로 다시 만들면 배정 원장이 같다
    again = build_terminal(build_h21_profile(), SEED + load,
                           params=TerminalStreamParams(load_4h=load), obs=obs,
                           layout=layout)
    deterministic = again["assignment"] == built["assignment"]
    return {"event_time_order": order, "information_boundary": leak_free,
            "physical_constraints": physical, "ledger_conservation": conserved,
            "achievable_vessel_deadline": achievable,
            "deterministic_replay": deterministic}


def run_cell(load: int, obs: ObservationContract) -> dict:
    # ★H-21 은 **YT** 구조다 — 구 파일럿은 AGV fleet 인 build_calibrated_profile 을 써서
    #   코드와 Dashboard 정의가 어긋나 있었다(외부 감사 지적 2026-08-06).
    prof = build_h21_profile()
    layout = terminal_layout()
    params = TerminalStreamParams(load_4h=load)
    built = build_terminal(prof, SEED + load, params=params, obs=obs, layout=layout)
    out = _run_terminal(built, obs)
    mbt = out["mbt"]
    rows = _ledger_rows(mbt)
    snaps = snapshots(rows, obs, tuple(built["scenarios"]))
    hrs = hourly(rows, obs)

    assigned = {b: len([j for j in s.jobs if j.is_external_truck])
                for b, s in built["scenarios"].items()}
    ext_total = sum(assigned.values())
    last_arrival = max((r["a"] for r in rows if r["a"] is not None), default=None)
    vessel_starts = sorted(round(v.plan.planned_start_s, 1)
                           for s in built["scenarios"].values() for v in s.vessels)
    vessel_blocks = sorted(b for b, s in built["scenarios"].items() if s.vessels)

    # ★L 의 정확한 뜻: **명목 예약 도착강도**다. 실제 gate-in 은 예약 준수오차로 측정창
    #   경계를 넘나들어 L 과 정확히 같지 않다(외부 감사 정정 2026-08-06).
    t0, t1 = obs.warmup_s, obs.observe_s
    nominal = sum(1 for s in built["scenarios"].values() for j in s.jobs
                  if j.is_external_truck and t0 <= j.appointment_gate_time < t1)
    actual = sum(1 for r in rows if r["a"] is not None and t0 <= r["a"] < t1)

    fin = [r for r in rows if r["o"] is not None and r["a"] is not None
           and t0 <= r["o"] < t1]
    mean_a2o = fmean(r["o"] - r["a"] for r in fin) if fin else None
    free_flow = free_flow_a2o_s(rows, layout, SVC_REF_S,
                                params.exit_travel_mu_s)
    blk = [s["block_wip_max"] for s in snaps]
    return {
        "load_4h": load,
        "profile": {"terminal_id": prof.terminal_id, "transfer_kind": prof.transfer.kind,
                    "transfer_units": prof.transfer.n_units,
                    "transfer_move_time_s": prof.transfer.move_time_s},
        "n_total_expected": built["n_total"],
        "n_assigned": ext_total,
        "assignment_sums_to_stream": ext_total == built["n_total"],
        "per_block_assigned": assigned,
        "distribution": "uniform" if not params.hotspot_blocks else "hotspot",
        "ledger_registered": len(rows),
        "ledger_matches_assigned": len(rows) == ext_total,
        "nominal_appointments_in_window": nominal,
        "actual_gate_in_in_window": actual,
        "nominal_vs_actual_note": ("L 은 명목 **예약** 도착강도다. 실제 gate-in 은 예약 "
                                   "준수오차로 측정창 경계를 넘나들어 L 과 다를 수 있다."),
        "last_arrival_s": None if last_arrival is None else round(last_arrival, 1),
        "observe_s": obs.observe_s,
        "arrivals_reach_end": (last_arrival is not None
                               and last_arrival >= 0.9 * obs.observe_s),
        "unfinished_at_end": sum(1 for r in rows if r["o"] is None),
        "vessel_starts_s": vessel_starts, "vessel_blocks": vessel_blocks,
        "n_blocks_without_vessel": len(built["scenarios"]) - len(vessel_blocks),
        "block_wip_max_over_window": max(blk) if blk else 0,
        "block_wip_spread_max": max((s["block_wip_spread"] for s in snaps), default=0),
        "min_blocks_idle_at_any_snapshot": min((s["n_blocks_idle"] for s in snaps),
                                               default=None),
        "snapshots": snaps, "hourly": hrs,
        "internal_checks": internal_checks(mbt, built, rows, obs, load),
        "classification": classify(snaps, hrs, obs, mean_a2o_s=mean_a2o,
                                   free_flow_s=free_flow),
        "policy_exceptions": out["policy_exceptions"], "decisions": out["decisions"],
    }


ANCHORS = Path("configs/anchors/external_anchors_v1.json")


def submit_scenario_gate(cells: list[dict], obs: ObservationContract) -> dict:
    """현실성 게이트에 **실제로 제출**한다 — 통과를 위해 값을 지어내지 않는다.

    등록부에 근거가 있는 앵커만 넣는다. 나머지는 비워 두고 게이트가 '미수집'으로
    실패하게 둔다. 실패도 결과이며, 그 실패 목록이 곧 "무엇을 더 구해야 하는가"다.
    """
    from .gate_harness import AnchorEvidence, judge_scenario_validity

    reg = json.loads(ANCHORS.read_text(encoding="utf-8"))
    layout = terminal_layout()
    lo, hi = layout.gate_time_range_s()
    anchors: dict[str, AnchorEvidence] = {}
    rec = reg["anchors"].get("gate_to_block_time")
    if rec:
        anchors["gate_to_block_time"] = AnchorEvidence(
            observed_min=float(rec["observed_range"][0]),
            observed_max=float(rec["observed_range"][1]),
            simulated_min=lo, simulated_max=hi, unit=rec["unit"],
            source_path=str(ANCHORS.as_posix()), source_sha256=_sha256(ANCHORS))

    merged = {k: all(c["internal_checks"][k] for c in cells)
              for k in cells[0]["internal_checks"]}
    states = {c["load_4h"]: c["classification"]["state"] for c in cells}
    flow = {
        "continuous_arrivals": all(c["arrivals_reach_end"] for c in cells),
        "warmup_excluded": all(h["t0"] >= obs.warmup_s for c in cells
                               for h in c["hourly"]),
        "fixed_measurement_window": all(c["observe_s"] == obs.observe_s for c in cells),
        "load_state_classified": all(v in ("CLEAR", "BUSY", "OVERLOADED")
                                     for v in states.values()),
        "flow_balance_consistent_with_classification": all(
            (c["classification"]["state"] != "OVERLOADED")
            == (c["classification"]["completion_ratio"] >= 0.9) for c in cells),
    }
    outcome = judge_scenario_validity(
        internal_checks=merged, flow_checks=flow, anchors=anchors,
        continuous_operation=True, request_real_terminal_claim=False)
    return {"outcome": outcome.as_dict(),
            "anchor_registry": str(ANCHORS.as_posix()),
            "anchor_registry_sha256": _sha256(ANCHORS),
            "anchors_unavailable": sorted(reg.get("unavailable", {})),
            "note": "이 제출은 통과를 목표로 하지 않는다 — 근거 없는 앵커를 지어내지 "
                    "않으므로 미수집 항목이 있으면 그대로 실패한다."}


def run() -> dict:
    obs = ObservationContract()
    cells = [run_cell(L, obs) for L in LOADS]
    gate = submit_scenario_gate(cells, obs)
    checks = {
        "Q1_stream_split_exact": all(c["assignment_sums_to_stream"] for c in cells),
        "Q2_arrivals_reach_end": all(c["arrivals_reach_end"] for c in cells),
        "Q3_snapshot_grid_ok": all(
            len(c["snapshots"]) == int(obs.measure_s // obs.snapshot_s) + 1
            and c["snapshots"][-1]["t"] == obs.warmup_s + obs.measure_s
            for c in cells),
        "Q4_vessels_spread": all(
            len(set(c["vessel_blocks"])) >= 5
            and max(c["vessel_starts_s"]) >= 0.7 * obs.observe_s for c in cells),
        "Q5_ledger_preserved": all(c["ledger_matches_assigned"] for c in cells),
        "Q6_state_classified_posthoc": all(
            c["classification"]["state"] in ("CLEAR", "BUSY", "OVERLOADED")
            for c in cells),
        "Q7_block_level_snapshots": all(
            all("wip_by_block" in s and len(s["wip_by_block"]) == 21
                for s in c["snapshots"]) for c in cells),
        "Q8_profile_is_h21_yt": all(c["profile"]["transfer_kind"] == "YT"
                                    and c["profile"]["terminal_id"] == "H21-SHARED-YT"
                                    for c in cells),
        "Q9_internal_checks_pass": all(all(c["internal_checks"].values())
                                       for c in cells),
        "no_policy_exceptions": all(c["policy_exceptions"] == 0 for c in cells),
    }
    verdict = {
        "qualification_all_pass": all(checks.values()),
        "checks": checks,
        "states": {c["load_4h"]: c["classification"]["state"] for c in cells},
        "scenario_gate_status": gate["outcome"]["status"],
        "scenario_gate_reasons": gate["outcome"]["reasons"],
        "note": "자격 파일럿 전용 — 정책 비교·성능 주장 없음. 부하 이름표는 사전에 붙이지 "
                "않고 실현된 재공량·처리량으로 사후 분류했다. **자격 통과는 현실성 게이트 "
                "통과와 다르다** — 게이트 판정은 위 scenario_gate_status 를 본다.",
        "scope_limits": [
            "균등 배분 1종만 검사 — hotspot 배분은 미검사(계약상 분리 필요)",
            "부하별로 시드가 다르므로 부하 효과의 인과 비교가 아니다(방향성만)",
            "각 부하 1시드 — 복수 독립 시드 아님",
            "본선이 21블록 중 일부에만 배치돼 나머지 블록에는 본선 경쟁이 없다",
        ],
    }
    # ★미추적 신규 파일까지 본다 — `--untracked-files=no` 는 새 실험 코드를 통째로
    #   "깨끗하다"고 보고해 재현 사슬을 조용히 끊는다(2026-08-06 실측).
    dirty = bool(code_dirty())
    res = {"stage": "1", "task": "YR-150", "structure": "H-21",
           "runtime": {"commit": _git("rev-parse", "HEAD"), "git_dirty": dirty,
                       "remote_ref": "origin/master",
                       "remote_head": _git("rev-parse", "origin/master"),
                       "prereg_file": str(PREREG),
                       "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
                       "params": {"LOADS": list(LOADS), "load_window_s": LOAD_WINDOW_S,
                                  "observation": obs.as_dict(),
                                  "terminal_blocks": 21,
                                  "layout": terminal_layout().as_dict(),
                                  "profile": {"builder": "build_h21_profile",
                                              "terminal_id": "H21-SHARED-YT",
                                              "transfer_kind": "YT"},
                                  "stream": dataclass_dict(TerminalStreamParams(load_4h=0))},
                       "seeds": {"cells": [SEED + L for L in LOADS]}},
           "verdict": verdict, "scenario_gate": gate, "cells": cells}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "pilot_h21.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "pilot_h21.json.sha256").write_text(_sha256(p) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "dirty": dirty}, ensure_ascii=False))
    return res


def dataclass_dict(obj) -> dict:
    import dataclasses
    return {k: v for k, v in dataclasses.asdict(obj).items() if k != "load_4h"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        obs = ObservationContract()
        c = run_cell(50, obs)
        print(json.dumps({k: v for k, v in c.items()
                          if k not in ("snapshots", "per_block_assigned")},
                         ensure_ascii=False, indent=1))
    elif a.run:
        run()
    print("DONE")
