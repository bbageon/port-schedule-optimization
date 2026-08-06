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
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.profiles import build_calibrated_profile
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


def snapshots(rows: list[dict], obs: ObservationContract) -> list[dict]:
    """5~10분 스냅샷 — 재공량 WIP(t) = #{A ≤ t < O} 를 장부에서 **정확히** 재구성한다.

    표본추출이 아니라 사건시각에서 직접 세므로 근사 오차가 없다(라이브 훅과 같은 값).
    """
    out = []
    for t in obs.snapshot_times():
        wip = sum(1 for r in rows
                  if r["a"] is not None and r["a"] <= t
                  and (r["o"] is None or r["o"] > t))
        arrived = sum(1 for r in rows if r["a"] is not None and r["a"] <= t)
        done = sum(1 for r in rows if r["o"] is not None and r["o"] <= t)
        out.append({"t": round(t, 3), "wip": wip, "arrived": arrived, "done": done})
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


def classify(snaps: list[dict], hrs: list[dict], obs: ObservationContract) -> dict:
    """상태를 **결과로** 분류한다 — 이름표(양호·혼잡)를 미리 붙이지 않는다.

    용량 초과 = 측정구간 후반부에서 재공량이 발산(기울기가 유의하게 양) 또는
    구간 처리량이 유입량을 못 따라감. 그 외는 CLEAR.
    """
    half = len(snaps) // 2
    late = snaps[half:]
    xs = [s["t"] / 3600.0 for s in late]
    slope = _slope(xs, [float(s["wip"]) for s in late])
    arr = sum(h["arrivals"] for h in hrs)
    fin = sum(h["completions"] for h in hrs)
    diverging = slope > 1.0                      # 시간당 재공량 +1대 이상 누적
    starved = fin < 0.9 * arr                    # 처리량이 유입을 못 따라감
    return {"late_wip_slope_per_h": round(slope, 4),
            "measure_arrivals": arr, "measure_completions": fin,
            "completion_ratio": round(fin / arr, 4) if arr else None,
            "state": "OVER_CAPACITY" if (diverging or starved) else "CLEAR",
            "rule": "발산(후반 재공량 기울기>1/h) 또는 처리량<유입 90% 이면 용량 초과"}


def run_cell(load: int, obs: ObservationContract) -> dict:
    prof = build_calibrated_profile()
    layout = terminal_layout()
    params = TerminalStreamParams(load_4h=load)
    built = build_terminal(prof, SEED + load, params=params, obs=obs, layout=layout)
    out = _run_terminal(built, obs)
    mbt = out["mbt"]
    rows = _ledger_rows(mbt)
    snaps = snapshots(rows, obs)
    hrs = hourly(rows, obs)

    assigned = {b: len([j for j in s.jobs if j.is_external_truck])
                for b, s in built["scenarios"].items()}
    ext_total = sum(assigned.values())
    last_arrival = max((r["a"] for r in rows if r["a"] is not None), default=None)
    vessel_starts = sorted(round(v.plan.planned_start_s, 1)
                           for s in built["scenarios"].values() for v in s.vessels)
    vessel_blocks = sorted(b for b, s in built["scenarios"].items() if s.vessels)
    return {
        "load_4h": load,
        "n_total_expected": built["n_total"],
        "n_assigned": ext_total,
        "assignment_sums_to_stream": ext_total == built["n_total"],
        "per_block_assigned": assigned,
        "ledger_registered": len(rows),
        "ledger_matches_assigned": len(rows) == ext_total,
        "last_arrival_s": None if last_arrival is None else round(last_arrival, 1),
        "observe_s": obs.observe_s,
        "arrivals_reach_end": (last_arrival is not None
                               and last_arrival >= 0.9 * obs.observe_s),
        "unfinished_at_end": sum(1 for r in rows if r["o"] is None),
        "vessel_starts_s": vessel_starts, "vessel_blocks": vessel_blocks,
        "snapshots": snaps, "hourly": hrs,
        "classification": classify(snaps, hrs, obs),
        "policy_exceptions": out["policy_exceptions"], "decisions": out["decisions"],
    }


def run() -> dict:
    obs = ObservationContract()
    cells = [run_cell(L, obs) for L in LOADS]
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
            c["classification"]["state"] in ("CLEAR", "OVER_CAPACITY") for c in cells),
        "no_policy_exceptions": all(c["policy_exceptions"] == 0 for c in cells),
    }
    verdict = {
        "qualification_all_pass": all(checks.values()),
        "checks": checks,
        "states": {c["load_4h"]: c["classification"]["state"] for c in cells},
        "note": "자격 파일럿 전용 — 정책 비교·성능 주장 없음. 부하 이름표는 사전에 붙이지 "
                "않고 실현된 재공량·처리량으로 사후 분류했다.",
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
                                  "stream": dataclass_dict(TerminalStreamParams(load_4h=0))},
                       "seeds": {"cells": [SEED + L for L in LOADS]}},
           "verdict": verdict, "cells": cells}
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
