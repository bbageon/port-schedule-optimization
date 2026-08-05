"""YR-149 3단계 선결 — 5부하 셀 데이터 자격시험 (정책 비교 전, 동결 2026-08-05).

■ 셀: A/B = 50/50 · 75/50 · 100/50 · 125/50 · 150/50 (터미널 합계 100·125·150·175·200).
  A 는 A150 master 의 중첩추출, B 는 전 셀 동일. 실행정책은 SF-SPT·이송 없음(순수 데이터).
■ 자격 검사(하드): ①A 중첩 A50⊂A75⊂A100⊂A125⊂A150 ②A 유지 작업의 필드 바이트 동일
  ③B 5셀 완전 동일(작업·ETA·실현·배치·본선) ④초기 적재·본선 지문 셀 간 동일
  ⑤정책 예외 0 ⑥장부 항등(A 등록 수 = 외부트럭 수).
■ 원자료: 시간가중 재공량(A≤t<O)·블록 점유·결정시점 최대 대기·평균 A→O·YC 가동률·
  처리량·4h/6h 잔여·비움시각.
■ 용량상태(결과 후 기계 분류): CLEAR(4h·6h 잔여 0) / BUSY(4h>0·6h 0) / OVERLOADED(6h>0).
  부하 이름(50~150)은 입력값일 뿐 혼잡 판정이 아니다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from statistics import fmean

from ..integrated import TerminalSimulator
from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference, _apply,
                                    _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.load_cells import (B_SIZE, CELL_SIZES, MASTER_SIZE, config_digest,
                                     ext_job_rows, generate_block, make_a_cell,
                                     make_b_cell)
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.profiles import build_calibrated_profile
from ..integrated.repro import repro_stamp
from ..integrated.seedbank import assign_band, independence_report, realization_hash
from .yr088_joint_rl import LEVEL
from .yr138_episode_pilot import _v2_hard_total

OUT = Path("outputs/reports/yr149_load_cells")
BAND_PATH = OUT / "band.json"
BAND_START, N_PAIRS = 907_000, 4


def _sim_from(scn) -> TerminalSimulator:
    s = TerminalSimulator(build_calibrated_profile(), scn, check_invariants=True)
    s.info_level = LEVEL
    return s


def _gen_for_band(key, _params, seed):
    return generate_block(seed, MASTER_SIZE if key == "A" else B_SIZE)


def make_band():
    import re
    pat = re.compile(r"rz1:[0-9a-f]{16}")
    exclude: set[str] = set()
    for p in Path("outputs/reports").rglob("*.json"):
        if OUT in p.parents:
            continue
        try:
            exclude |= set(pat.findall(p.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    band = assign_band(family="y149-cells", cells={"A": None, "B": None}, n=N_PAIRS,
                       generate=_gen_for_band, exclude=exclude, start_seed=BAND_START)
    rep = independence_report(band, forbidden={"past-recorded": exclude})
    assert rep["ok"], rep
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    BAND_PATH.write_text(json.dumps(
        {**band.freeze_json(), "independence": rep, "n_excluded_hashes": len(exclude),
         "created_commit": head}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[band] master {N_PAIRS} 쌍 frozen (A={MASTER_SIZE}·B={B_SIZE})")


def run_cell(a_seed: int, b_seed: int, m: int) -> dict:
    """SF-SPT·이송 없음으로 한 셀을 돌리고 원자료를 남긴다."""
    a_scn, b_scn = make_a_cell(a_seed, m), make_b_cell(b_seed)
    mbt = MultiBlockTerminal({"A": _sim_from(a_scn), "B": _sim_from(b_scn)})
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    st = {"exc": 0, "dec": 0, "busy_s": 0.0, "qmax": 0}
    horizon = a_scn.horizon_s
    snap4h: dict[int, int] = {}
    _bid = {id(s): b for b, s in mbt.blocks.items()}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator())
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        st["dec"] += 1
        st["qmax"] = max(st["qmax"], sim.kpis.waiting_count())
        if sim.now >= horizon and id(sim) not in snap4h:
            snap4h[id(sim)] = sim.unfinished_backlog()
        try:
            assign = pol.decide(sim, dp, gb)
        except Exception:
            st["exc"] += 1
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})
            return
        for c in dp.crane_ids:                       # YC 가동시간 = 확정 plan 소요 합
            p = getattr(assign[c], "plan", None)
            if p is not None:
                st["busy_s"] += float(p.duration_s)
        _apply(sim, assign)

    res = mbt.run(policy, None, lambda sim, t0, t1, raw: 0.0)
    mbt.check_invariants()
    end = float(res["end"])
    n_cranes = sum(len(s.profile.cranes) for s in mbt.blocks.values())
    recs = list(mbt.ledger.records.values())
    a2o = mbt.ledger.a_to_o_samples_s(end)
    o_all = [r.o_gate_out for r in recs if r.o_gate_out is not None]
    n_ext_reg = sum(1 for r in recs if r.a_gate_in is not None)
    term_area = sum(s.time_ledger.terminal_area_s for s in mbt.blocks.values()
                    if s.time_ledger)
    block_area = sum(s.time_ledger.block_area_s for s in mbt.blocks.values()
                     if s.time_ledger)
    done = sum(1 for s in mbt.blocks.values()
               for j in s.jobs.values() if j.status.name == "DONE")
    n_jobs = sum(len(s.jobs) for s in mbt.blocks.values())
    backlog_6h = sum(s.unfinished_backlog() for s in mbt.blocks.values())
    backlog_4h = sum(snap4h.values()) if snap4h else None
    complete_all = len(o_all) == n_ext_reg and backlog_6h == 0
    state = ("OVERLOADED" if backlog_6h > 0
             else "BUSY" if (backlog_4h or 0) > 0 else "CLEAR")
    return {"cell": f"A{m}/B{B_SIZE}", "m": m, "a_seed": a_seed, "b_seed": b_seed,
            "capacity_state": state,
            "wip_time_weighted": term_area / end,
            "block_occupancy_time_weighted": block_area / end,
            "queue_max_at_decisions": st["qmax"],
            "a2o_mean_min": (fmean(a2o) / 60.0) if a2o else None,
            "n_a2o_samples": len(a2o), "n_ext_registered": n_ext_reg,
            "yc_busy_frac": st["busy_s"] / (n_cranes * end),
            "throughput_per_h": done / (end / 3600.0),
            "backlog_4h": backlog_4h, "backlog_6h": backlog_6h,
            "drain_to_empty_s": (max(o_all) if complete_all and o_all else None),
            "completion_rate": done / max(1, n_jobs),
            "policy_exceptions": st["exc"], "decisions": st["dec"],
            "v2_total": sum(_v2_hard_total(s) for s in mbt.blocks.values()),
            "config": {"A": config_digest(a_scn), "B": config_digest(b_scn)}}


def qualify() -> dict:
    band = json.loads(BAND_PATH.read_text(encoding="utf-8"))
    pairs = list(zip(band["seeds"]["A"], band["seeds"]["B"]))
    checks, rows = [], []
    for pi, (sa, sb) in enumerate(pairs):
        # --- 자격 검사 ① 중첩 ② 유지 작업 필드 동일 ③ B 전 셀 동일 ④ 배치·본선 동일
        a_cells = {m: make_a_cell(sa, m) for m in CELL_SIZES}
        a_rows = {m: {r[0]: r for r in ext_job_rows(a_cells[m])} for m in CELL_SIZES}
        nested_ok, fields_ok = True, True
        for small, big in zip(CELL_SIZES, CELL_SIZES[1:]):
            if not set(a_rows[small]) <= set(a_rows[big]):
                nested_ok = False
            for jid, row in a_rows[small].items():
                if jid in a_rows[big] and a_rows[big][jid] != row:
                    fields_ok = False
        b_digests = {m: config_digest(make_b_cell(sb)) for m in CELL_SIZES}
        b_same = len({json.dumps(d, sort_keys=True) for d in b_digests.values()}) == 1
        a_layout_same = len({config_digest(a_cells[m])["layout"]
                             for m in CELL_SIZES}) == 1
        a_vessel_same = len({config_digest(a_cells[m])["vessels"]
                             for m in CELL_SIZES}) == 1
        size_ok = all(len(a_rows[m]) == m for m in CELL_SIZES)
        checks.append({"pair": pi, "a_seed": sa, "b_seed": sb,
                       "nested_subset": nested_ok, "kept_job_fields_identical": fields_ok,
                       "b_cell_identical_across_cells": b_same,
                       "a_layout_identical": a_layout_same,
                       "a_vessel_identical": a_vessel_same,
                       "exact_sizes": size_ok,
                       "flow_ratio_by_cell": {m: config_digest(a_cells[m])
                                              ["flow_ratio_gate_out"]
                                              for m in CELL_SIZES},
                       "a_master_hash": realization_hash(
                           generate_block(sa, MASTER_SIZE)),
                       "b_hash": realization_hash(make_b_cell(sb))})
        for m in CELL_SIZES:
            print(f"[cells] pair{pi} A{m}", flush=True)
            r = run_cell(sa, sb, m)
            r["pair"] = pi
            rows.append(r)
    hard = {"nested_subset": all(c["nested_subset"] for c in checks),
            "kept_job_fields_identical": all(c["kept_job_fields_identical"]
                                             for c in checks),
            "b_cell_identical": all(c["b_cell_identical_across_cells"] for c in checks),
            "a_layout_identical": all(c["a_layout_identical"] for c in checks),
            "a_vessel_identical": all(c["a_vessel_identical"] for c in checks),
            "exact_sizes": all(c["exact_sizes"] for c in checks),
            "policy_exceptions_zero": all(r["policy_exceptions"] == 0 for r in rows),
            "ledger_identity": all(r["n_ext_registered"] == r["m"] + B_SIZE
                                   for r in rows)}
    hard["all_pass"] = all(hard.values())
    by_cell = {}
    for m in CELL_SIZES:
        sub = [r for r in rows if r["m"] == m]
        states = [r["capacity_state"] for r in sub]
        by_cell[f"A{m}"] = {
            "capacity_states": states,
            "capacity_state_consensus": (states[0] if len(set(states)) == 1
                                         else "MIXED"),
            "wip_time_weighted": fmean(r["wip_time_weighted"] for r in sub),
            "queue_max_at_decisions": max(r["queue_max_at_decisions"] for r in sub),
            "a2o_mean_min": fmean(r["a2o_mean_min"] for r in sub
                                  if r["a2o_mean_min"] is not None),
            "yc_busy_frac": fmean(r["yc_busy_frac"] for r in sub),
            "throughput_per_h": fmean(r["throughput_per_h"] for r in sub),
            "backlog_4h": [r["backlog_4h"] for r in sub],
            "backlog_6h": [r["backlog_6h"] for r in sub],
            "drain_to_empty_h": [None if r["drain_to_empty_s"] is None
                                 else round(r["drain_to_empty_s"] / 3600.0, 3)
                                 for r in sub],
            "v2_total": fmean(r["v2_total"] for r in sub)}
    res = {"repro": repro_stamp(
               experiment="YR-149 5부하 셀 데이터 자격시험 (SF-SPT·이송 없음)",
               seeds={"A_master": band["seeds"]["A"], "B": band["seeds"]["B"]},
               profile_id="calibrated",
               prereg="A150 master 중첩추출(150→125→100→75→50)·A/B 동결 template·"
                      "하드 검사 8종·용량상태 기계 분류. 부하 이름은 입력값일 뿐 혼잡 "
                      "판정이 아니다.",
               extra={"band_digest": band["digest"], "cell_sizes": list(CELL_SIZES)}),
           "hard_checks": hard, "per_pair_checks": checks,
           "by_cell": by_cell, "runs": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qualification.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"hard": hard,
                      **{k: {"state": v["capacity_state_consensus"],
                             "wip": round(v["wip_time_weighted"], 2),
                             "busy": round(v["yc_busy_frac"], 3),
                             "a2o_min": round(v["a2o_mean_min"], 1),
                             "tph": round(v["throughput_per_h"], 1),
                             "bl4": v["backlog_4h"], "bl6": v["backlog_6h"]}
                         for k, v in by_cell.items()}}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-band", action="store_true")
    ap.add_argument("--qualify", action="store_true")
    a = ap.parse_args()
    if a.make_band:
        make_band()
    if a.qualify:
        qualify()
    print("DONE")
