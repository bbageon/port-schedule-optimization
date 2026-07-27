"""YR-099 창중(gate-in 후) review — 도착 실현 정보로 반입 재배정 상금 측정.

■ 사전등록 (결과 미열람 동결 — YR-103 판정의 직접 처방)
- 결정시점 = **각 반입 트럭의 실제 gate-in** (spec 결정시점 트리거 "실제 gate-in").
  창 = [gate-in, block-in) — 아직 블록 미도착("도착 안 함"은 현재 관측 사실)일 때만.
- 결정규칙 (트럭당 1회·이후 lock): 소스/타깃 블록의 **관측 혼잡 격차**
  `C_zero(src) − C_zero(dst) ≥ THRESH(0.10, assumed)` 면 이송 (rollout·미래열람 0 —
  YR-101/103 이 기각한 quote 대신 YR-103 이 예측력을 실증한 혼잡 관측만 사용).
- 이식 물리: 새 도착 = gate_in + **신규 draw**(계약분포 180~420, 전용 스트림
  "mid:{seed}:{소스sim}:{jid}" — 실현 draw 재사용 금지·소스별 독립 = 누출 0) + ROUTE_S(180).
  **이송 추가주행 비용 = 건당 ROUTE_S/3600 numeraire 를 C_mid total 에 계상**
  (검증 F3 정정 — yr103/yr099-G1 선례 정합, pro-이송 편향 제거).
  타깃 시계가 이미 지난 경우 skip (fail-closed, 드리프트 근사 문서화).
- **판정런 = 신규 대역 840k/841k (결과 미열람)** — 구 838k/839k 는 개발런 강등
  (검증 probe·테스트가 seed0 소비 + 정정 전 전체 열람. 검증 F8 위생 조치).
- 환경: yr103 과 동일 블록쌍(A=high-tight·B=mid-loose)·gate_block_contract=True·
  SF 고정. 신규 seed 대역 838000/839000+i, N=8.
- arm: A0(무이송) vs C_mid(창중 규칙) — 같은 lockstep runner(짝지음)·같은 seed.
- **주판정: paired Δ[terminal(C_mid) − terminal(A0)] CI(t df=7) 상한 < 0 →
  "창중(도착 실현 후) 재배정 상금 실증" / 아니면 "이 규칙(혼잡 격차 단독) 한계"**
  (상금 부재 단정 금지 — 규칙군 한계와 미분리, YR-103 비대칭 조항과 동일).
- 부판정(보고): THRESH ∈ {0.05, 0.20} 민감도·이송 수·방향별 분해·완주/berth guard.
- G0: 소유권 유일(이식 후 정확히 한 sim)·보존·큐 이벤트 중화·원장 이관·결정론.
한계(정직): lockstep 드리프트 ~1 결정간격 — 혼잡 스냅샷이 최대 그만큼 낡음.
"""
from __future__ import annotations

import argparse
import dataclasses
import heapq
import json
import random
from pathlib import Path
from statistics import fmean

from ..domain.enums import InformationLevel, JobFlow, JobStatus
from ..integrated import TerminalSimulator
from ..integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference, _apply, _wait_of
from ..integrated.block_congestion import block_congestion
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import (GATE_BLOCK_MAX_S, GATE_BLOCK_MEAN_S, GATE_BLOCK_MIN_S,
                                       GATE_BLOCK_SIGMA_S, calibrated_load_params,
                                       generate_terminal_scenario, trunc_normal)

RC = RewardCalculator.numeraire_v1()
OUT = Path("outputs/reports/yr099_midrun_review")
CELL_A, CELL_B = ("high", 0.5), ("mid", 2.0)
# 판정 대역 840k/841k — 구 838k/839k 는 개발런(검증 probe·테스트가 seed0 소비 + 전체 열람)
# 으로 강등. 검증 정정(route 비용·원장·방향counter·스트림 키) 후 새 대역 결과 미열람 판정.
BASE_A, BASE_B = 840_000, 841_000
N_SEEDS, THRESH, ROUTE_S = 8, 0.10, 180.0
LEVEL = InformationLevel.PRE_ADVICE


def _sim(cell, seed):
    prof = build_calibrated_profile()
    p = dataclasses.replace(calibrated_load_params(cell[0], vessel_deadline_mult=cell[1]),
                            time_contract_v2=True, gate_block_contract=True)
    s = TerminalSimulator(prof, generate_terminal_scenario(prof, seed, p),
                          check_invariants=True)
    s.info_level = LEVEL
    return s


def move_inflight_job(src, dst, jid: str, tag: str) -> str:
    """창중 이식 (원자적) — 소스 제거+큐 중화+원장 해제 → 타깃 삽입+이벤트+원장 등록.

    새 도착 = gate_in + 신규 계약 draw + ROUTE_S (실현 draw 미재사용 — 누출 0).
    실패 시 ValueError — 호출부는 KEEP (소스 원상태 유지: 검증 후 pop 순서)."""
    j = src.jobs.get(jid)
    if (j is None or j.status != JobStatus.PLANNED or not j.is_external_truck
            or j.flow != JobFlow.GATE_IN):                 # 검증 F2: 반입(STORE)만 — 반출 위치고정
        raise ValueError(f"{jid}: 이식 불가 상태")
    new_id = f"{jid}@M"
    if new_id in dst.jobs:
        raise ValueError(f"{new_id}: 타깃 중복")
    rng = random.Random(f"mid:{tag}:{jid}")
    travel = trunc_normal(rng, GATE_BLOCK_MEAN_S, GATE_BLOCK_SIGMA_S / GATE_BLOCK_MEAN_S,
                          lo=GATE_BLOCK_MIN_S, hi=GATE_BLOCK_MAX_S)
    arr = j.actual_gate_in + travel + ROUTE_S
    if arr <= dst.now:                              # lockstep 드리프트 — fail-closed
        raise ValueError(f"{jid}: 타깃 시계 경과")
    src.jobs.pop(jid)
    src.queue._heap = [e for e in src.queue._heap
                       if not (e.kind_name == "BLOCK_ARRIVAL" and e.payload == jid)]
    heapq.heapify(src.queue._heap)
    if src.time_ledger is not None:                        # 검증 F1: _a_sorted 도 이관 (잔여 A 제거)
        src.time_ledger.records.pop(jid, None)
        try:
            src.time_ledger._a_sorted.remove(j.actual_gate_in)
        except ValueError:
            pass
    j.job_id = new_id
    j.actual_block_arrival = arr
    if getattr(j, "estimated_block_arrival", None) is not None:
        j.estimated_block_arrival = j.actual_gate_in + GATE_BLOCK_MEAN_S + ROUTE_S
    dst.jobs[new_id] = j
    from ..integrated.events import EventKind
    dst.queue.push(arr, EventKind.BLOCK_ARRIVAL, new_id)
    if dst.time_ledger is not None:                        # 검증 F1: 정렬 유지 삽입 (seal 이후)
        import bisect
        from ..integrated.time_contract import TruckTimes
        dst.time_ledger.records[new_id] = TruckTimes(gate_in=j.actual_gate_in)
        bisect.insort(dst.time_ledger._a_sorted, j.actual_gate_in)
    return new_id


def run_pair(seed_i: int, *, transfer: bool, thresh: float = THRESH) -> dict:
    sims = {"A": _sim(CELL_A, BASE_A + seed_i), "B": _sim(CELL_B, BASE_B + seed_i)}
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens = {k: CandidateGenerator() for k in sims}
    total = {k: 0.0 for k in sims}
    last_b = {k: 0.0 for k in sims}
    decided: set[tuple[str, str]] = set()
    stats = {"A->B": 0, "B->A": 0, "skipped": 0, "missed": 0}   # 검증 F7·F4: 방향·미검토 창
    dps = {}
    for k, s in sims.items():
        dps[k] = s.run_until_decision()
        s.cost.cut()
        last_b[k] = s.now
    while any(dp is not None for dp in dps.values()):
        active = [k for k, dp in dps.items() if dp is not None]
        k = min(active, key=lambda x: sims[x].now)          # 뒤처진 sim 먼저 (lockstep)
        sim = sims[k]
        if transfer:
            _review_gateins(sims, decided, thresh, seed_i, stats)
        gen_by = {c: gens[k].generate(sim, c, LEVEL) for c in dps[k].crane_ids}
        raw = sim.cost.cut()
        total[k] += RC.cost_for(interval_start_s=last_b[k], interval_end_s=sim.now,
                                raw=raw, risk_max=0.0).total_normalized
        last_b[k] = sim.now
        try:
            _apply(sim, pol.decide(sim, dps[k], gen_by))
        except Exception:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dps[k].crane_ids})
        dps[k] = sim.run_until_decision()
    for k, sim in sims.items():                              # 종결 구간
        raw = sim.cost.cut()
        total[k] += RC.cost_for(interval_start_s=last_b[k], interval_end_s=sim.now,
                                raw=raw, risk_max=0.0).total_normalized
    compl = {k: (sum(1 for j in s.jobs.values() if j.status == JobStatus.DONE)
                 / max(1, len(s.jobs))) for k, s in sims.items()}
    berth = {k: getattr(s.kpis, "berth_overrun_s", 0.0) / 60.0 for k, s in sims.items()}
    n_moved = stats["A->B"] + stats["B->A"]
    route_cost = n_moved * ROUTE_S / 3600.0            # 검증 F3: 이송 추가주행 비용 계상 (선례 정합)
    return {"total": round(total["A"] + total["B"] + route_cost, 3),
            "total_a": round(total["A"], 3), "total_b": round(total["B"], 3),
            "route_cost": round(route_cost, 3),
            "compl_min": round(min(compl.values()), 4),
            "berth_sum": round(berth["A"] + berth["B"], 1),
            "n_moved": n_moved, "n_ab": stats["A->B"], "n_ba": stats["B->A"],
            "n_skipped": stats["skipped"], "n_missed": stats["missed"],
            "n_jobs": sum(len(s.jobs) for s in sims.values())}


def _review_gateins(sims, decided, thresh, seed_i, stats):
    """gate-in 창 진입 트럭들 결정 (트럭당 1회).

    관측 한계(검증 F4, 정직): 혼잡은 각 sim 의 자기 시계 스냅샷 — 결정시점(sync_now)
    대비 선행 sim 쪽은 최대 1 결정간격 **앞선** 관측, 뒤처진 쪽은 그만큼 낡음.
    결정 지연 중 이미 도착해버린 창은 missed 로 집계(검토 없이 닫힘)."""
    sync_now = min(s.now for s in sims.values())
    cong = {k: block_congestion(s)["C_zero"] for k, s in sims.items()}
    for k, sim in sims.items():
        other = "B" if k == "A" else "A"
        for jid, j in list(sim.jobs.items()):
            if (k, jid) in decided or not j.is_external_truck or j.flow != JobFlow.GATE_IN:
                continue
            gi = getattr(j, "actual_gate_in", None)
            if gi is None or gi > sync_now:
                continue
            if j.status != JobStatus.PLANNED:              # 검토 전에 도착 = 창 놓침
                decided.add((k, jid))
                stats["missed"] += 1
                continue
            decided.add((k, jid))
            if cong[k] - cong[other] >= thresh:
                try:
                    new_id = move_inflight_job(sim, sims[other], jid,
                                               f"s{seed_i}:{k}")     # 검증 F6: 소스 식별자
                    decided.add((other, new_id))
                    stats[f"{k}->{other}"] += 1
                except ValueError:
                    stats["skipped"] += 1


def run(thresh: float = THRESH) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    from .yr103_info_sufficiency import _ci_t
    rows = []
    for i in range(N_SEEDS):
        a0 = run_pair(i, transfer=False)
        c = run_pair(i, transfer=True, thresh=thresh)
        assert a0["n_jobs"] == c["n_jobs"]                   # G0 보존
        d = round(c["total"] - a0["total"], 3)
        rows.append({"seed": i, "a0": a0, "c": c, "d_total": d})
        print(f"[seed {i}] A0={a0['total']:.2f} C={c['total']:.2f} d={d:+.2f} "
              f"moved={c['n_moved']}(A>B {c['n_ab']}/B>A {c['n_ba']}) "
              f"missed={c['n_missed']} compl={c['compl_min']}", flush=True)
    dts = [r["d_total"] for r in rows]
    # YR-106: 총비용은 본선분산이 지배 — 채널 분해 판정을 함께 낸다(설계감사 critical)
    from ..integrated.evalkit import paired
    yr106 = {}
    if all("chan" in r["a0"] and "chan" in r["c"] for r in rows):
        delta = {"truck": 3.0, "vessel": 10.0, "move": 1.0, "other": 1.0}
        for ch, d in delta.items():
            yr106[ch] = paired([r["c"]["chan"].get(ch, 0.0) - r["a0"]["chan"].get(ch, 0.0)
                                for r in rows], delta_interest=d).as_dict()
    res = {"rows": rows, "thresh": thresh, "d_total_ci": _ci_t(dts),
           "yr106_channels": yr106,
           "moved_mean": round(fmean(r["c"]["n_moved"] for r in rows), 1),
           "compl_min": min(min(r["a0"]["compl_min"], r["c"]["compl_min"]) for r in rows),
           "prereg": "상한<0 → 창중 재배정 상금 실증 / 아니면 '혼잡격차 규칙 한계'"
                     "(상금 부재 단정 금지)"}
    name = f"results_t{int(thresh*100):02d}.json"
    (OUT / name).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nMIDRUN d_total CI={res['d_total_ci']} moved={res['moved_mean']} "
          f"compl_min={res['compl_min']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=THRESH)
    a = ap.parse_args()
    run(a.thresh)
    print("DONE")
