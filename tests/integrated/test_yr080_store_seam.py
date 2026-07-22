"""YR-080 단계 6 — 신규 적재(STORE) seam 계약 (기본 off·골든 불변).

seam: sim.store_slot_selector — None(기본)이면 find_slot greedy 그대로.
deployable_store_selector: 매장(burial) 회피 — 반출 예정 컨테이너 위에 쌓지 않기.
"""
from __future__ import annotations

import pytest

from yard_rl.contract.schema import CandidateKind
from yard_rl.integrated import (CraneAssignment, ReferenceDispatcher, TerminalSimulator,
                                build_integrated_profile,
                                build_minimal_terminal_scenario)
from yard_rl.integrated.rehandle_oracle import deployable_store_selector


def _drive(sim):
    disp = ReferenceDispatcher()
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            break
        for cid in dp.crane_ids:
            cands = sim.candidates_for(cid)
            sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE,
                                            job_ref=disp.select(sim, cid, cands))
                       if cands else CraneAssignment(cid, CandidateKind.WAIT))
        sim.close_decision()
    return sim


def _fresh(selector=None):
    sim = TerminalSimulator(build_integrated_profile(), build_minimal_terminal_scenario())
    if selector is not None:
        sim.store_slot_selector = selector
    return sim


def test_seam_off_matches_golden_hash():
    """기본(off) = find_slot greedy 그대로 — 터미널 골든과 동일 (seam 무해성)."""
    sim = _drive(_fresh())
    assert sim.event_stream_hash() == "63556f0e932dcdfd"


def test_deployable_selector_completes_and_deterministic():
    s1 = _drive(_fresh(deployable_store_selector))
    s2 = _drive(_fresh(deployable_store_selector))
    assert s1.event_stream_hash() == s2.event_stream_hash()
    assert all(v.done for v in s1.vessels.values())
    assert s1.kpis.completed_external == 3 and s1.kpis.completed_vessel == 4


def test_deployable_selector_avoids_burying_pending_outbound():
    """반출 예정(target 존재·미완) 컨테이너 위에 쌓지 않는다 — 비방해 계약."""
    sim = _fresh()
    spec = sim.fleet.spec("YC-A")
    job = sim.jobs["J-VES-D0"]           # STORE (inbound FT40)
    pending = {j.target_container for j in sim.jobs.values()
               if j.target_container is not None and j.status.name != "DONE"}
    # 반출 대상 C-B1(bay 35) 바로 옆에서 요청 — greedy 는 (35,1) 을 고를 유인
    dest = deployable_store_selector(sim, sim.stacks, job, spec, 35.0, 1.0, frozenset())
    assert dest is not None
    pile = sim.stacks.stack(*dest)
    assert not (pile and pile[-1] in pending), (
        f"{dest} 는 반출 예정 {pile[-1] if pile else None} 을 매장")


def test_bad_selector_postcondition_raises():
    """후조건 위반 selector 는 조용히 진행되지 않고 즉시 발화."""
    def bad(sim, stk, job, spec, bay, row, exclude):
        return (5, 1)     # C-A1/C-A2 pile — tier 여유는 있으나 exclude/범위와 무관하게 검사됨
    sim = _fresh(lambda s, st, j, sp, b, r, e: (0, 1))   # 서비스 범위 밖 bay 0
    with pytest.raises(RuntimeError):
        sim._store_slot(sim.jobs["J-VES-D0"], sim.fleet.spec("YC-A"), 5.0, 1.0)
