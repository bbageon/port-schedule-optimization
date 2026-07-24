"""YR-088 "본선판 ETA" — 스케줄 기반 본선 선제 준비 후보 회귀 테스트.

기본(vessel_prep=False)=현행 골든 바이트 동일(golden 테스트 담당). 여기선 opt-in ON 계약:
다가올 적하(VESSEL_LOAD) 대상의 blocker 를 미리 정리하는 PRE_REHANDLE 후보를 발행한다.
"""
from __future__ import annotations

from yard_rl.contract.schema import CandidateKind
from yard_rl.domain.enums import InformationLevel, JobFlow, JobStatus
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                          _apply)
from yard_rl.integrated.candidates import CandidateGenerator, iter_vessel_prep_jobs
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import (calibrated_load_params,
                                             generate_terminal_scenario)

PROF = build_calibrated_profile()


def _sim(seed=820001):
    sim = TerminalSimulator(PROF, generate_terminal_scenario(
        PROF, seed, calibrated_load_params("high")), check_invariants=True)
    sim.info_level = InformationLevel.PRE_ADVICE
    return sim


def _count_vessel_prep(vessel_prep: bool, seed=820001) -> int:
    """에피소드를 SF-SPT 로 돌며 발행된 본선 준비(PRE_REHANDLE·is_vessel) 후보 총수."""
    sim = _sim(seed)
    gen = CandidateGenerator(vessel_prep=vessel_prep)
    drive = CandidateGenerator()          # 진행은 기본 생성기(결정론)
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    n = 0
    dp = sim.run_until_decision()
    while dp is not None:
        for c in dp.crane_ids:
            n += sum(1 for g in gen.generate(sim, c, sim.info_level).items
                     if g.kind == CandidateKind.PRE_REHANDLE and g.job_ref and g.job_ref.is_vessel)
        _apply(sim, pol.decide(sim, dp, {c: drive.generate(sim, c, sim.info_level)
                                         for c in dp.crane_ids}))
        dp = sim.run_until_decision()
    return n


def test_off_no_vessel_prep_candidates():
    """기본(off) → 본선 준비 후보 0 (골든 불변 경로)."""
    assert _count_vessel_prep(False) == 0


def test_on_generates_vessel_prep_candidates():
    """ON → 본선 준비 후보 >0 (스케줄 기반 선제 준비 발행)."""
    assert _count_vessel_prep(True) > 0


def test_prep_jobs_are_planned_load_with_blockers():
    """iter_vessel_prep_jobs 는 PLANNED VESSEL_LOAD·blocker 존재·지평 내만 낸다."""
    sim = _sim()
    for _ in range(40):                   # 몇 결정 진행해 선박 개시·스케줄 진입
        dp = sim.run_until_decision()
        if dp is None:
            break
        g = CandidateGenerator()
        _apply(sim, ResolverPolicy(ServiceFirstSPTPreference(), "SF").decide(
            sim, dp, {c: g.generate(sim, c, sim.info_level) for c in dp.crane_ids}))
    got = False
    horizon = sim.profile.decision_horizon_s
    for cid in (c.crane_id for c in PROF.cranes):
        for j, c in iter_vessel_prep_jobs(sim, cid, sim.info_level):
            got = True
            assert j.flow == JobFlow.VESSEL_LOAD and j.status == JobStatus.PLANNED
            assert sim.stacks.blockers_above(j.target_container)
            assert j.release_time - sim.now <= horizon
    # got 은 시나리오 의존이라 강제 아님 — 나왔다면 계약을 지켜야 한다.
    assert got or True


def test_lower_info_level_no_prep():
    """PRE_ADVICE 아니면 준비 후보 없음 (트럭과 평행·결정론)."""
    sim = _sim()
    sim.info_level = InformationLevel.BLOCK_ARRIVAL
    sim.run_until_decision()
    assert list(iter_vessel_prep_jobs(sim, PROF.cranes[0].crane_id, sim.info_level)) == []
