"""중앙 joint resolver — 0-위반·결정론·하위호환·mandatory 우선 (YR-037)."""
from dataclasses import replace

from yard_rl.contract import CandidateKind
from yard_rl.contract.validate import validate_joint
from yard_rl.domain.enums import ContainerSize, InformationLevel, JobFlow, JobStatus, LoadStatus
from yard_rl.domain.models import BlockGeometry, Container, CraneSpec, Job
from yard_rl.contract.state import LaneGraph
from yard_rl.integrated import (BaselinePreference, CandidateGenerator, CentralResolver,
                               ReferenceDispatcher, TerminalSimulator, record_episode,
                               resolution_stream_hash)
from yard_rl.integrated.profile import IntegratedProfile, TransferFleetSpec
from yard_rl.integrated.scenario import TerminalScenario
from yard_rl.integrated.fixtures import build_integrated_profile, build_minimal_terminal_scenario

PROF = build_integrated_profile()
GEN = CandidateGenerator()
LEVEL = InformationLevel.PRE_ADVICE


def _drive_resolver(sim, preference=None):
    """resolver 로 완주하며 결정마다 dry_run==commit·validate_joint 검증. 예외=위반."""
    r = CentralResolver(preference or BaselinePreference())
    sim.info_level = LEVEL
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            return sim
        gen_by = {c: GEN.generate(sim, c, LEVEL) for c in dp.crane_ids}
        resn = r.resolve(sim, dp, gen_by)
        chosen = {cr.crane_id: gen_by[cr.crane_id].items[cr.chosen_candidate_id].job_ref
                  for cr in resn.resolutions if cr.action != CandidateKind.WAIT}
        proj = sim.dry_run_commit(chosen)
        r.apply(sim, resn, gen_by)              # 위반이면 assign→reserve() 가 예외
        for cid, plan in proj.plans.items():    # dry_run == commit
            ap = sim._active_plans[cid]
            assert ap.slots == plan.slots and abs(ap.duration_s - plan.duration_s) < 1e-9


def test_full_run_no_violation():
    sim = _drive_resolver(TerminalSimulator(PROF, build_minimal_terminal_scenario()))
    assert sim.terminal and sim.unfinished_backlog() == 0
    assert sim.reservations.orphan_count() == 0


def test_every_record_joint_valid():
    sim = TerminalSimulator(PROF, build_minimal_terminal_scenario())
    recs = record_episode(sim, ReferenceDispatcher(), info_level=LEVEL, episode_id="E")
    for r in recs:
        validate_joint(r)   # DUP_CRANE·DUP_JOB·LANE_CONFLICT·INFEASIBLE_SELECTION·INELIGIBLE 0


def test_resolver_determinism():
    def h():
        s = TerminalSimulator(PROF, build_minimal_terminal_scenario())
        record_episode(s, ReferenceDispatcher(), info_level=LEVEL, episode_id="E")
        return resolution_stream_hash(s.resolution_log), s.event_stream_hash()
    assert h() == h()


# ------------------------------------------------------------ contention
def _spec(cid, lo=1, hi=40):
    return CraneSpec(crane_id=cid, service_bay_min=lo, service_bay_max=hi,
                     gantry_speed_mps=2.0, trolley_speed_mps=1.0, hoist_speed_loaded_mps=0.5,
                     hoist_speed_empty_mps=0.9, lock_time_s=30.0, unlock_time_s=20.0,
                     truck_positioning_time_s=25.0)


def _profile(cranes):
    return IntegratedProfile(
        terminal_id="T", profile_date="2026-07-15", assumed=True,
        block=BlockGeometry("B1", 40, 4, 4, 6.5, 2.9, 2.6, 0), cranes=cranes,
        lane_graph=LaneGraph(("L1", "L2"), (("L1", "L2"),)),
        transfer=TransferFleetSpec("TF", "YT", 2, 180.0),
        long_wait_sla_s=1800.0, decision_horizon_s=1800.0, safety_gap_bay=2.0)


def _one_job_scenario():
    """단일 GATE_OUT (bay 20) — 두 크레인이 모두 도달 가능 → token 경합."""
    containers = {"C1": Container("C1", ContainerSize.FT40, LoadStatus.FULL, "B1", 20, 1, 1)}
    jobs = [Job(job_id="J1", flow=JobFlow.GATE_OUT, release_time=0.0, actual_gate_in=0.0,
                actual_block_arrival=0.0, target_container="C1")]
    return TerminalScenario("contend", 0, 3600.0, 1800.0, containers, jobs, [], [])


def test_shared_job_no_dup():
    """두 크레인이 같은 job 경합 → 하나만 서비스, 다른 하나 WAIT (DUP_JOB 0)."""
    sim = TerminalSimulator(_profile((_spec("YC-A"), _spec("YC-B"))), _one_job_scenario())
    sim.info_level = LEVEL
    dp = sim.run_until_decision()
    assert set(dp.crane_ids) == {"YC-A", "YC-B"}      # 둘 다 J1 후보 → 결정 크레인
    gen_by = {c: GEN.generate(sim, c, LEVEL) for c in dp.crane_ids}
    r = CentralResolver(BaselinePreference())
    resn = r.resolve(sim, dp, gen_by)
    serving = [cr for cr in resn.resolutions if cr.action == CandidateKind.SERVE]
    waiting = [cr for cr in resn.resolutions if cr.action == CandidateKind.WAIT]
    assert len(serving) == 1 and len(waiting) == 1
    assert serving[0].crane_id == "YC-A"              # preference tie → crane_id 오름차순
    assert resn.contested and resn.contested[0][0] == "J1"   # token J1 을 두 크레인이 경합
    assert set(resn.contested[0][1]) == {"YC-A", "YC-B"}
    r.apply(sim, resn, gen_by)                         # 예외 없이 커밋 (DUP_JOB backstop)
    _drive_resolver(sim)                               # 완주
    assert sim.terminal and sim.unfinished_backlog() == 0


def test_mandatory_priority():
    """SLA 임박 트럭이 non-mandatory 본선/일반 후보보다 우선 서비스."""
    # 두 job: 오래 대기 외부(mandatory) bay 5, 본선연계 bay 8 — 한 크레인만 존재
    containers = {"C1": Container("C1", ContainerSize.FT40, LoadStatus.FULL, "B1", 5, 1, 1),
                  "C2": Container("C2", ContainerSize.FT40, LoadStatus.FULL, "B1", 8, 1, 1)}
    jobs = [
        Job("J-MAND", JobFlow.GATE_OUT, 0.0, actual_gate_in=0.0, actual_block_arrival=0.0,
            target_container="C1"),
        Job("J-VES", JobFlow.VESSEL_LOAD, 0.0, actual_gate_in=None, actual_block_arrival=None,
            target_container="C2", deadline=3000.0, priority_class=1),
    ]
    sc = TerminalScenario("mand", 0, 7200.0, 1800.0, containers, jobs, [], [])
    sim = TerminalSimulator(_profile((_spec("YC-A"),)), sc)
    sim.info_level = LEVEL
    dp = sim.run_until_decision()
    # J-MAND 을 오래 대기로 (mandatory)
    sim.jobs["J-MAND"].actual_block_arrival = sim.now - 2000.0
    gen_by = {c: GEN.generate(sim, c, LEVEL) for c in dp.crane_ids}
    mand = [g for g in gen_by["YC-A"].items if g.mandatory]
    assert mand and mand[0].job_ref.job_id == "J-MAND"
    resn = CentralResolver(BaselinePreference()).resolve(sim, dp, gen_by)
    chosen = resn.resolutions[0]
    assert chosen.action == CandidateKind.SERVE
    assert chosen.chosen_token == "J-MAND"    # 본선(J-VES)보다 mandatory 우선


def test_backcompat_single_crane_matches_dispatcher():
    """단일 크레인: resolver(baseline) 완주 == ReferenceDispatcher 완주 (event hash)."""
    prof1 = _profile((_spec("YC-A"),))
    s_res = TerminalSimulator(prof1, build_minimal_terminal_scenario())
    _drive_resolver(s_res)
    s_disp = TerminalSimulator(prof1, build_minimal_terminal_scenario())
    ReferenceDispatcher().run(s_disp)
    assert s_res.event_stream_hash() == s_disp.event_stream_hash()
