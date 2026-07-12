"""시뮬레이터 불변조건·결정론·backlog 테스트 (05 §1.2).

check_invariants=True 로 실행 — 매 이벤트 후 위반 시 즉시 예외.
"""
import pytest

from yard_rl.domain.enums import ContainerSize, JobFlow, JobStatus, LoadStatus
from yard_rl.domain.models import Container, Job
from yard_rl.domain.scenario import Scenario
from yard_rl.io.profile_loader import load_profile
from yard_rl.sim.engine import YardSimulator

PROFILE = load_profile("configs/terminals/poc_single_crane.yaml")


def _c(cid, bay, row, tier, size=ContainerSize.FT40):
    return Container(container_id=cid, size=size, load_status=LoadStatus.FULL,
                     block="B1", bay=bay, row=row, tier=tier)


def _gate_out(jid, target, arrival):
    return Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0,
               actual_gate_in=max(0.0, arrival - 600), actual_block_arrival=arrival,
               target_container=target)


def _mini_scenario(horizon=7200.0, drain=3600.0):
    containers = {
        "C1": _c("C1", 3, 2, 1),
        "C2": _c("C2", 3, 2, 2),   # C1 위 blocker
        "C3": _c("C3", 10, 4, 1),
        "C4": _c("C4", 20, 1, 1),
    }
    jobs = [
        _gate_out("J01", "C1", 100.0),   # blocker 1개 → 재조작 1회 발생해야 함
        _gate_out("J02", "C3", 200.0),
        Job(job_id="J03", flow=JobFlow.GATE_IN, release_time=0.0,
            actual_gate_in=100.0, actual_block_arrival=700.0,
            inbound_size=ContainerSize.FT20, inbound_load=LoadStatus.FULL),
        Job(job_id="J04", flow=JobFlow.VESSEL_LOAD, release_time=300.0,
            actual_gate_in=None, actual_block_arrival=None,
            target_container="C4", deadline=4000.0, priority_class=1),
    ]
    return Scenario(scenario_id="mini", seed=0, horizon_s=horizon, drain_window_s=drain,
                    jobs=jobs, containers=containers)


def _run_greedy(sim):
    """의사결정마다 job_id 최소 작업 dispatch (결정론 기준 정책)."""
    while True:
        dp = sim.run_until_decision()
        if dp is None:
            return
        sim.execute_job(sim.dispatchable_jobs()[0].job_id)


def test_full_run_completes_all_jobs_with_invariants():
    sim = YardSimulator(PROFILE, _mini_scenario(), check_invariants=True)
    _run_greedy(sim)
    assert sim.terminal
    assert all(j.status == JobStatus.DONE for j in sim.jobs.values())
    assert sim.unfinished_backlog() == 0
    assert sim.kpis.rehandle_count == 1          # C2 가 C1 위 → 정확히 1회
    assert sim.kpis.completed_external == 3
    assert sim.kpis.completed_vessel == 1
    assert sim.kpis.queue_area_s > 0
    assert len(sim.kpis.wait_samples_s) == 3     # 외부트럭만


def test_determinism_same_seed_same_trace():
    s1 = YardSimulator(PROFILE, _mini_scenario())
    s2 = YardSimulator(PROFILE, _mini_scenario())
    _run_greedy(s1)
    _run_greedy(s2)
    assert s1.event_log == s2.event_log
    assert s1.kpis.snapshot() == s2.kpis.snapshot()


def test_double_dispatch_blocked():
    sim = YardSimulator(PROFILE, _mini_scenario())
    dp = sim.run_until_decision()
    assert dp is not None
    first = sim.dispatchable_jobs()[0].job_id
    sim.execute_job(first)
    with pytest.raises(Exception):   # 크레인 이중 할당은 즉시 차단
        sim.execute_job(sim.dispatchable_jobs()[0].job_id)


def test_backlog_counted_when_drain_expires():
    """drain=0, 종료 직전 동시 도착 2건 → 1건만 처리되고 backlog 1."""
    containers = {"C1": _c("C1", 24, 6, 1), "C3": _c("C3", 1, 1, 1)}
    jobs = [_gate_out("J01", "C1", 990.0), _gate_out("J02", "C3", 990.0)]
    sc = Scenario(scenario_id="drain0", seed=0, horizon_s=1000.0, drain_window_s=0.0,
                  jobs=jobs, containers=containers)
    sim = YardSimulator(PROFILE, sc)
    _run_greedy(sim)
    assert sim.terminal
    assert sim.unfinished_backlog() == 1
    # 미처리 차량 대기도 적분에 포함 (종료시각 또는 마지막 완료시각까지)
    assert sim.kpis.queue_area_s >= 10.0


def test_service_range_restricts_dispatch():
    """service range 밖 컨테이너 작업은 dispatch 후보에서 제외."""
    import dataclasses
    narrow_crane = dataclasses.replace(PROFILE.crane, service_bay_max=10)
    narrow = dataclasses.replace(PROFILE, crane=narrow_crane)
    containers = {"C4": _c("C4", 20, 1, 1)}
    jobs = [_gate_out("J01", "C4", 100.0)]
    sc = Scenario(scenario_id="range", seed=0, horizon_s=2000.0, drain_window_s=0.0,
                  jobs=jobs, containers=containers)
    sim = YardSimulator(narrow, sc)
    dp = sim.run_until_decision()
    assert dp is None            # 후보 자체가 없어 종료
    assert sim.unfinished_backlog() == 1
