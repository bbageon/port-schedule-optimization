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
    # 회귀 가드(EventKind alias 버그): 전부 완료 시 대기열은 비어야 하고
    # queue-area == 완료 대기 합 (정확 적분 정합성)
    assert sim.kpis.waiting_count() == 0
    assert abs(sim.kpis.queue_area_s - sum(sim.kpis.wait_samples_s)) < 1e-6


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
    """drain=0, 종료 직전 동시 도착 2건 → 1건만 처리되고 backlog 1.

    리뷰 확정건 회귀 가드:
    - 적분은 평가 윈도우 end_time 에서 절단 (마지막 작업 길이와 무관)
    - 미서비스 트럭은 (end - 도착) 검열 대기 표본으로 포함
    - 동시각 도착 2건은 첫 의사결정에서 모두 후보로 보여야 함
    """
    containers = {"C1": _c("C1", 24, 6, 1), "C3": _c("C3", 1, 1, 1)}
    jobs = [_gate_out("J01", "C1", 990.0), _gate_out("J02", "C3", 990.0)]
    sc = Scenario(scenario_id="drain0", seed=0, horizon_s=1000.0, drain_window_s=0.0,
                  jobs=jobs, containers=containers)
    sim = YardSimulator(PROFILE, sc)
    dp = sim.run_until_decision()
    assert dp is not None
    assert [j.job_id for j in sim.dispatchable_jobs()] == ["J01", "J02"]  # 동시각 모두 공개
    sim.execute_job("J01")
    _run_greedy(sim)
    assert sim.terminal
    assert sim.unfinished_backlog() == 1
    # J01 은 990 에 즉시 서비스(대기 0), J02 는 미서비스 → 검열 표본 (1000-990)=10
    assert sorted(round(w, 6) for w in sim.kpis.wait_samples_s) == [0.0, 10.0]
    # 적분 절단: queue_area == Σ 대기표본 (검열 포함 정합성)
    assert abs(sim.kpis.queue_area_s - 10.0) < 1e-9
    assert abs(sim.kpis.queue_area_s - sum(sim.kpis.wait_samples_s)) < 1e-9


def test_unfinished_vessel_delay_charged_at_end():
    """미완료 본선작업의 deadline 초과가 종료시점에 계상 — 방치 무벌점 방지."""
    import dataclasses
    narrow = dataclasses.replace(PROFILE, crane=dataclasses.replace(PROFILE.crane,
                                                                    service_bay_max=10))
    containers = {"C4": _c("C4", 20, 1, 1)}  # service range 밖 → 영구 미서비스
    jobs = [Job(job_id="JV", flow=JobFlow.VESSEL_LOAD, release_time=0.0,
                actual_gate_in=None, actual_block_arrival=None,
                target_container="C4", deadline=500.0, priority_class=1)]
    sc = Scenario(scenario_id="vessel-stuck", seed=0, horizon_s=1000.0,
                  drain_window_s=0.0, jobs=jobs, containers=containers)
    sim = YardSimulator(narrow, sc)
    assert sim.run_until_decision() is None
    assert sim.unfinished_backlog() == 1
    assert abs(sim.kpis.vessel_delay_s - 500.0) < 1e-9  # end(1000) - deadline(500)


def test_rehandle_slot_exhaustion_excluded_not_crash():
    """재조작 슬롯 고갈 → 후보 제외 (NO_SAFE_SLOT 크래시 방지)."""
    import dataclasses
    geom = dataclasses.replace(PROFILE.block, bay_count=2, row_count=1, tier_max=2)
    spec = dataclasses.replace(PROFILE.crane, service_bay_min=1, service_bay_max=2)
    tiny = dataclasses.replace(PROFILE, block=geom, crane=spec)
    containers = {
        "C1": _c("C1", 1, 1, 1), "C2": _c("C2", 1, 1, 2),      # C2 = blocker(FT40)
        "D1": _c("D1", 2, 1, 1, size=ContainerSize.FT20),       # 유일한 다른 스택은
        "D2": _c("D2", 2, 1, 2, size=ContainerSize.FT20),       # FT20 만재 → 슬롯 없음
    }
    jobs = [_gate_out("J01", "C1", 100.0)]
    sc = Scenario(scenario_id="full-yard", seed=0, horizon_s=1000.0, drain_window_s=0.0,
                  jobs=jobs, containers=containers)
    sim = YardSimulator(tiny, sc)
    assert sim.run_until_decision() is None   # 크래시 없이 후보 제외 → 종료
    assert sim.unfinished_backlog() == 1


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
