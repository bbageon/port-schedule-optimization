"""Exp-2/3 기능 테스트 — 정보수준별 예측·포지셔닝·선재조작·누출 방지."""
from yard_rl.domain.enums import (ContainerSize, ControlScope, InformationLevel,
                                  JobFlow, JobStatus, LoadStatus, PriorityRule)
from yard_rl.domain.models import Container, Job
from yard_rl.domain.scenario import Scenario
from yard_rl.envs.info_filter import predicted_arrival
from yard_rl.envs.yard_env import YardEnv
from yard_rl.io.profile_loader import load_profile

P = load_profile("configs/terminals/poc_single_crane.yaml")


def _c(cid, bay, row, tier, size=ContainerSize.FT40):
    return Container(container_id=cid, size=size, load_status=LoadStatus.FULL,
                     block="B1", bay=bay, row=row, tier=tier)


def _gate_out(jid, target, arrival, eta=None):
    return Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0,
               actual_gate_in=max(0.0, arrival - 600.0), actual_block_arrival=arrival,
               provided_eta=eta, target_container=target)


def test_predicted_arrival_by_level():
    j = _gate_out("J", "C", arrival=1000.0, eta=1100.0)
    assert predicted_arrival(j, InformationLevel.BLOCK_ARRIVAL, 600.0) is None
    # Exp-2: 게이트 진입 + 자체추정 (실제도착·제공ETA 미사용 — 누출 방지)
    assert predicted_arrival(j, InformationLevel.GATE_IN, 600.0) == 400.0 + 600.0
    assert predicted_arrival(j, InformationLevel.PRE_ADVICE, 600.0) == 1100.0


def test_exp2_does_not_use_provided_eta():
    """Exp-2 예측자는 제공 ETA(Exp-3 정보)와 무관해야 한다."""
    j1 = _gate_out("J1", "C", arrival=1000.0, eta=100.0)     # 극단적으로 다른 ETA
    j2 = _gate_out("J2", "C", arrival=1000.0, eta=2000.0)
    est1 = predicted_arrival(j1, InformationLevel.GATE_IN, 600.0)
    est2 = predicted_arrival(j2, InformationLevel.GATE_IN, 600.0)
    assert est1 == est2  # gate_in 동일 → 추정 동일 (ETA 영향 0)


def test_positioning_moves_idle_crane_toward_future_job():
    containers = {"C1": _c("C1", 20, 3, 1)}
    jobs = [_gate_out("J01", "C1", arrival=1000.0, eta=1000.0)]
    sc = Scenario(scenario_id="pos", seed=0, horizon_s=4000.0, drain_window_s=0.0,
                  jobs=jobs, containers=containers)
    env = YardEnv(P, info_level=InformationLevel.PRE_ADVICE,
                  control_scope=ControlScope.PLUS_POSITIONING, check_invariants=True)
    state, info = env.reset(sc)
    assert state is not None
    assert info.action_mask[PriorityRule.EARLIEST_PROVIDED_ARRIVAL]  # 미래작업 포지셔닝
    assert env.sim.now < 1000.0                     # 도착 전 idle 의사결정
    s2, r, done, info2 = env.step(int(PriorityRule.EARLIEST_PROVIDED_ARRIVAL))
    assert env.sim.kpis.positioning_count == 1
    assert env.sim.crane.position_bay == 20.0       # 목표 bay 로 이동
    assert env.sim.jobs["J01"].status in (JobStatus.PLANNED, JobStatus.WAITING)
    # 이후 도착한 작업을 정상 서비스하고 종료 가능해야 함
    while s2 is not None:
        a = next(i for i, m in enumerate(info2.action_mask) if m)
        s2, _r, _d, info2 = env.step(a)
    assert env.sim.jobs["J01"].status == JobStatus.DONE


def test_pre_rehandle_clears_blockers_before_arrival():
    containers = {"C1": _c("C1", 5, 2, 1), "C2": _c("C2", 5, 2, 2)}  # C2 가 blocker
    jobs = [_gate_out("J01", "C1", arrival=2000.0, eta=2000.0)]
    sc = Scenario(scenario_id="prereh", seed=0, horizon_s=5000.0, drain_window_s=0.0,
                  jobs=jobs, containers=containers)
    env = YardEnv(P, info_level=InformationLevel.PRE_ADVICE,
                  control_scope=ControlScope.PLUS_PRE_REHANDLE, check_invariants=True)
    state, info = env.reset(sc)
    assert info.action_mask[PriorityRule.PRE_REHANDLE]
    t_before = env.sim.now
    assert t_before < 2000.0                          # 도착 전 idle 의사결정
    env.step(int(PriorityRule.PRE_REHANDLE))          # step 은 다음 결정(도착 후)까지 진행
    assert env.sim.kpis.pre_rehandle_count == 1
    assert env.sim.kpis.rehandle_count == 1           # 비용 동일 계상
    assert env.sim.stacks.blockers_above("C1") == []  # blocker 제거됨
    pre = [e for e in env.sim.event_log if e[1] == "PRE_REHANDLE"]
    assert len(pre) == 1 and pre[0][0] < 2000.0       # 실행 자체는 도착 전


def test_pre_rehandle_masked_in_sequence_only():
    containers = {"C1": _c("C1", 5, 2, 1), "C2": _c("C2", 5, 2, 2)}
    jobs = [_gate_out("J01", "C1", arrival=2000.0, eta=2000.0),
            _gate_out("J02", "C1x", arrival=10.0)]
    containers["C1x"] = _c("C1x", 2, 1, 1)
    sc = Scenario(scenario_id="seqonly", seed=0, horizon_s=5000.0, drain_window_s=0.0,
                  jobs=jobs, containers=containers)
    env = YardEnv(P, info_level=InformationLevel.PRE_ADVICE,
                  control_scope=ControlScope.SEQUENCE_ONLY)
    _state, info = env.reset(sc)
    assert not info.action_mask[PriorityRule.PRE_REHANDLE]   # scope 밖
    # 3A: EPA 는 현재 후보 순서화로만 동작 (mask 는 후보 도착 후 열림)
    assert env.sim.now >= 10.0 or not info.action_mask[PriorityRule.EARLIEST_PROVIDED_ARRIVAL]
