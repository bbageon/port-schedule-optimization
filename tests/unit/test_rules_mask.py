"""priority rule 선택·mask 단위 테스트 (05 §1.1)."""
import pytest

from yard_rl.domain.enums import (ContainerSize, ControlScope, InformationLevel,
                                  JobFlow, JobStatus, LoadStatus, PriorityRule)
from yard_rl.domain.models import Container, CraneState, Job
from yard_rl.envs.action_mask import build_mask
from yard_rl.envs.rules import PriorityRuleExecutor
from yard_rl.io.profile_loader import load_profile
from yard_rl.sim.stack import YardStacks

P = load_profile("configs/terminals/poc_single_crane.yaml")


def _c(cid, bay, row, tier):
    return Container(container_id=cid, size=ContainerSize.FT40,
                     load_status=LoadStatus.FULL, block="B1", bay=bay, row=row, tier=tier)


def _stacks(*cs):
    return YardStacks(P.block, {c.container_id: c for c in cs})


def _crane(bay=1.0):
    return CraneState(crane_id="YC-01", position_bay=bay, trolley_row=0.0,
                      service_bay_min=1, service_bay_max=24)


def _out(jid, target, gate_in, arrival, status=JobStatus.WAITING):
    j = Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0,
            actual_gate_in=gate_in, actual_block_arrival=arrival, target_container=target)
    j.status = status
    return j


def test_fifo_vs_longest_wait_differ_by_gate_vs_block_order():
    """A 가 게이트는 먼저, 블록 도착은 늦게 → FIFO=A, LONGEST_WAIT=B."""
    stacks = _stacks(_c("CA", 2, 1, 1), _c("CB", 8, 1, 1))
    a = _out("JA", "CA", gate_in=0.0, arrival=900.0)
    b = _out("JB", "CB", gate_in=100.0, arrival=800.0)
    ex = PriorityRuleExecutor(P)
    kw = dict(crane=_crane(), stacks=stacks, now=1000.0)
    assert ex.select(PriorityRule.FIFO, [a, b], **kw).job_id == "JA"
    assert ex.select(PriorityRule.LONGEST_WAIT, [a, b], **kw).job_id == "JB"


def test_nearest_and_min_rehandle():
    stacks = _stacks(_c("CN", 2, 1, 1), _c("CF", 20, 1, 1), _c("CF2", 20, 1, 2))
    near = _out("JN", "CN", 0.0, 100.0)     # blocker 0, bay 2
    far = _out("JF", "CF", 0.0, 100.0)      # blocker 1(CF2), bay 20
    ex = PriorityRuleExecutor(P)
    kw = dict(crane=_crane(bay=1.0), stacks=stacks, now=200.0)
    assert ex.select(PriorityRule.NEAREST_JOB, [near, far], **kw).job_id == "JN"
    assert ex.select(PriorityRule.MIN_REHANDLE, [far, near], **kw).job_id == "JN"


def test_sla_exceeded_wins_tiebreak():
    """같은 primary 값이면 SLA 초과 작업이 우선 (02 §6 동점 1순위)."""
    stacks = _stacks(_c("CA", 5, 1, 1), _c("CB", 5, 2, 1))
    a = _out("JA", "CA", 0.0, 100.0)      # 대기 2000s > SLA 1800
    b = _out("JB", "CB", 0.0, 1900.0)     # 대기 200s
    ex = PriorityRuleExecutor(P)
    # NEAREST: 두 작업 bay 동일(5) → trolley 차이만. CA(row1) 가까움 + SLA 초과.
    sel = ex.select(PriorityRule.NEAREST_JOB, [b, a], crane=_crane(5.0),
                    stacks=stacks, now=2100.0)
    assert sel.job_id == "JA"


def test_mask_conditions():
    stacks = _stacks(_c("CA", 3, 1, 1))
    out = _out("JA", "CA", 0.0, 100.0)
    gate_in = Job(job_id="JI", flow=JobFlow.GATE_IN, release_time=0.0,
                  actual_gate_in=0.0, actual_block_arrival=50.0,
                  inbound_size=ContainerSize.FT20, inbound_load=LoadStatus.FULL)
    gate_in.status = JobStatus.WAITING
    kw = dict(level=InformationLevel.BLOCK_ARRIVAL, scope=ControlScope.SEQUENCE_ONLY,
              crane=_crane(3.0), stacks=stacks, profile=P)
    m = build_mask([out, gate_in], **kw)
    assert m[PriorityRule.FIFO] and m[PriorityRule.LONGEST_WAIT] and m[PriorityRule.NEAREST_JOB]
    assert m[PriorityRule.MIN_REHANDLE]          # target 있는 후보 존재
    assert not m[PriorityRule.VESSEL_PRIORITY]   # 본선 후보 없음
    assert m[PriorityRule.SAME_BAY_BATCH]        # crane bay=3 == CA bay
    assert not m[PriorityRule.EARLIEST_PROVIDED_ARRIVAL]  # Exp-1 에선 항상 mask
    assert not m[PriorityRule.PRE_REHANDLE] and not m[PriorityRule.WAIT_YIELD]
    # GATE_IN 만 있으면 MIN_REHANDLE mask
    m2 = build_mask([gate_in], **kw)
    assert not m2[PriorityRule.MIN_REHANDLE]
    assert build_mask([], **kw) == [False] * 9


def test_executor_deterministic_tiebreak_by_job_id():
    stacks = _stacks(_c("CA", 5, 1, 1), _c("CB", 5, 1, 2))
    # CB 가 CA 위 → CA 는 blocker 1. 같은 도착시각.
    a = _out("JA", "CA", 0.0, 100.0)
    b = _out("JB", "CB", 0.0, 100.0)
    ex = PriorityRuleExecutor(P)
    kw = dict(crane=_crane(5.0), stacks=stacks, now=200.0)
    s1 = ex.select(PriorityRule.LONGEST_WAIT, [a, b], **kw)
    s2 = ex.select(PriorityRule.LONGEST_WAIT, [b, a], **kw)
    assert s1.job_id == s2.job_id  # 입력 순서 무관 결정론
    # 동점 체인: SLA 동일·slack 동일·대기 동일 → blockers 적은 CB
    assert s1.job_id == "JB"


def test_vessel_priority_picks_tightest_deadline():
    stacks = _stacks(_c("CV1", 4, 1, 1), _c("CV2", 6, 1, 1))
    v1 = Job(job_id="JV1", flow=JobFlow.VESSEL_LOAD, release_time=0.0, actual_gate_in=None,
             actual_block_arrival=None, target_container="CV1", deadline=5000.0)
    v2 = Job(job_id="JV2", flow=JobFlow.VESSEL_LOAD, release_time=0.0, actual_gate_in=None,
             actual_block_arrival=None, target_container="CV2", deadline=3000.0)
    v1.status = v2.status = JobStatus.RELEASED
    ex = PriorityRuleExecutor(P)
    sel = ex.select(PriorityRule.VESSEL_PRIORITY, [v1, v2],
                    crane=_crane(), stacks=stacks, now=1000.0)
    assert sel.job_id == "JV2"
