from pathlib import Path
from types import SimpleNamespace
import copy

import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts/v7'))
    import audit_admission_ledger as module
    return module


@pytest.mark.parametrize('times,end,expected',[
    ([None,None,None,None,None],86400,'NOT_ENTERED'),
    ([86000,86100,None,None,None],86400,'WAITING'),
    ([86000,86100,86300,86500,86800],86400,'SERVICE'),
    ([86000,86100,86300,86500,86800],86600,'EXIT_TRAVEL'),
    ([86000,86100,86300,86500,86800],86800,'EXITED'),
    ([86300,86500,None,None,None],86400,'INBOUND_TRAVEL')])
def test_observation_end_controls_stage(audit,times,end,expected):
    assert audit.lifecycle(dict(zip(audit.FIELDS,times)),end)==expected


def test_26h_projection_preserves_official_24h_records_and_future_exit(audit):
    from yard_rl.v7.schema.record import ExecutionRecord
    from yard_rl.v7.world.integrated.time_contract import TruckTimes
    r=ExecutionRecord(doc_key='D-1',copino_notice_s=0.)
    tt=TruckTimes(gate_in=86000,block_arrival=86100,service_start=87000,job_done=93500,gate_out=94000)
    sim=SimpleNamespace(time_ledger=SimpleNamespace(records={'D-1':tt}),jobs={})
    mbt=SimpleNamespace(blocks={'Y01':sim})
    original={'D-1':r};before=copy.deepcopy(original)
    r24=audit.project_records(mbt,original,86400)['D-1']
    assert original==before
    assert r24.block_in_s==86100 and r24.service_start_s is None
    r26=audit.project_records(mbt,{'D-1':r24},93600)['D-1']
    assert r24.job_done_s is None and r26.job_done_s==93500
    assert r26.gate_out_s is None  # exit is scheduled, but after the observation end


def test_capacity_breakdown_distinguishes_physical_and_reserved_space(audit):
    from yard_rl.v7.world.domain.enums import JobStatus,JobFlow
    jobs={'a':SimpleNamespace(status=JobStatus.PLANNED,flow=JobFlow.GATE_IN),
          'b':SimpleNamespace(status=JobStatus.PLANNED,flow=JobFlow.VESSEL_DISCHARGE)}
    sim=SimpleNamespace(profile=SimpleNamespace(block=SimpleNamespace(bay_count=2,row_count=2,tier_max=2)),
                        stacks=SimpleNamespace(containers=list(range(4))),jobs=jobs)
    mbt=SimpleNamespace(blocks={'Y01':sim},_reserved_inbound={'Y01':1},capacity_margin=2,
                        free_slots=lambda bid:1)
    c=audit.capacity_state(mbt,'Y01')
    assert c['physical_free']==4 and c['free']==1 and c['free_without_discharge']==2


def test_rejected_order_is_not_assigned_zero_turntime(audit):
    row=dict(zip(audit.FIELDS,[None]*5))
    assert audit.censored_seconds(row,86400) is None
    row.update(gate_in_s=86000,gate_out_s=87000)
    assert audit.censored_seconds(row,86400)==400
