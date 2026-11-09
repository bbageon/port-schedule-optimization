"""A finite branch cannot accrue costs after its simulated world ends."""
from types import SimpleNamespace

import pytest

from yard_rl.v7.crane.rollout import CraneRollout


class World:
    def __init__(self, end):
        self.blocks = {'Y01':SimpleNamespace(end=end)}

    def run(self, policy, review_fn):
        pass


@pytest.mark.parametrize('decision,horizon,world_end,expected', [
    (100., 500., 1000., 600.), (800., 500., 1000., 1000.),
    (86026.08813690123, 10800., 93600., 93600.)])
def test_all_cost_channels_and_sync_share_actual_world_end(monkeypatch, decision, horizon, world_end, expected):
    calls = []
    bridge = SimpleNamespace(_sync=lambda world,end: calls.append(('sync',end)), review=lambda *a:None)
    ctx = SimpleNamespace(make_market=lambda *a,**k:None, make_bridge=lambda *a,**k:bridge,
        announcer=SimpleNamespace(clone_fresh=lambda:SimpleNamespace(review=lambda *a:None)),
        make_exec_policy=lambda **k:(lambda sim,dp:None),
        vessel_idle=lambda world,end:calls.append(('vessel',end)) or {},
        yc_extra_move_s=lambda world:0., rehandles=lambda world:0)
    def cost(records, *, end_s, **kwargs):
        calls.append(('cost',end_s))
        return SimpleNamespace(total=end_s)
    monkeypatch.setattr('yard_rl.v7.crane.rollout.terminal_cost_krw',cost)
    job = SimpleNamespace(t=decision,mbt=World(world_end),orders={},records={},decided=set(),
                          block='Y01',crane='YC-A',decision=None)
    result = CraneRollout(ctx,horizon_s=horizon).branch(job,force_job=None)
    assert calls == [('sync',expected),('vessel',expected),('cost',expected)]
    assert result['cost_end_s'] == expected
    assert result['requested_end_s'] == decision+horizon
    assert result['effective_horizon_s'] == expected-decision
    assert job.mbt.blocks['Y01'].end == world_end


def test_no_time_remaining_is_rejected_before_generating_labels():
    job = SimpleNamespace(t=1000.,mbt=World(1000.))
    with pytest.raises(ValueError,match='no remaining'):
        CraneRollout(None,horizon_s=500.).branch(job,force_job=None)
