"""Observed arrival order is distinct from notice order and future truth."""
import copy
from types import SimpleNamespace as NS

import pytest

from yard_rl.v7.dispatch import ArrivalFIFO, FIFO, make_preference
from yard_rl.v7.world.contract.schema import CandidateKind


class World:
    def __init__(self, events=()):
        self.clock = 100.
        self.event_log = list(events)
        self.jobs = {
            'A': NS(release_time=0., provided_eta=0., actual_block_arrival=80.),
            'B': NS(release_time=40., provided_eta=40., actual_block_arrival=60.),
        }


def candidate(jid, kind=CandidateKind.SERVE):
    return NS(kind=kind, job_ref=NS(job_id=jid))


def first(pref, sim, candidates):
    return min(candidates, key=lambda g: pref.rank(sim, 'YC', g)).job_ref.job_id


def test_arrival_order_reverses_notice_order():
    sim = World([(60., 'BLOCK_ARRIVAL', 'B'), (80., 'BLOCK_ARRIVAL', 'A')])
    cs = [candidate('A'), candidate('B')]
    assert first(FIFO(), sim, cs) == 'A'
    assert first(ArrivalFIFO(), sim, cs) == 'B'


def test_discharge_uses_real_handover_not_scheduled_release():
    sim = World([(50., 'BLOCK_ARRIVAL', 'B'), (90., 'VESSEL_RELEASED', 'A')])
    assert first(ArrivalFIFO(), sim, [candidate('A'), candidate('B')]) == 'B'
    sim = World([(40., 'JOB_RELEASED', 'A'), (50., 'BLOCK_ARRIVAL', 'B')])
    assert first(ArrivalFIFO(), sim, [candidate('B'), candidate('A')]) == 'A'


def test_unrealized_times_never_supply_service_priority():
    sim = World()
    sim.jobs['A'].actual_block_arrival = 5000.
    p = ArrivalFIFO()
    with pytest.raises(ValueError, match='no observed queue entry'):
        p.rank(sim, 'YC', candidate('A'))
    # Preparations use public notice only, irrespective of hidden future arrival.
    a = p.rank(sim, 'YC', candidate('A', CandidateKind.PRE_REHANDLE))
    sim.jobs['A'].actual_block_arrival = 15000.
    assert p.rank(sim, 'YC', candidate('A', CandidateKind.PRE_REHANDLE)) == a


def test_reject_future_event_in_history():
    with pytest.raises(ValueError, match='future event'):
        ArrivalFIFO().rank(World([(101., 'BLOCK_ARRIVAL', 'A')]), 'YC', candidate('A'))


def test_incremental_events_and_independent_worlds():
    p = ArrivalFIFO()
    a = World([(20., 'BLOCK_ARRIVAL', 'A')])
    assert p.rank(a, 'YC', candidate('A'))[1] == 20.
    b = copy.deepcopy(a)
    b.event_log = [(70., 'BLOCK_ARRIVAL', 'A')]
    assert p.rank(b, 'YC', candidate('A'))[1] == 70.
    a.event_log += [(90., 'JOB_RELEASED', 'B')]
    assert p.rank(a, 'YC', candidate('B'))[1] == 90.
    assert p.rank(a, 'YC', candidate('A'))[1] == 20.


def test_service_fence_ties_and_registration():
    p = make_preference('FCFS_ARRIVAL')
    assert isinstance(p, ArrivalFIFO) and not p.USES_FUTURE_INFORMATION
    sim = World([(80., 'BLOCK_ARRIVAL', 'B'), (80., 'BLOCK_ARRIVAL', 'A')])
    assert first(p, sim, [candidate('B'), candidate('A')]) == 'A'
    assert p.rank(sim, 'YC', candidate('B')) < p.rank(sim, 'YC', candidate('A', CandidateKind.PRE_REHANDLE))
    assert p.rank(sim, 'YC', NS(job_ref=None)) == (2, 0., '')


def test_real_simulator_uses_block_entry_events():
    from test_dispatch_reuse import snapshot, generate
    from yard_rl.v7.world.integrated.baselines import ResolverPolicy, _apply
    sim, dp = snapshot(n=8)
    pref = ArrivalFIFO()
    policy = ResolverPolicy(pref, 'FCFS_ARRIVAL')
    served = []
    while dp is not None:
        assignment = policy.decide(sim, dp, generate(sim, dp))
        for cid, gc in assignment.items():
            if gc.kind == CandidateKind.SERVE:
                served.append(gc.job_ref.job_id)
        _apply(sim, assignment)
        dp = sim.run_until_decision()
    assert len(served) == 8
    assert len(set(served)) == 8
    assert all(j.service_end is not None for j in sim.jobs.values())
