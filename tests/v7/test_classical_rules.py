"""Ranking definitions, observed direction, and actual simulator integration."""
from types import SimpleNamespace as NS

import pytest

from yard_rl.v7.dispatch import (DistanceWait, Nearest, SPT,
                                Unidirectional, make_preference)
from yard_rl.v7.world.contract.schema import CandidateKind
from yard_rl.v7.world.integrated.fixtures import build_integrated_profile
from yard_rl.v7.world.sim.travel_time import gantry_m


RULES = ('FCFS_ARRIVAL', 'SPT', 'NEAREST', 'VESSEL_FCFS', 'VESSEL_NEAREST',
         'UNIDIRECTIONAL', 'DISTANCE_WAIT')


class World:
    def __init__(self):
        self.clock = 1000.
        self.profile = build_integrated_profile()
        self.position = {'A': 5., 'B': 20.}
        self.fleet = NS(get=lambda c: NS(state=NS(position_bay=self.position[c])),
                        spec=lambda c: self.profile.cranes[0])
        self.event_log = []
        self.jobs = {}


def item(jid, target, *, vessel=False, duration=100., kind=CandidateKind.SERVE):
    return NS(kind=kind, job_ref=NS(job_id=jid, is_vessel=vessel, is_external=not vessel),
              plan=NS(moves=(NS(src=(target, 1, 1)),), end_bay=target,
                      corridor=(5., target), duration_s=duration))


def first(p, w, cs, cid='A'):
    return min(cs, key=lambda c: p.rank(w, cid, c)).job_ref.job_id


def test_nearest_measures_destination_not_reservation_or_duration():
    w = World()
    near, far = item('Z-near', 10, duration=900), item('A-far', 25, duration=10)
    assert Nearest().key(w, 'A', near, near.job_ref) == 5
    assert Nearest().key(w, 'A', far, far.job_ref) == 20
    assert first(Nearest(), w, [far, near]) == 'Z-near'
    assert first(SPT(), w, [far, near]) == 'A-far'
    # The touched interval may include a relocation on the other side.
    near.plan.corridor = (-20., 30.)
    assert first(Nearest(), w, [far, near]) == 'Z-near'


def test_nearest_uses_first_handling_not_final_rehandle_destination():
    w = World()
    c = item('prep', 10, kind=CandidateKind.PRE_REHANDLE)
    c.plan.end_bay = 30.
    assert Nearest().key(w, 'A', c, c.job_ref) == 5
    c.plan.moves = ()
    assert Nearest().key(w, 'A', c, c.job_ref) == 25


@pytest.mark.parametrize('rule', ('VESSEL_FCFS', 'VESSEL_NEAREST'))
def test_vessel_priority_has_service_fence_and_orders_within_class(rule):
    w = World()
    a, b, t = item('old-vessel', 12, vessel=True), item('new-vessel', 18, vessel=True), item('truck', 5)
    w.event_log = [(0., 'BLOCK_ARRIVAL', 'truck'), (100., 'VESSEL_RELEASED', 'old-vessel'),
                   (200., 'JOB_RELEASED', 'new-vessel')]
    p = make_preference(rule)
    assert first(p, w, [t, b, a]) == 'old-vessel'
    a.kind = b.kind = CandidateKind.PRE_REHANDLE
    w.jobs = {c.job_ref.job_id: NS(release_time=0.) for c in (a,b)}
    assert first(p, w, [a,b,t]) == 'truck'


def test_sweep_is_rank_order_independent_and_reverses_after_real_movement():
    w = World()
    p = Unidirectional()
    left, right = item('left', 4), item('right', 10)
    assert first(p,w,[left,right]) == first(p,w,[right,left]) == 'right'
    assert first(p,w,[left]) == 'left'  # A proposal is not completed movement.
    assert first(p,w,[left,right]) == 'right'
    w.position['A'] = 4.  # The selected left job has now actually completed.
    cs = [item('further-left',2),item('right',5)]
    assert first(p,w,cs) == 'further-left'
    assert first(p,w,list(reversed(cs))) == 'further-left'
    # Other crane and world still start facing increasing bay numbers.
    w.position['B'] = 4.
    assert first(p,w,cs,'B') == 'right'
    other = World(); other.position['A'] = 4.
    assert first(p,other,cs) == 'right'


def test_distance_wait_can_protect_old_far_job_in_seconds():
    w = World()
    p = DistanceWait()
    near, far = item('near',6),item('far',15)
    w.event_log = [(0., 'BLOCK_ARRIVAL', 'far'), (999., 'JOB_RELEASED', 'near')]
    assert first(Nearest(),w,[near,far]) == 'near'
    assert first(p,w,[near,far]) == 'far'
    travel = gantry_m(w.profile.block,5.,15.) / w.profile.cranes[0].gantry_speed_mps
    assert p.key(w,'A',far,far.job_ref) == pytest.approx(travel-1000.)
    # Equal observed arrivals reduce the rule to approach distance.
    other = World(); other.event_log = [(900.,'BLOCK_ARRIVAL','far'),(900.,'JOB_RELEASED','near')]
    assert first(p,other,[far,near]) == 'near'


@pytest.mark.parametrize('rule', RULES)
def test_all_rules_complete_real_jobs_without_mutating_candidates(rule):
    from test_dispatch_reuse import snapshot, generate
    from yard_rl.v7.world.integrated.baselines import ResolverPolicy, _apply
    sim, dp = snapshot(n=8)
    p = make_preference(rule)
    assert p.USES_FUTURE_INFORMATION is False
    policy = ResolverPolicy(p,rule)
    served = []
    while dp is not None:
        gb = generate(sim,dp)
        before = repr(gb)
        assignment = policy.decide(sim,dp,gb)
        assert repr(gb) == before
        served.extend(g.job_ref.job_id for g in assignment.values() if g.kind==CandidateKind.SERVE)
        _apply(sim,assignment)
        dp = sim.run_until_decision()
    assert len(served)==len(set(served))==8
    assert all(j.service_end is not None for j in sim.jobs.values())
