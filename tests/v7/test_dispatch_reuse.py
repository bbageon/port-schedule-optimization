"""Candidate preservation, invalidation, lifecycle and independent worlds."""
import copy
from dataclasses import replace
import pickle

import pytest
import torch

from yard_rl.v7.job_index import IndexedJobs
from yard_rl.v7.crane.policy import CraneNet, CranePolicy
from yard_rl.v7.stage.exec_config import DEFAULT_EXEC_CONFIG
from yard_rl.v7.world.domain.enums import ContainerSize, JobFlow, JobStatus, LoadStatus, InformationLevel
from yard_rl.v7.world.domain.models import Job, Container
from yard_rl.v7.world.integrated.candidates import CandidateGenerator
from yard_rl.v7.world.integrated.engine import TerminalSimulator
from yard_rl.v7.world.integrated.fixtures import build_integrated_profile, build_minimal_terminal_scenario
from yard_rl.v7.world.integrated.scenario import TerminalScenario
from yard_rl.v7.world.integrated.baselines import ResolverPolicy, run_joint_episode
from yard_rl.v7.world.integrated.cost_config import RewardCalculator
from yard_rl.v7.world.integrated.reservation import Reservation, Corridor


def inbound(jid, at=0, size=ContainerSize.FT20):
    return Job(jid, JobFlow.GATE_IN, at, at, at, inbound_size=size, inbound_load=LoadStatus.FULL)


def snapshot(n=12, history=0):
    p = build_integrated_profile()
    p = replace(p, cranes=(p.cranes[0],))
    jobs = [inbound(f'T{i:06}', size=ContainerSize.FT20 if i % 2 else ContainerSize.FT40)
            for i in range(n)]
    sim = TerminalSimulator(p, TerminalScenario('reuse', 9903321, 8000., 0., {}, jobs, []))
    dp = sim.run_until_decision()
    for i in range(history):
        j = inbound(f'H{i:06}')
        j.status = JobStatus.DONE
        sim.jobs[j.job_id] = j
    return sim, dp


def generate(sim, dp, reuse=True, k=10000):
    gen = CandidateGenerator(k_max=k, reuse_plans=reuse, block_pre_rehandle=True,
                             config=DEFAULT_EXEC_CONFIG)
    return {cid: gen.generate(sim, cid, sim.info_level) for cid in dp.crane_ids}


def assert_index(jobs):
    for status in JobStatus:
        expected = tuple(sorted(jid for jid, j in jobs.items() if j.status == status))
        assert jobs.ids(status) == expected
        assert jobs.count(status) == len(expected)


def test_lifecycle_mutations_move_delete_replace_and_rename():
    a = IndexedJobs(A=inbound('A'), B=inbound('B'))
    for s in JobStatus:
        a['A'].status = s
        assert_index(a)
    b = IndexedJobs()
    b['A'] = a.pop('A')
    b['A'].status = JobStatus.WAITING
    with pytest.raises(ValueError):
        b['A'].job_id = 'RENAMED'
    assert_index(a)
    assert_index(b)
    with pytest.raises(ValueError):
        a['A'] = b['A']
    a['B'] = inbound('B')
    a |= {'C': inbound('C')}
    assert a.setdefault('C', inbound('C')) is a['C']
    a.popitem()
    a.clear()
    b.clear()
    assert_index(a)
    assert_index(b)


@pytest.mark.parametrize('clone', [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))])
def test_clone_rebinds_membership_and_drops_cache(clone):
    sim, dp = snapshot()
    sim.dispatch_cache.rehandle_slots = True
    generate(sim, dp)
    other = clone(sim)
    assert not other.dispatch_cache.states
    assert other.dispatch_cache.rehandle_slots
    other.jobs['T000000'].status = JobStatus.CANCELLED
    assert sim.jobs['T000000'].status == JobStatus.WAITING
    assert_index(sim.jobs)
    assert_index(other.jobs)
    assert generate(other, dp) == generate(other, dp, reuse=False)
    lone = clone(sim.jobs['T000001'])
    lone.status = JobStatus.DONE
    assert sim.jobs['T000001'].status == JobStatus.WAITING


def test_reuse_retimes_plans_and_does_not_reuse_scores():
    sim, dp = snapshot(history=500)
    first = generate(sim, dp)
    before = sim.dispatch_cache.stats.copy()
    sim._advance(11.)
    again = generate(sim, dp)
    assert again == generate(sim, dp, reuse=False)
    assert sim.dispatch_cache.stats['plan_hits'] - before['plan_hits'] == 12
    assert all(g.plan.start_s == 11. for g in again['YC-A'].items if g.plan)
    torch.manual_seed(9903321)
    p = CranePolicy(CraneNet())
    pairs = [('YC-A', g) for g in again['YC-A'].items if g.feasible]
    batched = p.rank_many(sim, pairs)
    scalar = [p.rank(sim, cid, g) for cid, g in pairs]
    for a, b in zip(batched, scalar):
        assert a[::2] == b[::2] and a[1] == pytest.approx(b[1], abs=1e-6)
    assert first != again


@pytest.mark.parametrize('change', ['arrival', 'cancel', 'position', 'yard', 'reservation',
                                    'size', 'custom_hook', 'deadline'])
def test_invalidation_keeps_all_candidate_plans_exact(change):
    sim, dp = snapshot()
    generate(sim, dp)
    before = sim.dispatch_cache.stats.copy()
    if change == 'arrival':
        j = inbound('NEW'); j.status = JobStatus.WAITING
        sim.jobs[j.job_id] = j
        sim.kpis.truck_arrived(j.job_id, sim.clock)
    elif change == 'cancel':
        sim.jobs['T000000'].status = JobStatus.CANCELLED
    elif change == 'position':
        sim.fleet.get('YC-A').state.position_bay += 1
    elif change == 'yard':
        sim.stacks.place(Container('NEW', ContainerSize.FT20, LoadStatus.FULL,
                                  'B1', 1, 1, 1), 1, 1)
    elif change == 'reservation':
        sim.reservations.reserve(Reservation('OTHER', 'OTHER', Corridor(38, 40),
                                             frozenset({(1, 1)}), None, 100))
    elif change == 'size':
        sim.jobs['T000000'].inbound_size = ContainerSize.FT20
    elif change == 'custom_hook':
        sim.store_slot_selector = lambda sim, stk, job, spec, bay, row, exclude: stk.find_slot(
            job.inbound_size, spec, bay, row, exclude=exclude)
    else:
        sim.jobs['T000000'].deadline = 0
    assert generate(sim, dp) == generate(sim, dp, reuse=False)
    if change in ('position', 'yard', 'reservation'):
        assert sim.dispatch_cache.stats['plan_misses'] - before['plan_misses'] == 12
    if change == 'arrival':
        assert sim.dispatch_cache.stats['plan_hits'] - before['plan_hits'] == 12
        assert sim.dispatch_cache.stats['plan_misses'] - before['plan_misses'] == 1
    assert_index(sim.jobs)


@pytest.mark.parametrize('seed', [9903321, 9903322, 9903323])
@pytest.mark.parametrize('serve_first', [True, False])
def test_mixed_continuous_two_crane_episode_equivalence(seed, serve_first):
    torch.manual_seed(seed)
    net = CraneNet()
    results = []
    for fast in (False, True):
        sim = TerminalSimulator(build_integrated_profile(), build_minimal_terminal_scenario(),
                                info_level=InformationLevel.PRE_ADVICE, yard_handover_cap=2)
        if not fast:
            sim.jobs = dict(sim.jobs)
        sim.dispatch_cache.rehandle_slots = fast
        policy = ResolverPolicy(CranePolicy(net, serve_first=serve_first, batched=fast))
        gen = CandidateGenerator(k_max=10000, vessel_prep=True, reuse_plans=fast)
        result = run_joint_episode(sim, policy, RewardCalculator.numeraire_v1(), generator=gen)
        results.append((sim.event_log, result, [(j.job_id, j.status, j.service_start, j.service_end)
                                               for j in sim.jobs.values()]))
        if fast:
            assert_index(sim.jobs)
    assert results[0] == results[1]


def test_force_and_exact_ties_preserve_selection():
    sim, dp = snapshot()
    net = CraneNet()
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    gb = generate(sim, dp)
    assignments = []
    for fast in (False, True):
        p = CranePolicy(net, batched=fast)
        p.force_once['YC-A'] = 'T000005'
        assignments.append(ResolverPolicy(p).decide(sim, dp, gb))
    assert assignments[0] == assignments[1]
    assert assignments[1]['YC-A'].job_ref.job_id == 'T000005'


@pytest.mark.parametrize('change', ['none', 'container_size', 'yard', 'position',
                                    'reservation', 'custom_selector'])
def test_rehandle_slot_reuse_exact_after_state_changes(change):
    sim = TerminalSimulator(build_integrated_profile(), build_minimal_terminal_scenario(),
                            info_level=InformationLevel.PRE_ADVICE)
    sim.dispatch_cache.rehandle_slots = True
    dp = sim.run_until_decision()
    spec = sim.fleet.spec(dp.crane_ids[0])
    cid = dp.crane_ids[0]
    def query(exclude):
        return sim.dispatch_cache.rehandle_slot(sim.stacks, ContainerSize.FT20,
                                                spec, 2., 1., exclude)
    with sim.dispatch_cache.scope(sim, cid):
        for exclude in (frozenset(), frozenset({(1, 1)})):
            assert query(exclude) == query(exclude)
    assert sim.dispatch_cache.stats['slot_hits'] >= 2
    if change == 'container_size':
        c = next(iter(sim.stacks.containers.values()))
        c.size = ContainerSize.FT40 if c.size == ContainerSize.FT20 else ContainerSize.FT20
    elif change == 'yard':
        c = next(iter(sim.stacks.containers.values()))
        top = sim.stacks.stack(c.bay, c.row)[-1]
        sim.stacks.remove(top)
    elif change == 'position':
        sim.fleet.get(cid).state.position_bay += 1
    elif change == 'reservation':
        sim.reservations.reserve(Reservation('OTHER', 'OTHER', Corridor(38, 40),
                                             frozenset({(1, 1)}), None, 100))
    elif change == 'custom_selector':
        sim.slot_selector = lambda *args: None
    with sim.dispatch_cache.scope(sim, cid):
        for exclude in (frozenset(), frozenset({(1, 1)}), frozenset({(2, 1)})):
            assert query(exclude) == sim.stacks.find_slot(ContainerSize.FT20, spec, 2., 1., exclude)
    # Outside candidate generation, assignment always recomputes from live state.
    assert sim.dispatch_cache.active is None
