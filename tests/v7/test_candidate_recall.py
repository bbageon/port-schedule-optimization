"""Uncapped membership has an independent small-state oracle."""
from dataclasses import replace

import pytest
import torch

from yard_rl.v7.candidate_probe import CandidateRecallProbe, action_key
from yard_rl.v7.crane.policy import CraneNet, CranePolicy
from yard_rl.v7.stage.exec_config import DEFAULT_EXEC_CONFIG
from yard_rl.v7.world.contract.schema import CandidateKind
from yard_rl.v7.world.domain.enums import ContainerSize, JobFlow, JobStatus, LoadStatus
from yard_rl.v7.world.domain.models import Job
from yard_rl.v7.world.integrated.candidates import CandidateGenerator
from yard_rl.v7.world.integrated.engine import TerminalSimulator
from yard_rl.v7.world.integrated.fixtures import build_integrated_profile
from yard_rl.v7.world.integrated.scenario import TerminalScenario


def state():
    profile = build_integrated_profile()
    profile = replace(profile, cranes=(profile.cranes[0],))
    jobs = [Job(f'T{i:03}', JobFlow.GATE_IN, 0., 0., 0.,
                inbound_size=ContainerSize.FT20, inbound_load=LoadStatus.FULL) for i in range(30)]
    sim = TerminalSimulator(profile, TerminalScenario('recall', 9913421, 8000., 0., {}, jobs, []))
    return sim, sim.run_until_decision()


def generate(sim, dp, cap, observer=None):
    gen = CandidateGenerator(k_max=cap, block_pre_rehandle=True,
                             config=DEFAULT_EXEC_CONFIG, observer=observer)
    return {cid: gen.generate(sim, cid, sim.info_level) for cid in dp.crane_ids}


def test_all_candidates_match_independent_ready_work_enumeration():
    sim, dp = state()
    sim.jobs['T000'].status = JobStatus.CANCELLED
    sim.jobs['T001'].status = JobStatus.PLANNED
    sim.jobs['T001'].provided_eta = 10.
    sim.jobs['T001'].actual_block_arrival = 100.
    all_g = generate(sim, dp, None)['YC-A']
    served = {g.job_ref.job_id for g in all_g.items if g.feasible and g.kind == CandidateKind.SERVE}
    assert served == {f'T{i:03}' for i in range(2, 30)}
    assert served == {r.job_id for r in sim.candidates_for('YC-A')}
    assert all_g == generate(sim, dp, 10000)['YC-A']
    sim.jobs['T001'].actual_block_arrival = 1000.
    assert all_g == generate(sim, dp, None)['YC-A']


def test_cap_recall_and_mandatory_overflow_are_separate():
    sim, dp = state()
    p = CandidateRecallProbe()
    gb = generate(sim, dp, 12, p)
    torch.manual_seed(9913421)
    pref = CranePolicy(CraneNet())
    p.before_select(sim, dp, gb, pref)
    assert p.counts['eligible'] == 30 and p.counts['retained'] == 11
    uncapped = generate(sim, dp, None)
    assert len(uncapped['YC-A'].items) == 31
    sim._advance(7200.)
    capped = generate(sim, dp, 12)['YC-A']
    full = generate(sim, dp, None)['YC-A']
    mandatory = {action_key(g) for g in full.items if g.mandatory}
    assert mandatory and mandatory <= {action_key(g) for g in capped.items}


@pytest.mark.parametrize('cap', [0, -1, True, 1.5])
def test_bad_cap_rejected(cap):
    with pytest.raises(ValueError):
        CandidateGenerator(k_max=cap)
