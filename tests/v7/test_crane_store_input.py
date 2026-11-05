"""One additional physical direction input, initialized without changing actions."""
import copy
from dataclasses import replace
import pickle

import pytest
import torch

from yard_rl.v7.crane.features import crane_features
from yard_rl.v7.crane.policy import CraneNet, CranePolicy, add_store_input
from yard_rl.v7.world.domain.enums import JobFlow, ServiceMode
from yard_rl.v7.world.integrated.baselines import ResolverPolicy
from test_candidate_recall import state, generate


@pytest.mark.parametrize('seed',[9913501,9913502,9913503])
def test_zero_extension_preserves_initial_ranks_and_assignment(seed):
    sim, dp = state()
    gb = generate(sim,dp,None)
    torch.manual_seed(seed)
    original = CraneNet(feature_mode='feasible')
    extended = add_store_input(original)
    assert extended.net[0].weight[:,9].count_nonzero() == 0
    pairs = [('YC-A',g) for g in gb['YC-A'].items]
    a,b = CranePolicy(original),CranePolicy(extended)
    for x,y in zip(a.rank_many(sim,pairs),b.rank_many(sim,pairs)):
        assert x[::2] == y[::2]
        assert x[1] == pytest.approx(y[1],abs=1e-7)
    assert ResolverPolicy(a).decide(sim,dp,gb) == ResolverPolicy(b).decide(sim,dp,gb)
    restored = CraneNet.from_checkpoint(dict(crane=extended.state_dict(),model_config=extended.model_config()))
    assert restored.model_config() == extended.model_config()
    assert pickle.loads(pickle.dumps(extended)).include_flow


def test_bit_uses_physical_direction_and_only_adds_one_value():
    sim,dp = state()
    gc=generate(sim,dp,None)['YC-A'].items[0]
    j=sim.jobs[gc.job_ref.job_id]
    for vessel in (False, True):
        g=replace(gc,job_ref=replace(gc.job_ref,is_vessel=vessel,is_external=not vessel))
        for store in (False,True):
            j.target_container = None if store else 'OUT'
            j.flow = JobFlow.VESSEL_DISCHARGE if vessel else JobFlow.GATE_IN
            if not store:
                j.inbound_size = None
            else:
                from yard_rl.v7.world.domain.enums import ContainerSize
                j.inbound_size = ContainerSize.FT20
            row=crane_features(sim,'YC-A',g,feature_mode='feasible',include_flow=True)
            old=crane_features(sim,'YC-A',g,feature_mode='feasible')
            assert len(row)==10 and row[:9]==old
            assert row[0]==float(vessel)
            assert row[9]==float(j.service_mode==ServiceMode.STORE)==float(store)


@pytest.mark.parametrize('seed',[9913501,9913502,9913503])
def test_zero_extension_preserves_mixed_episode(seed):
    from yard_rl.v7.world.domain.enums import InformationLevel
    from yard_rl.v7.world.integrated.engine import TerminalSimulator
    from yard_rl.v7.world.integrated.fixtures import build_integrated_profile, build_minimal_terminal_scenario
    from yard_rl.v7.world.integrated.baselines import run_joint_episode
    from yard_rl.v7.world.integrated.candidates import CandidateGenerator
    from yard_rl.v7.world.integrated.cost_config import RewardCalculator
    torch.manual_seed(seed)
    source=CraneNet(feature_mode='feasible')
    results=[]
    for net in (source,add_store_input(source)):
        sim=TerminalSimulator(build_integrated_profile(),build_minimal_terminal_scenario(),
                              info_level=InformationLevel.PRE_ADVICE,yard_handover_cap=2)
        result=run_joint_episode(sim,ResolverPolicy(CranePolicy(net)),RewardCalculator.numeraire_v1(),
                                 generator=CandidateGenerator(k_max=None,vessel_prep=True))
        results.append((sim.event_log,result))
    assert results[0]==results[1]
