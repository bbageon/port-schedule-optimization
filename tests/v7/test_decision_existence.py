"""Existence queries stop early; policy candidate enumeration stays complete."""
import copy
import pickle

import pytest
import torch

from test_candidate_recall import state
from yard_rl.v7.crane.policy import CraneNet, CranePolicy
from yard_rl.v7.world.domain.enums import InformationLevel
from yard_rl.v7.world.integrated.baselines import ResolverPolicy, run_joint_episode
from yard_rl.v7.world.integrated.candidates import CandidateGenerator
from yard_rl.v7.world.integrated.cost_config import RewardCalculator
from yard_rl.v7.world.integrated.engine import TerminalSimulator
from yard_rl.v7.world.integrated.fixtures import build_integrated_profile, build_minimal_terminal_scenario


def test_query_stops_after_feasibility_and_leaves_full_api_unchanged(monkeypatch):
    sim,dp=state()
    calls=[]
    original=sim._plan
    def plan(*a,**kw):
        calls.append(a)
        return original(*a,**kw)
    monkeypatch.setattr(sim,'_plan',plan)
    full=sim.candidates_for('YC-A')
    assert len(full)==30 and len(calls)==30
    calls.clear()
    assert sim.candidates_for('YC-A',first_only=True)==full[:1]
    assert len(calls)==1
    assert sim.candidates_for('YC-A')==full


def test_custom_selectors_disable_short_circuit(monkeypatch):
    sim,dp=state()
    sim.decision_probe_shortcircuit=True
    modes=[]
    original=sim.candidates_for
    def query(*a,**kw):
        modes.append(kw['first_only'])
        return original(*a,**kw)
    monkeypatch.setattr(sim,'candidates_for',query)
    sim._decision_cranes()
    assert modes==[True]
    modes.clear()
    sim.slot_selector=lambda *a:None
    sim._decision_cranes()
    assert modes==[False]


@pytest.mark.parametrize('seed',[9913490,9913491,9913492])
def test_mixed_episode_matches_full_existence_scan(seed):
    torch.manual_seed(seed)
    net=CraneNet()
    runs=[]
    for fast in (False,True):
        sim=TerminalSimulator(build_integrated_profile(),build_minimal_terminal_scenario(),
                              info_level=InformationLevel.PRE_ADVICE,yard_handover_cap=2)
        sim.decision_probe_shortcircuit=fast
        assert copy.deepcopy(sim).decision_probe_shortcircuit==fast
        assert pickle.loads(pickle.dumps(sim)).decision_probe_shortcircuit==fast
        result=run_joint_episode(sim,ResolverPolicy(CranePolicy(net)),RewardCalculator.numeraire_v1(),
                                 generator=CandidateGenerator(k_max=12,vessel_prep=True))
        runs.append((sim.event_log,result))
    assert runs[0]==runs[1]
