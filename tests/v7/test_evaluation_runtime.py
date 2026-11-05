"""Daily evaluation switches preserve physics and keep observers out of training."""
import pytest
import torch

from yard_rl.v7.crane.policy import CraneNet
from yard_rl.v7.evaluation_probe import EvaluationProbe, digest
from yard_rl.v7.stage.episode import EpisodeResult, _rule_policy, run_episode
from yard_rl.v7.stage.rollout import RolloutBudget
from yard_rl.v7.world.domain.enums import InformationLevel
from yard_rl.v7.world.integrated.engine import TerminalSimulator
from yard_rl.v7.world.integrated.fixtures import build_integrated_profile, build_minimal_terminal_scenario
from yard_rl.v7.world.integrated.multiblock import MultiBlockTerminal


def world():
    sim = TerminalSimulator(build_integrated_profile(), build_minimal_terminal_scenario(),
                            info_level=InformationLevel.PRE_ADVICE, yard_handover_cap=2)
    return MultiBlockTerminal({'Y01': sim})


def test_real_stage_policy_reference_batch_and_default_outputs_match():
    torch.manual_seed(9913300)
    net = CraneNet()
    outputs = []
    for mode in ('reference', 'batch', 'optimized'):
        mbt = world()
        if mode != 'optimized':
            sim = mbt.blocks['Y01']
            jobs = dict(sim.jobs)
            sim.jobs.clear()
            sim.jobs = jobs
        probe = EvaluationProbe()
        probe.start({'schedule': []}, mbt)
        policy, exc = _rule_policy('RL_CRANE', crane_net=net,
            batched=mode != 'reference', reuse_plans=mode == 'optimized', evaluation_probe=probe)
        mbt.run(policy)
        probe.finish(mbt, {}, EpisodeResult())
        assert exc == {'n': 0, 'hook': 0}
        assert probe.result['decision']['count'] > 0
        outputs.append({k:v for k,v in probe.result.items() if k.endswith('_sha256')})
    assert outputs[0] == outputs[1] == outputs[2]


def test_probe_failure_is_visible_instead_of_silently_waiting():
    class BrokenProbe:
        def decision(self, *args):
            raise RuntimeError('probe failure')
    policy, _ = _rule_policy(evaluation_probe=BrokenProbe())
    with pytest.raises(RuntimeError, match='probe failure'):
        world().run(policy)


@pytest.mark.parametrize('kwargs', [dict(evaluation_mode='reference'),
    dict(evaluation_mode='batch'), dict(evaluation_probe=EvaluationProbe())])
@pytest.mark.parametrize('teacher', ['budget', 'crane_budget'])
def test_evaluation_switches_cannot_change_teacher_contract(kwargs, teacher):
    with pytest.raises(ValueError, match='teachers disabled'):
        run_episode(load=1, seed=9913300, **kwargs, **{teacher: RolloutBudget(max_labels=1)})


def test_hash_is_independent_of_set_insertion_order():
    assert digest({'stages': {'IN', 'OUT'}}) == digest({'stages': set(['OUT', 'IN'])})


def test_recall_observer_leaves_whole_trajectory_unchanged():
    from yard_rl.v7.candidate_probe import CandidateRecallProbe
    torch.manual_seed(9913422)
    net=CraneNet()
    outputs=[]
    for probe in (EvaluationProbe(),CandidateRecallProbe()):
        mbt=world()
        probe.start({'schedule':[]},mbt)
        policy,exc=_rule_policy('RL_CRANE',crane_net=net,candidate_cap=None,evaluation_probe=probe)
        mbt.run(policy)
        probe.finish(mbt,{},EpisodeResult())
        assert exc=={'n':0,'hook':0}
        outputs.append({k:v for k,v in probe.result.items() if k.endswith('_sha256')})
    assert outputs[0]==outputs[1]
    assert probe.result['recall']['recall']==1.
