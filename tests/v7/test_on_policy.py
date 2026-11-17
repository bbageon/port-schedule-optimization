from types import SimpleNamespace as NS

import pytest
import torch

from yard_rl.v7.crane.actor_critic import GraphState, SchedulingActorCritic, pack_states
from yard_rl.v7.crane.on_policy import (DecisionSample, OnPolicyResolver, action_statistics,
                                        ppo_update)


torch.set_num_threads(1)


def state(n, seed=1):
    rng = torch.Generator().manual_seed(seed)
    return GraphState(torch.randn(n, 10, generator=rng), torch.arange(n),
                       torch.arange(n) % 2, .4)


@pytest.mark.parametrize('kind', ['mlp', 'transformer'])
def test_permutation_padding_checkpoint_gradient(kind, tmp_path):
    torch.manual_seed(10)
    net = SchedulingActorCritic(kind).eval()
    a, b = state(33), state(50, 2)
    logits, values = net(pack_states([a]))
    batched, bv = net(pack_states([a, b]))
    assert torch.allclose(logits, batched[:1, :33], atol=2e-6)
    assert torch.allclose(values, bv[:1], atol=2e-6)
    perm = torch.randperm(33)
    rearranged = GraphState(a.features[perm], a.operations[perm], a.cranes[perm], .4)
    lp, vp = net(pack_states([rearranged]))
    assert torch.allclose(lp, logits[:, perm], atol=2e-6)
    assert torch.allclose(vp, values, atol=2e-6)
    (logits.square().mean() + values.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    net.save(tmp_path/'net.pt')
    restored = SchedulingActorCritic.load(tmp_path/'net.pt').eval()
    assert all(torch.equal(a, b) for a, b in zip(net(pack_states([a])), restored(pack_states([a]))))


def test_all_1200_candidates_kept():
    net = SchedulingActorCritic().eval()
    with torch.no_grad():
        score, value = net(pack_states([state(1200)]))
    assert score.shape == (1, 1200) and torch.isfinite(score).all()
    assert torch.isfinite(value).all()


def test_conditional_joint_probability_is_product():
    s = DecisionSample(state(4), [[0, 1], [2, 3]], [1, 2], 0., 0., 0.)
    logits = torch.tensor([[0., 1., 3., 2.]], requires_grad=True)
    lp, entropy = action_statistics(logits, [s])
    expected = torch.log_softmax(logits[0, :2], 0)[1] + torch.log_softmax(logits[0, 2:], 0)[0]
    assert torch.allclose(lp[0], expected)
    assert entropy[0] > 0


def test_joint_duplicate_and_priority_masks():
    from yard_rl.v7.world.contract.schema import CandidateKind as K
    def gc(i, token, mandatory=False, kind=K.SERVE):
        return NS(candidate_id=i, kind=kind, feasible=True, mandatory=mandatory,
                  job_ref=NS(token=token) if token else None)
    a, b = gc(0, 'shared', True), gc(1, None, kind=K.WAIT)
    sim = NS(clock=0., dry_run_commit=lambda assignments: NS(plans={c:1 for c,r in assignments.items() if r}))
    resolver = OnPolicyResolver(SchedulingActorCritic())
    # A mandatory operation wins deterministically; B cannot execute it twice.
    result = resolver.resolve(sim, NS(crane_ids=['A', 'B'], time=0.),
                              {'A':NS(items=[a,b]),'B':NS(items=[gc(0,'shared'),b])})
    assert [r.action for r in result.resolutions] == [K.SERVE, K.WAIT]
    assert resolver.last_assignment == {'A':0, 'B':1}


@pytest.mark.parametrize('kind', ['mlp', 'transformer'])
def test_ppo_increases_repeated_rewarded_action(kind):
    torch.manual_seed(5)
    net = SchedulingActorCritic(kind).eval()
    s = state(2)
    with torch.no_grad():
        logits, value = net(pack_states([s]))
        logp = torch.log_softmax(logits[0], 0)
    corpus = [DecisionSample(s, [[0,1]], [i%2], float(logp[i%2]), float(value[0]), 0.,
                              1. if i%2 == 0 else -1.) for i in range(64)]
    before = float(logits[0,0]-logits[0,1])
    info = ppo_update(net, torch.optim.Adam(net.parameters(), lr=.001), corpus,
                      seed=12, epochs=4, batch_size=16)
    after = net(pack_states([s]))[0][0]
    assert float(after[0]-after[1]) > before
    assert info['replay_logp_max_error'] < 2e-5


@pytest.mark.parametrize('kind', ['mlp', 'transformer'])
def test_actual_episode_reward_accounting_and_action_replay(kind):
    from yard_rl.v7.stage.episode import run_episode
    torch.manual_seed(17)
    net = SchedulingActorCritic(kind).eval()
    resolver = OnPolicyResolver(net, stochastic=True, seed=18, collect=True)
    ep = run_episode(load=100, seed=9950001, dispatcher='RL_CRANE', crane_net=net,
                     joint_resolver=resolver, candidate_cap=None, reuse_rehandle_slots=True,
                     decision_probe_shortcircuit=True)
    info = resolver.finish(ep.phi_krw, ep.phi_krw)
    assert info['samples'] > 0 and info['reward_sum'] == pytest.approx(-1.)
    assert ep.admitted == 100 and ep.policy_exceptions == ep.rollout_worlds == ep.traded_edges == 0
    update = ppo_update(net, torch.optim.Adam(net.parameters(), lr=.0003), resolver.samples,
                        seed=19, epochs=1, max_samples=64, batch_size=16)
    assert update['selected'] == min(64, len(resolver.samples))


def test_on_policy_errors_do_not_become_wait():
    from yard_rl.v7.stage.episode import _rule_policy
    from yard_rl.v7.world.integrated.fixtures import build_integrated_profile, build_minimal_terminal_scenario
    from yard_rl.v7.world.integrated.engine import TerminalSimulator
    class Broken:
        def resolve(self, *args):
            raise RuntimeError('deliberate')
    sim = TerminalSimulator(build_integrated_profile(), build_minimal_terminal_scenario())
    decision = sim.run_until_decision()
    fn, _ = _rule_policy(joint_resolver=Broken())
    with pytest.raises(RuntimeError, match='deliberate'):
        fn(sim, decision)


def test_multicrane_physical_execution_matches_recorded_actions():
    from yard_rl.v7.world.integrated.fixtures import build_integrated_profile, build_minimal_terminal_scenario
    from yard_rl.v7.world.integrated.engine import TerminalSimulator
    from yard_rl.v7.world.integrated.candidates import CandidateGenerator
    from yard_rl.v7.world.domain.enums import InformationLevel
    torch.manual_seed(27)
    net=SchedulingActorCritic().eval()
    resolver=OnPolicyResolver(net,stochastic=True,collect=True,seed=28)
    sim=TerminalSimulator(build_integrated_profile(),build_minimal_terminal_scenario())
    generator=CandidateGenerator(k_max=None)
    sim.info_level=InformationLevel.PRE_ADVICE
    while (decision:=sim.run_until_decision()) is not None:
        gb={c:generator.generate(sim,c,sim.info_level) for c in decision.crane_ids}
        result=resolver.resolve(sim,decision,gb)
        selected={c:gb[c].items[i] for c,i in resolver.last_assignment.items()}
        plans=sim.dry_run_commit({c:g.job_ref for c,g in selected.items()}).plans
        resolver.apply(sim,result,gb)
        for c,plan in plans.items():
            assert sim._active_plans[c].duration_s==plan.duration_s
            assert sim._active_plans[c].slots==plan.slots
    sim.check_invariants()
    assert sim.unfinished_backlog()==0
    assert resolver.samples
    for s in resolver.samples:
        with torch.no_grad():
            logits,_=net(pack_states([s.state]))
            lp,_=action_statistics(logits,[s])
        assert float(lp[0])==pytest.approx(s.old_log_probability,abs=2e-6)
