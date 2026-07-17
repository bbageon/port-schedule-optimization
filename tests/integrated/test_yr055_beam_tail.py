"""YR-055 — BEAM 2단 lookahead 의 tail 계약.

YR-045 locked run 에서 BEAM 이 JointRollout 과 60/60 seed 완전 동일함을 발견했다.
원인: 첫 window rollout 은 지평 도달 시 미해소 결정(pending) 상태의 scratch 를 반환하는데,
구 `_tail` 은 pending 을 dp=None 으로 간주해 즉시 반환 → 전 분기 tail≈0 → 순위 불변.
여기서는 (1) tail 이 pending 을 넘어 실제 진행하는지, (2) 그 결과 BEAM 이 JointRollout 과
달라질 수 있는지를 고정한다.
"""
from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import (BeamLookahead, CandidateGenerator, JointRolloutGreedy,
                                TerminalSimulator, build_integrated_profile,
                                run_joint_episode)
from yard_rl.integrated.baselines import _feasible_joint, _rollout_cost
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.scenario_gen import generate_terminal_scenario

PROF = build_integrated_profile()
PA = InformationLevel.PRE_ADVICE
RC = RewardCalculator.assumed_default()
SEED = 310000


def test_tail_advances_past_pending_decision():
    """첫 window 종료 시 pending 인 scratch 에서 tail 이 0 으로 조기 반환하면 안 된다."""
    sim = TerminalSimulator(PROF, generate_terminal_scenario(PROF, SEED), info_level=PA)
    beam = BeamLookahead(RC, horizon_s=600.0, width=2)
    dp = sim.run_until_decision()
    gen = CandidateGenerator()
    gen_by = {c: gen.generate(sim, c, PA) for c in dp.crane_ids}
    combo = next(c for c in beam._admissible_combos(sim, dp, gen_by)
                 if _feasible_joint(sim, dict(zip(dp.crane_ids, c))))
    _, scratch = _rollout_cost(sim, dict(zip(dp.crane_ids, combo)), RC,
                               horizon_s=600.0, base_policy=beam.base_policy)
    assert not scratch.terminal and scratch._pending, "전제: window 끝 = 결정 pending"
    tail = beam._tail(scratch)
    assert tail > 0.0, "tail 이 pending 결정에서 즉시 반환 — 2단 lookahead 무효 (YR-055 회귀)"


def test_beam_can_diverge_from_joint_rollout():
    """수정 후 BEAM 은 최소 한 seed 에서 JointRollout 과 다른 결과를 낸다 (동일성 해소).

    (역은 요구하지 않음 — 같은 seed 에서 같아도 됨. 310000~310004 다섯 seed 중 하나라도
    총비용이 다르면 tail 이 순위에 실제로 개입한다는 증거.)
    """
    diverged = False
    for seed in range(310000, 310005):
        rj = run_joint_episode(
            TerminalSimulator(PROF, generate_terminal_scenario(PROF, seed), info_level=PA),
            JointRolloutGreedy(RC, horizon_s=600.0), RC, level=PA)
        rb = run_joint_episode(
            TerminalSimulator(PROF, generate_terminal_scenario(PROF, seed), info_level=PA),
            BeamLookahead(RC, horizon_s=600.0, width=3), RC, level=PA)
        assert rb["completion_rate"] == 1.0
        if abs(rj["total_cost"] - rb["total_cost"]) > 1e-9:
            diverged = True
            break
    assert diverged, "BEAM 이 5개 seed 전부 JointRollout 과 동일 — tail 무효 재발 의심"
