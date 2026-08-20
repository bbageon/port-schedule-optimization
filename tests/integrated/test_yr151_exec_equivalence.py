"""YR-151 — 채택 실행 정책 연결의 **행동 일치** 테스트 (감사 요구 2026-08-09).

"체크포인트가 로드됐다"만으로는 부족하다 — 같은 상태에서 기존 채택 판정 경로
(YR-146 조립: RLPolicy + DeployGuard)와 새 SELL 실험 경로(AdoptedExecFleet)가
**같은 실행 순서를 선택**해야 제대로 연결된 것이다. 결정 궤적 전체를 비교한다.
"""
from pathlib import Path

import pytest

CKPT = Path("outputs/reports/yr143_no_repo/confirm/c0/ppo_s221000/net.pt")
NORM = Path("outputs/reports/yr125_diff_credit/diff1_s88000/rl_net.pt")

pytestmark = pytest.mark.skipif(
    not (CKPT.is_file() and NORM.is_file()),
    reason="채택 실행 체크포인트가 로컬에 없음 — 일치 테스트는 체크포인트 필요")

SEED = 6_500_000


def _sig(assign: dict) -> tuple:
    """결정 1건의 서명 — 크레인별 (행동 종류, 대상 작업)."""
    out = []
    for c, g in sorted(assign.items()):
        jid = getattr(getattr(g, "job_ref", None), "job_id", None)
        out.append((c, str(getattr(g, "kind", None)), str(jid)))
    return tuple(out)


def _trace(policy_builder) -> list[tuple]:
    from yard_rl.experiments.yr088_joint_rl import LEVEL
    from yard_rl.experiments.yr100_candidate_eval import RC_EVAL
    from yard_rl.experiments.yr149_load_cells import _sim_from
    from yard_rl.integrated.baselines import run_joint_episode
    from yard_rl.integrated.candidates import CandidateGenerator
    from yard_rl.integrated.load_cells import make_a_cell
    from yard_rl.integrated.policy_config import ADOPTED_C0_GUARD

    sim = _sim_from(make_a_cell(SEED, 50))
    sim.info_level = LEVEL
    pol = policy_builder(sim)
    rec: list[tuple] = []

    class Tracer:
        name = "trace"

        def decide(self, sim_, dp, gb):
            a = pol.decide(sim_, dp, gb)
            rec.append(_sig(a))
            return a

    run_joint_episode(
        sim, Tracer(), RC_EVAL,
        generator=CandidateGenerator(config=ADOPTED_C0_GUARD))
    return rec


def test_fleet_matches_judgment_assembly():
    """동일 시드·동일 상태에서 두 조립 경로의 결정 궤적이 완전히 같아야 한다."""
    from yard_rl.experiments.yr088_joint_rl import RLPolicy
    from yard_rl.experiments.yr100_candidate_eval import RC_EVAL
    from yard_rl.experiments.yr146_deploy_guard import DeployGuard
    from yard_rl.experiments.yr151_transfer_ppo import (AdoptedExecFleet,
                                                       load_adopted_execution_head)
    from yard_rl.integrated.baselines import JointRolloutGreedy
    from yard_rl.integrated.candidates import CandidateGenerator
    from yard_rl.integrated.policy_config import ADOPTED_C0_GUARD

    actor, norm = load_adopted_execution_head()

    def old_assembly(sim):                    # YR-146 판정 경로 그대로
        jr = JointRolloutGreedy(RC_EVAL, horizon_s=1800.0,
                                generator=CandidateGenerator(
                                    config=ADOPTED_C0_GUARD),
                                forbid_strategic_wait=True)
        return DeployGuard(RLPolicy(actor, norm, name="exec:c0"), actor, norm, jr)

    fleet = AdoptedExecFleet(actor, norm, config=ADOPTED_C0_GUARD)

    trace_old = _trace(old_assembly)
    trace_new = _trace(lambda sim: fleet.get(sim))
    assert trace_old, "결정이 하나도 기록되지 않음 — 하네스 결함"
    assert trace_old == trace_new, (
        f"조립 경로 간 결정 불일치 — 첫 분기: "
        f"{next((i, a, b) for i, (a, b) in enumerate(zip(trace_old, trace_new)) if a != b)}")


def test_invalid_exec_head_rejected():
    """'adopted'/'sf' 외 값은 즉시 거절 — 오타로 SF 가 몰래 돌지 못하게 (감사)."""
    from yard_rl.experiments.yr151_transfer_ppo import load_kf, run_episode
    from yard_rl.v1.ppo_policy import PpoSellPolicy, TransferActor, TransferCritic
    pol = PpoSellPolicy(TransferActor(), TransferCritic(), mode="live", seed=0)
    with pytest.raises(ValueError):
        run_episode(1, pol, load_kf(), exec_head="adotped")   # 오타 재현


def test_adopted_exec_config_must_be_explicit():
    from yard_rl.experiments.yr151_transfer_ppo import load_kf, run_episode
    from yard_rl.v1.ppo_policy import PpoSellPolicy, TransferActor, TransferCritic
    pol = PpoSellPolicy(TransferActor(), TransferCritic(), mode="live", seed=0)
    with pytest.raises(ValueError, match="explicit exec_config"):
        run_episode(1, pol, load_kf(), exec_head="adopted")
