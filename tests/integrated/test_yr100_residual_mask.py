"""YR-100 잔여망 마스크 가드 — 금지 본선 긴급도 feature 가 입력에 부재함을 실제 결정에서 확인."""
from yard_rl.domain.enums import InformationLevel
from yard_rl.experiments.yr090_dense_vessel import _sim
from yard_rl.experiments.yr100_single_block import (_ban_indices, _grp_ban, build_rows_cvec)
from yard_rl.integrated.baselines import JointRolloutGreedy
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.cost_config import RewardCalculator


def test_mask_zeroes_banned_and_keeps_rest():
    sim = _sim("mid-tight", 830_200 + 90)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RewardCalculator.numeraire_v1(), horizon_s=1800.0, generator=gen)
    dp = sim.run_until_decision()
    assert dp is not None
    gen_by = {c: gen.generate(sim, c, InformationLevel.PRE_ADVICE) for c in dp.crane_ids}
    rows, assigns, cvec = build_rows_cvec(sim, dp, gen_by, None, jr, 0)
    assert rows and len(rows) == len(assigns) == len(cvec)
    from yard_rl.contract import SCHEMA
    g2 = 2 * len(SCHEMA.group_specs("global"))
    from yard_rl.integrated.encoding import K_VESSEL
    v2 = K_VESSEL * 2 * len(SCHEMA.group_specs("vessel"))
    yc2 = 2 * len(SCHEMA.group_specs("yc"))
    q2 = 2 * len(SCHEMA.group_specs("queue"))
    c2 = 2 * len(SCHEMA.group_specs("candidate"))
    ban = set(_ban_indices(g2, v2, yc2, q2, c2))
    assert len(rows[0]) == g2 + v2 + 2 * (yc2 + q2 + c2)          # 레이아웃 검산
    for r in rows:
        assert all(r[i] == 0.0 for i in ban)                       # 금지 feature 부재
    # 마스크가 전부를 죽이지 않음 — 남은 feature 에 비영 값 존재 (입력이 살아 있음)
    assert any(any(r[i] != 0.0 for i in range(len(r)) if i not in ban) for r in rows)


def test_ban_covers_spec_fields():
    # 스펙 금지 4종 + 파생 긴급도(queue.vessel_urgency_max·global.sts_wait_accum_s) 커버
    assert _grp_ban("candidate")                                   # is_vessel 등 3필드
    assert _grp_ban("queue")
    assert _grp_ban("global")
