"""YR-059 — 상태 scale-only 정규화 계약.

고정하는 것: (1) norm 미지정 시 기존 인코딩과 완전 동일(회귀 0), (2) scale-only —
결측=0 보존·부호 보존·±clip, (3) fit 은 결정론·양수·val/test 미접촉 구조,
(4) 정규화는 인코딩 전용 — 결정 경로(zero-init 점수)와 계약 record 는 불변.
"""
import pytest

torch = pytest.importorskip("torch")

from yard_rl.contract import SCHEMA, build_feature_vector
from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import TerminalSimulator, build_integrated_profile
from yard_rl.integrated.dqn_learner import run_episode
from yard_rl.integrated.encoding import StateNorm, fv_to_vec
from yard_rl.integrated.qnet import QPreference
from yard_rl.integrated.scenario_gen import generate_terminal_scenario
from yard_rl.experiments.yr013_qmix_experiment import quick_yr013_config, _params
from yard_rl.experiments.yr059_state_norm import fit_state_norm

PROF = build_integrated_profile()
PA = InformationLevel.PRE_ADVICE


def _cand_fv(**over):
    raw = {"action_kind_idx": 1 / 3, "is_external": 1.0, "is_vessel": 0.0,
           "predicted_arrival_gap_s": -1800.0, "reach_s": 120.0,
           "expected_service_time_s": 50.0, "expected_handling_count": 1.0,
           "blocker_count": 1.0, "expected_rehandle_time_s": 40.0, "end_bay": 20.0,
           "lane_congestion_local": 0.0, "interference_penalty_s": 0.0,
           "resequence_count": 0.0, "contention_risk": 0.5}
    raw.update(over)
    return build_feature_vector("candidate", raw, now=500.0, info_level=PA)


def test_no_norm_is_bitwise_legacy():
    fv = _cand_fv()
    assert fv_to_vec(fv) == fv_to_vec(fv, None)


def test_scale_only_preserves_missing_sign_and_clips():
    fv = _cand_fv()
    names = list(fv.names)
    norm = StateNorm(refs={}, clip=5.0)          # 스키마 assumed 사용 (S=3600 등)
    vec = fv_to_vec(fv, norm)
    n = len(names)
    # 부호 보존: 연착 gap -1800s / 3600 = -0.5
    i = names.index("predicted_arrival_gap_s")
    assert vec[i] == pytest.approx(-0.5)
    # 결측(known=0)은 0 그대로 (예: cum_wait_s 미도착)
    j = names.index("cum_wait_s")
    assert fv.known[j] is False and vec[j] == 0.0
    # bay 좌표: 20/40 = 0.5
    assert vec[names.index("end_bay")] == pytest.approx(0.5)
    # 클리핑: reach 120s 에 극단 기준 1e-2 를 주면 +5 에서 절단
    tight = StateNorm(refs={"candidate.reach_s": 0.01}, clip=5.0)
    assert fv_to_vec(fv, tight)[names.index("reach_s")] == pytest.approx(5.0)
    # known 지시자 채널은 정규화와 무관
    assert vec[n:] == fv_to_vec(fv)[n:]


def test_schema_norm_refs_all_positive_and_in_descriptor():
    from yard_rl.contract.schema import schema_descriptor
    for sp in SCHEMA.specs:
        assert sp.norm_ref > 0.0, f"{sp.group}.{sp.name}: norm_ref 양수 위반"
    desc = schema_descriptor()
    assert all("norm_ref" in f for f in desc["fields"])


def test_fit_is_deterministic_and_positive():
    cfg = quick_yr013_config()
    params = _params(cfg)
    seeds = cfg.train_seeds[:2]
    n1, d1 = fit_state_norm(PROF, params, seeds, progress=lambda s: None)
    n2, _ = fit_state_norm(PROF, params, seeds, progress=lambda s: None)
    assert n1.refs == n2.refs and n1.basis == "fitted_baseline_p90"
    assert all(v > 0.0 for v in n1.refs.values())
    # 표본이 실제로 잡힌 대표 필드 — 기준값이 스키마 assumed 가 아닌 실측로 대체됨
    assert "candidate.reach_s" in n1.refs and d1["candidate.reach_s"]["n"] > 0


def test_norm_changes_encoding_but_not_zero_init_decisions():
    """정규화는 망 입력만 바꾼다 — zero-init(전 후보 0점) 정책의 결정·총비용은 불변."""
    cfg = quick_yr013_config()
    params = _params(cfg)
    norm = StateNorm(refs={}, clip=5.0)
    res = {}
    for label, sn in (("off", None), ("on", norm)):
        sim = TerminalSimulator(PROF, generate_terminal_scenario(PROF, 310000, params),
                                info_level=PA)
        res[label] = run_episode(sim, level=PA, preference=QPreference(), state_norm=sn)
    assert res["on"].total_cost == res["off"].total_cost
    assert res["on"].extras["action_counts"] == res["off"].extras["action_counts"]
