"""YR-106-b 게이트 C(통계)·YR-108(실현 지문)·YR-107(배포 자격) 회귀.

각 테스트는 **정정 전 결함을 재현하는 형태**로 쓴다 — 회귀하면 곧바로 터지도록.
"""
import dataclasses

import pytest

from yard_rl.integrated.baselines import (FIFOPreference, JointRolloutGreedy, ResolverPolicy,
                                          ServiceFirstSPTPreference, is_deployable,
                                          uses_future_information)
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.evalkit import paired, paired_by_channel, required_n
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from yard_rl.integrated.seedbank import (assign_band, independence_report, realization_hash)
from yard_rl.integrated.statfuncs import chi2_ppf, sd_upper_conf, t_ppf

# 공개 t 표 (구 evalkit 하드코딩과 같은 출처) — 계산식이 표를 재현해야 한다
_T95 = {1: 12.706, 2: 4.303, 5: 2.571, 7: 2.365, 10: 2.228, 20: 2.086, 29: 2.045}
_T80 = {5: 0.920, 7: 0.896, 20: 0.860, 29: 0.854}


# ----------------------------------------------------------------- 게이트 C: 통계
def test_t_ppf_matches_published_tables():
    for df, v in _T95.items():
        assert t_ppf(0.975, df) == pytest.approx(v, abs=1e-3)
    for df, v in _T80.items():
        assert t_ppf(0.80, df) == pytest.approx(v, abs=1e-3)


def test_t_table_gaps_are_gone():
    """구 `_T80` 은 df 21·22·24~28 이 없어 0.920(df=5 값)으로 떨어졌다 — 7% 과대."""
    for df in (21, 22, 24, 25, 26, 27, 28):
        v = t_ppf(0.80, df)
        assert 0.85 < v < 0.87, f"df={df} 에서 {v}"
    assert t_ppf(0.975, 41) == pytest.approx(2.0195, abs=1e-3)   # n=42 확증 대역


def test_required_n_iterates_on_target_df():
    """구식은 상수 z(1.96+0.84)로 1회 계산 → 작은 n 에서 과소추정."""
    sd, d = 3.685, 3.0
    n = required_n(sd, d)
    assert n == 14
    df = n - 1
    assert (t_ppf(0.975, df) + t_ppf(0.80, df)) * sd / n ** 0.5 <= d
    df0 = n - 2                                          # n−1 은 조건을 못 지켜야 최소값
    assert (t_ppf(0.975, df0) + t_ppf(0.80, df0)) * sd / (n - 1) ** 0.5 > d


def test_sd_uncertainty_uses_pilot_df_not_target():
    """sd 불확실성은 **파일럿** 자유도에서 온다 — 목표 df 를 쓰면 자기충족적 과소추정."""
    sd, d = 3.685, 3.0
    n_pt = required_n(sd, d)
    n_pilot8 = required_n(sd, d, sd_conf=0.80, sd_df=7)
    n_pilot40 = required_n(sd, d, sd_conf=0.80, sd_df=39)
    assert n_pt < n_pilot40 < n_pilot8, (n_pt, n_pilot40, n_pilot8)
    assert n_pilot8 == 24                     # 정찰 교차검증값
    with pytest.raises(ValueError):
        required_n(sd, d, sd_conf=0.80)       # sd_df 없이는 불가


def test_sd_upper_conf_matches_chi2_identity():
    df = 7
    assert sd_upper_conf(1.0, df, 0.80) == pytest.approx((df / chi2_ppf(0.20, df)) ** 0.5)
    assert sd_upper_conf(1.0, 7, 0.80) > sd_upper_conf(1.0, 40, 0.80) > 1.0


def test_equivalence_requires_tost_not_mde():
    """구 규칙 `MDE ≤ δ` 는 **CI 가 δ 를 넘어도** '효과 없음'을 찍어냈다.

    평균이 δ 근처이고 분산이 작으면 그 거짓 주장이 실제로 발생한다.
    """
    diffs = [0.95, 1.0, 1.05, 0.9, 1.1, 1.0, 0.95, 1.05]      # 평균≈1.0, δ=1.0
    r = paired(diffs, delta_interest=1.0)
    assert r.mde80 <= 1.0                       # 구 규칙이라면 '효과 없음'
    assert r.equivalent is False                # TOST 는 통과하지 못한다
    assert not r.label.startswith("효과 없음")


def test_tost_passes_when_truly_equivalent():
    diffs = [0.05, -0.05, 0.02, -0.03, 0.01, 0.0, -0.02, 0.03]
    r = paired(diffs, delta_interest=1.0)
    assert r.equivalent is True and r.label.startswith("효과 없음")


def test_significant_but_below_delta_has_its_own_label():
    """5번째 상태 — 유의하면서 δ 보다 작다. 구 라벨 4종에는 자리가 없었다."""
    diffs = [-0.6, -0.62, -0.58, -0.61, -0.59, -0.63, -0.6, -0.6]
    r = paired(diffs, delta_interest=1.0)
    assert r.ci_hi < 0 and r.equivalent is True
    assert "유의하나" in r.label


def test_degenerate_channel_is_not_certified_equivalent():
    """가중치 0 채널(전 시드 차이 0)에 '효과 없음' 증명서를 내주면 구조적 거짓 주장."""
    r = paired([0.0] * 8, delta_interest=1.0)
    assert r.label.startswith("판정 대상 아님")


def test_primary_channel_is_recorded_in_output():
    """'1차 판정 = 트럭'이 산문이 아니라 **결과에** 남아야 다중비교 논리가 성립한다."""
    t = [{"truck": 1.0, "vessel": 2.0, "move": 0.0, "other": 0.0, "total": 3.0},
         {"truck": 1.2, "vessel": 2.5, "move": 0.1, "other": 0.0, "total": 3.8}]
    c = [{"truck": 0.5, "vessel": 2.2, "move": 0.0, "other": 0.0, "total": 2.7},
         {"truck": 0.6, "vessel": 2.1, "move": 0.05, "other": 0.0, "total": 2.75}]
    out = paired_by_channel(t, c, primary="truck")
    assert out["truck"]["role"] == "1차(확증)"
    assert out["vessel"]["role"].startswith("탐색적")
    with pytest.raises(ValueError):
        paired_by_channel(t, c, primary="없는채널")


# ----------------------------------------------------------------- YR-108: 실현 지문
def _gen(level, dm, seed, ach=False):
    prof = build_calibrated_profile()
    p = dataclasses.replace(calibrated_load_params(level, vessel_deadline_mult=dm),
                            time_contract_v2=True, gate_block_contract=True,
                            vessel_deadline_achievable=ach)
    return generate_terminal_scenario(prof, seed, p)


def test_deadline_axis_does_not_change_realization():
    """사고의 뿌리 — 마감 배율·정합 플래그는 난수를 소비하지 않는다.

    지문이 **같아야** 이 사고를 탐지할 수 있다(다르면 잡을 방법이 없다).
    """
    hs = {realization_hash(_gen("high", dm, 830_100, ach))
          for dm in (0.40, 0.5, 0.75, 1.0, 2.0) for ach in (False, True)}
    assert len(hs) == 1


def test_reproduces_yr041a_internal_duplication():
    """YR-041-a: high-0.40 과 high-0.75 가 같은 BASE(830100) → 명목 24행 중 고유 16."""
    a = [realization_hash(_gen("high", 0.40, 831_000 + i)) for i in range(8)]
    b = [realization_hash(_gen("high", 0.75, 831_000 + i)) for i in range(8)]
    m = [realization_hash(_gen("mid", 0.40, 830_900 + i)) for i in range(8)]
    assert a == b                                  # 8/8 동일 실현
    assert len(set(a + b + m)) == 16               # 명목 24 → 고유 16


def test_different_level_and_seed_differ():
    assert realization_hash(_gen("high", 0.5, 830_100)) != realization_hash(
        _gen("mid", 0.5, 830_100))
    assert realization_hash(_gen("high", 0.5, 830_100)) != realization_hash(
        _gen("high", 0.5, 830_101))


def test_assign_band_yields_disjoint_realizations():
    cells = {"A": ("high", 0.5), "B": ("mid", 2.0)}

    def gen(key, cell, seed):
        return _gen(cell[0], cell[1], seed)

    b1 = assign_band(family="t1", cells=cells, n=6, generate=gen, start_seed=911_000)
    b2 = assign_band(family="t2", cells=cells, n=6, generate=gen, start_seed=912_000,
                     exclude=b1.all_hashes)
    assert independence_report(b1)["ok"]
    r2 = independence_report(b2, forbidden={"t1": b1.all_hashes})
    assert r2["ok"] and r2["n_unique"] == 12
    assert not (b1.all_hashes & b2.all_hashes)
    assert "digest" in b1.freeze_json() and b1.freeze_json()["seeds"]["A"] == b1.seeds["A"]


def test_independence_report_flags_duplicates():
    from yard_rl.integrated.seedbank import BandSpec
    bad = BandSpec(family="x", seeds={"A": [1, 2], "B": [1, 2]},
                   hashes={"A": ["h1", "h2"], "B": ["h1", "h3"]})
    rep = independence_report(bad)
    assert not rep["ok"] and "h1" in rep["internal_duplicates"]
    assert rep["n_rows"] == 4 and rep["effective_n_upper"] == 3


# ----------------------------------------------------------------- YR-107: 배포 자격
def test_oracle_arms_are_not_deployable():
    assert not is_deployable("JR1800")
    assert not is_deployable("JOINT_ROLLOUT")
    assert not is_deployable("BEAM")
    assert is_deployable("SF")
    assert is_deployable("CALC:88000")
    assert is_deployable("CONTROL:99000")


def test_oracle_detected_on_objects_and_wrappers():
    jr = JointRolloutGreedy(RewardCalculator.numeraire_v1())
    assert uses_future_information(jr) and not is_deployable(jr)
    assert is_deployable(ResolverPolicy(ServiceFirstSPTPreference(), "SF"))
    # FIFO 는 actual_block_arrival 을 now 게이트 없이 읽는다 (약한 누출)
    assert not is_deployable(ResolverPolicy(FIFOPreference(), "FIFO_R"))
