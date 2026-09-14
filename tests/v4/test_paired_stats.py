"""짝비교 통계가 정확한가 ([[YR-314]] · `eval/paired.py`).

scipy 없이 순수 파이썬으로 짠 정확검정이 **손으로 센 값**과 맞는지 본다.
"""
from __future__ import annotations

import pytest

from yard_rl.v4.eval.paired import (boot_ci, paired_summary, sign_test,
                                    wilcoxon_exact)


def test_sign_test_all_wins():
    """28일 전부 이기면 p = 2 × 2^-28 — 논문 [[YR-233]] 의 28/28 과 같은 셈."""
    r = sign_test([-1.0] * 28)
    assert r["win"] == 28 and r["n"] == 28
    assert abs(r["p"] - 2 * 0.5 ** 28) < 1e-15


def test_sign_test_drops_ties_and_is_two_sided():
    r = sign_test([-1.0, -1.0, 0.0, 1.0, 0.0])
    assert r["ties"] == 2 and r["n"] == 3 and r["win"] == 2
    # 3번 중 2번: P(X ≤ 1) × 2 = (1+3)/8 × 2 = 1.0
    assert abs(r["p"] - 1.0) < 1e-12
    assert sign_test([1.0] * 5)["p"] == sign_test([-1.0] * 5)["p"]


def test_wilcoxon_small_hand_computed():
    """n=3 · 전부 음수 → W+ = 0. 정확분포에서 P(W+ ≤ 0) = 1/8 → 양측 0.25."""
    r = wilcoxon_exact([-3.0, -1.0, -2.0])
    assert r["n"] == 3 and r["w_plus"] == 0.0
    assert abs(r["p"] - 0.25) < 1e-12


def test_wilcoxon_uses_magnitude():
    """부호는 같은데 크기가 다르면 p 가 달라야 한다 — 부호검정과의 차이."""
    a = wilcoxon_exact([-10.0, -10.0, -10.0, -10.0, +1.0])
    b = wilcoxon_exact([-1.0, -1.0, -1.0, -1.0, +10.0])
    assert a["p"] < b["p"], "큰 승리 넷 + 작은 패배 하나가 더 유의해야 한다"
    assert sign_test([-10.0, -10.0, -10.0, -10.0, +1.0])["p"] == \
        sign_test([-1.0, -1.0, -1.0, -1.0, +10.0])["p"]


def test_wilcoxon_ties_in_magnitude_use_midranks():
    """|차이| 동점은 평균순위 — 터지지 않고 [0,1] 안의 p 를 낸다."""
    r = wilcoxon_exact([-2.0, -2.0, 2.0, -1.0])
    assert 0.0 < r["p"] <= 1.0


def test_bootstrap_ci_brackets_mean_and_is_reproducible():
    d = [-5.0, -3.0, -4.0, -6.0, 1.0, -2.0]
    a = boot_ci(d, n_boot=2_000)
    b = boot_ci(d, n_boot=2_000)
    assert a == b, "같은 시드면 같은 구간이어야 한다"
    assert a["lo"] <= a["mean"] <= a["hi"]
    assert a["p_below0"] > 0.9


def test_stratified_bootstrap_keeps_strata():
    d = [-1.0, -1.0, +50.0, +50.0]
    plain = boot_ci(d, n_boot=2_000)
    strat = boot_ci(d, ["a", "a", "b", "b"], n_boot=2_000)
    # 층화면 층마다 되뽑아 극단 조합이 덜 나온다 → 구간이 더 좁다
    assert (strat["hi"] - strat["lo"]) < (plain["hi"] - plain["lo"])


def test_paired_summary_aligns_days():
    a = {0: 90.0, 1: 80.0, 2: 70.0, 5: 1.0}       # 5 는 상대에 없다
    b = {0: 100.0, 1: 100.0, 2: 100.0, 9: 1.0}
    s = paired_summary(a, b, {0: "x", 1: "x", 2: "y"})
    assert s["days"] == [0, 1, 2] and s["n"] == 3
    assert s["sign"]["win"] == 3
    assert abs(s["median_ratio"] - (-0.2)) < 1e-12


@pytest.mark.parametrize("n", [5, 12, 28])
def test_wilcoxon_symmetric_under_sign_flip(n):
    d = [-(i + 1) * 1.5 for i in range(n)]
    assert abs(wilcoxon_exact(d)["p"] - wilcoxon_exact([-x for x in d])["p"]) < 1e-12
