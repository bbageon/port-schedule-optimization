"""블록당 크레인 수 컴포넌트 회귀 테스트 — 선택·게이트·bay 분담·identity."""
from __future__ import annotations

import pytest

from yard_rl.integrated.crane_layout import CraneLayout, crane_layout
from yard_rl.integrated.profiles import build_calibrated_profile

PROF = build_calibrated_profile()      # 2크레인 (YC-L·YC-W, 전 bay)
B = PROF.block.bay_count


def test_selects_crane_count():
    for n in (1, 2, 3, 5):
        p = crane_layout(n, "split").apply(PROF)
        assert len(p.cranes) == n


def test_n2_shared_is_identity():
    """N=2·shared·기존 2크레인 프로파일 → 그대로 반환 (FT golden-safe)."""
    assert crane_layout(2, "shared").apply(PROF) is PROF
    assert [c.crane_id for c in PROF.cranes] == ["YC-L", "YC-W"]


def test_policy_and_faithful_gate():
    assert crane_layout(2, "shared").policy_compatible is True
    assert crane_layout(2, "shared").faithful is True
    for cl in (crane_layout(1), crane_layout(3, "split"), crane_layout(2, "split")):
        assert cl.policy_compatible is False
        assert cl.faithful is False


def test_warnings():
    assert crane_layout(2, "shared").warnings() == ()
    w1 = crane_layout(1).warnings()
    assert w1 and any("2크레인 슬롯 고정" in x for x in w1)
    assert any("단일 크레인" in x for x in w1)
    w3 = crane_layout(3, "split").warnings()
    assert any("용량 스케일링 미충실" in x for x in w3)


def test_split_partitions_cover_block():
    ranges = [(c.service_bay_min, c.service_bay_max)
              for c in crane_layout(3, "split").apply(PROF).cranes]
    assert ranges[0][0] == 1 and ranges[-1][1] == B
    for (a_lo, a_hi), (b_lo, b_hi) in zip(ranges, ranges[1:]):
        assert b_lo == a_hi + 1                      # 비겹침·연속


def test_shared_all_full_range():
    for c in crane_layout(3, "shared").apply(PROF).cranes:
        assert (c.service_bay_min, c.service_bay_max) == (1, B)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        crane_layout(0)
    with pytest.raises(ValueError):
        CraneLayout(2, "diagonal")


def test_composition_with_profile():
    from yard_rl.integrated.congestion import congestion
    from yard_rl.integrated.scenario_gen import generate_terminal_scenario
    p = crane_layout(3, "split").apply(PROF)
    params = congestion("busy").to_gen_params(p)
    assert params.n_external == 10 * 3 * 4           # 크레인수에 비례
    sc = generate_terminal_scenario(p, 770000, params)
    assert len(sc.jobs) > 0
