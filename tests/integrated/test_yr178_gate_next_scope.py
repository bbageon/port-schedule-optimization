"""YR-178 — 게이트의 "지금 허가된 축"을 판정에서 유도한다.

구판은 세 칸이 고정 문자열이었고 `currently_authorized` 는 reliability 하나만
보고 갈렸다. scenario_validity 가 FAIL 이던 시절 문구가 남아, 그 축이 PASS 로
바뀐 뒤에도 계속 "그 축을 보정하라"고 안내했다 — 같은 JSON 안에서
`conditional_sequence` 와 모순됐다(2026-08-17 실측).
"""
from __future__ import annotations

import itertools

import pytest

from yard_rl.experiments.gate_harness import (GATE_DEPENDENCY_ORDER, GateOutcome,
                                              GateStatus, ResearchGateReport,
                                              derive_next_scope)


def _mk(**status: GateStatus) -> ResearchGateReport:
    return ResearchGateReport(
        performance=GateOutcome("performance", status["performance"], ""),
        reliability=GateOutcome("reliability", status["reliability"], ""),
        scenario_validity=GateOutcome("scenario_validity",
                                      status["scenario_validity"], ""))


ALL = (GateStatus.PASS, GateStatus.FAIL, GateStatus.INCONCLUSIVE)


@pytest.mark.parametrize(
    "perf,rel,scn", itertools.product((GateStatus.PASS, GateStatus.FAIL), repeat=3))
def test_eight_combinations_pick_first_unresolved(perf, rel, scn):
    """8조합 — 의존 순서상 **가장 앞선 미해소 축**이 허가 축이어야 한다."""
    rep = _mk(performance=perf, reliability=rel, scenario_validity=scn)
    out = derive_next_scope(rep)
    by = {"performance": perf, "reliability": rel, "scenario_validity": scn}
    expect = [n for n in GATE_DEPENDENCY_ORDER if by[n] is not GateStatus.PASS]
    assert out["unresolved_gates"] == expect
    if expect:
        assert out["currently_authorized"].startswith(expect[0])
        # 이미 통과한 축을 보정하라고 하지 않는다 — 이것이 이 작업의 요지다
        for name, st in by.items():
            if st is GateStatus.PASS:
                assert not out["currently_authorized"].startswith(name)
    else:
        assert "확증" in out["currently_authorized"]


def test_today_state_points_at_performance():
    """2026-08-17 실제 상태 — 구판은 scenario_validity 를 가리켰다."""
    out = derive_next_scope(_mk(performance=GateStatus.INCONCLUSIVE,
                                reliability=GateStatus.PASS,
                                scenario_validity=GateStatus.PASS))
    assert out["unresolved_gates"] == ["performance"]
    assert out["currently_authorized"].startswith("performance")
    assert "scenario_validity" not in out["currently_authorized"]
    assert out["conditional_sequence"] == []      # 뒤에 남은 축이 없다


def test_inconclusive_counts_as_unresolved():
    """INCONCLUSIVE 는 PASS 가 아니다 — 미해소로 센다."""
    out = derive_next_scope(_mk(performance=GateStatus.PASS,
                                reliability=GateStatus.INCONCLUSIVE,
                                scenario_validity=GateStatus.PASS))
    assert out["unresolved_gates"] == ["reliability"]


def test_sequence_lists_remaining_axes_in_order():
    """미해소가 여럿이면 의존 순서대로 뒤 축을 나열한다."""
    out = derive_next_scope(_mk(performance=GateStatus.FAIL,
                                reliability=GateStatus.FAIL,
                                scenario_validity=GateStatus.FAIL))
    assert out["unresolved_gates"] == list(GATE_DEPENDENCY_ORDER)
    assert len(out["conditional_sequence"]) == 2
    assert out["conditional_sequence"][0].startswith("reliability PASS 후")
    assert out["conditional_sequence"][1].startswith("scenario_validity PASS 후")


def test_all_pass_authorizes_confirmation():
    out = derive_next_scope(_mk(performance=GateStatus.PASS,
                                reliability=GateStatus.PASS,
                                scenario_validity=GateStatus.PASS))
    assert out["unresolved_gates"] == []
    assert out["conditional_sequence"] == []
    assert "확증" in out["currently_authorized"]
