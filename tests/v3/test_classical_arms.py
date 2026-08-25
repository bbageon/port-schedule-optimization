"""고전 규칙 팔 5종이 계약을 지키는가 ([[YR-211]]).

**주판정 축**이므로 여기가 깨지면 논문의 핵심 비교가 무너진다
([06 §3] — 주판정 = 규칙 대비, 안 팔기 대비는 판별력 0).
"""
from __future__ import annotations

import pytest

from yard_rl.v3.actors.classical import ARM_RULES, ClassicalMarket
from yard_rl.v3.reward import reset_rollout_calls, rollout_calls
from yard_rl.v3.stage import run_episode
from yard_rl.v3.stage.episode import ARMS, RULE_ARMS

LOAD, SEED = 600, 9_900_991


def _run(arm, seed=SEED):
    reset_rollout_calls()
    return run_episode(load=LOAD, arm=arm, seed=seed)


def test_all_arms_registered():
    for a in ARM_RULES:
        assert a in ARMS and a in RULE_ARMS


@pytest.mark.parametrize("arm", ARM_RULES)
def test_arm_runs_and_makes_no_rollouts(arm):
    """★판정 경로에서 반사실이 한 번도 안 돌아간다 — 교사 누출 금지."""
    r = _run(arm)
    assert rollout_calls() == 0 and r.rollout_worlds == 0
    assert r.phi_krw > 0 and r.decisions > 0


@pytest.mark.parametrize("arm", ARM_RULES)
def test_rule_arms_never_defer_time(arm):
    """★고전 팔은 **공간 이동만** 한다 (번역표의 목적지가 전부 블록이다).

    시간 이연이 섞이면 RL 과의 행동 폭 비교가 흐려진다 — 명시된 비대칭을 지킨다.
    """
    r = _run(arm)
    assert r.n_time == 0, f"{arm} 이 시간 이연을 했다"


@pytest.mark.parametrize("arm", ARM_RULES)
def test_trigger_limits_trades(arm):
    """★트리거가 실제로 제한한다 — 전건 무차별 재배치가 아니다."""
    r = _run(arm)
    assert 0 < r.traded_edges < r.decisions, (
        f"{arm}: 거래 {r.traded_edges} / 결정 {r.decisions}")


@pytest.mark.parametrize("arm", ARM_RULES)
def test_reproducible(arm):
    """같은 시드는 같은 값 — 규칙에 난수가 섞이면 짝비교가 무너진다."""
    a, b = _run(arm), _run(arm)
    assert a.phi_krw == b.phi_krw and a.traded_edges == b.traded_edges


def test_arms_actually_differ():
    """다섯 팔이 **같은 팔이 아니다** — 하나로 뭉개지면 대조군이 하나뿐인 셈이다."""
    phis = {a: _run(a).phi_krw for a in ARM_RULES}
    assert len(set(phis.values())) >= 3, f"팔들이 구분이 안 된다: {phis}"


def test_no_realloc_makes_no_trades():
    r = _run("NO_REALLOC")
    assert r.traded_edges == 0


def test_classical_market_rejects_unknown_arm():
    with pytest.raises(ValueError):
        ClassicalMarket("RL", layout=None)
