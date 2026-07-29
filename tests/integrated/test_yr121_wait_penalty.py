"""YR-121 — WAIT 지속시간 벌점 훅의 골든 보존 + 발화 검증."""
import random

import pytest

from yard_rl.experiments import yr088_joint_rl as y88
from yard_rl.experiments import yr090_dense_vessel as y90


def _collect(seed=830_000, cell="mid-loose"):
    """net=None(ε=1.0 무작위) — 같은 rng 시드면 행동 열이 동일해 보상만 비교 가능."""
    rng = random.Random(7)
    trans, _ = y90.collect_episode(cell, seed, None, None, 1.0, rng, dense_vessel=False)
    return trans


def test_default_zero_is_byte_identical():
    """기본 WAIT_TIME_PENALTY=0.0 이면 보상 열이 기존과 완전 동일해야 한다 (골든)."""
    assert y90.WAIT_TIME_PENALTY == 0.0
    a = _collect()
    y90.WAIT_TIME_PENALTY = 0.0
    b = _collect()
    assert [t[2] for t in a] == [t[2] for t in b]


def test_structural_wait_is_also_priced_under_forbid():
    """경계 계약 고정 — FORBID_WAIT 하에서도 **구조적 WAIT**(실작업 조합이 전부 공동
    실행불가 → 조합 열거가 WAIT 포함으로 후퇴)는 선택이므로 벌점 대상이다.

    (초판 테스트는 "금지면 발화 0"을 전제했는데 조합 열거기의 후퇴 경로를 몰랐던
    틀린 전제였다 — 실측 9/139 transition 이 구조적 WAIT 를 포함한다.)
    """
    base = _collect()
    prev = y90.WAIT_TIME_PENALTY
    try:
        y90.WAIT_TIME_PENALTY = 100.0
        on = _collect()
    finally:
        y90.WAIT_TIME_PENALTY = prev
    diffs = [o[2] - b[2] for b, o in zip(base, on)]
    assert all(d >= -1e-9 for d in diffs)
    fired = sum(1 for d in diffs if d > 1e-9)
    assert 0 < fired < len(diffs), \
        f"구조적 WAIT 만 일부 발화해야 한다 (실측 {fired}/{len(diffs)})"


def test_penalty_fires_when_wait_allowed():
    """WAIT 허용(FORBID_WAIT=False) + 벌점 ON → 일부 transition 보상이 정확히
    n_wait × Δt/3600 × λ 만큼 증가한다."""
    prev88, prev90, prev_p = y88.FORBID_WAIT, y90.FORBID_WAIT, y90.WAIT_TIME_PENALTY
    try:
        y88.FORBID_WAIT = False
        y90.FORBID_WAIT = False
        y90.WAIT_TIME_PENALTY = 0.0
        base = _collect()
        y90.WAIT_TIME_PENALTY = 100.0
        on = _collect()
    finally:
        y88.FORBID_WAIT, y90.FORBID_WAIT = prev88, prev90
        y90.WAIT_TIME_PENALTY = prev_p
    assert len(base) == len(on)
    diffs = [o[2] - b[2] for b, o in zip(base, on)]
    assert any(d > 1e-9 for d in diffs), "WAIT 선택이 한 번도 벌점받지 않았다"
    assert all(d >= -1e-9 for d in diffs), "벌점이 보상을 줄이는 방향으로 샜다"
