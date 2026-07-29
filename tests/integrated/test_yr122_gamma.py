"""YR-122 — γ 패치의 유효성(gdt=1.0)과 기본값 보존."""
import random

from yard_rl.experiments import yr088_joint_rl as y88
from yard_rl.experiments import yr090_dense_vessel as y90


def _gdts(seed=830_000, cell="mid-loose"):
    rng = random.Random(7)
    trans, _ = y90.collect_episode(cell, seed, None, None, 1.0, rng, dense_vessel=False)
    return [t[3] for t in trans]


def test_default_gamma_discounts():
    """기본(γ=0.99)에서는 비종결 transition 의 gdt 가 1 미만이어야 한다."""
    assert y90.GAMMA == 0.99 and y88.GAMMA == 0.99
    gdts = _gdts()
    assert any(g < 1.0 - 1e-9 for g in gdts[:-1]), "할인이 전혀 적용되지 않았다"


def test_gamma1_patch_makes_all_gdt_one():
    """y90.GAMMA=1.0 패치(YR-122 처치)면 모든 gdt 가 정확히 1.0 — 비할인 TD."""
    prev = y90.GAMMA
    try:
        y90.GAMMA = 1.0
        gdts = _gdts()
    finally:
        y90.GAMMA = prev
    assert all(abs(g - 1.0) < 1e-12 for g in gdts), f"비할인이 아니다: {gdts[:5]}"
